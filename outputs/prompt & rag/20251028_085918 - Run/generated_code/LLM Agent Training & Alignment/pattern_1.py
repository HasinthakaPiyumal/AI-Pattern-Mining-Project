
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from accelerate import Accelerator

# --- Configuration --- #
MODEL_NAME = "gpt2"  # Using a smaller model for demonstration purposes
REWARD_MODEL_NAME = "bert-base-uncased" # A simple BERT model for reward modeling
BEHAVIOR_CLONING_MODEL_PATH = "./behavior_cloned_model"
REWARD_MODEL_PATH = "./reward_model"
RLHF_MODEL_PATH = "./rlhf_model"

NUM_BEHAVIOR_CLONING_EPOCHS = 3
NUM_REWARD_MODEL_EPOCHS = 3
NUM_RLHF_EPOCHS = 2
N_REJECTION_SAMPLES = 5

# --- 1. Data Simulation/Loading Functions --- #
# In a real application, these would load actual data from files or databases.

def load_demonstration_data():
    """Simulates loading human demonstration data."""
    data = [
        {"query": "My internet is not working.", "response": "Please try restarting your router and modem. If the issue persists, let me know."},
        {"query": "I want to change my plan.", "response": "Sure, I can help with that. What kind of plan are you interested in?"},
        {"query": "How do I reset my password?", "response": "You can reset your password by visiting our website and clicking on 'Forgot Password' link."},
    ]
    return Dataset.from_list(data)

def load_preference_data():
    """Simulates loading human preference comparison data."""
    # (query, response_A, response_B, preferred_response_index)
    data = [
        {"query": "My internet is not working.", "response_A": "Try restarting your router.", "response_B": "Please try restarting your router and modem. If the issue persists, let me know.", "preferred": 1},
        {"query": "I want to change my plan.", "response_A": "What plan?", "response_B": "Sure, I can help with that. What kind of plan are you interested in?", "preferred": 1},
        {"query": "How do I reset my password?", "response_A": "Go to settings.", "response_B": "You can reset your password by visiting our website and clicking on 'Forgot Password' link.", "preferred": 1},
    ]
    return Dataset.from_list(data)

# --- 2. Behavior Cloning Module --- #
def train_behavior_cloning_model(tokenizer, base_model, demonstration_data):
    """Trains the LLM using behavior cloning on human demonstrations."""
    print("\n--- Starting Behavior Cloning Training ---")

    def tokenize_function(examples):
        # Prepend query to response for training
        inputs = [f"Customer: {q}\nAgent: {r}" for q, r in zip(examples["query"], examples["response"])]
        return tokenizer(inputs, truncation=True, padding="max_length", max_length=128)

    tokenized_datasets = demonstration_data.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.rename_columns({"input_ids": "labels"})
    tokenized_datasets = tokenized_datasets.remove_columns(["query", "response", "token_type_ids", "attention_mask"])

    training_args = TrainingArguments(
        output_dir="./results_bc",
        num_train_epochs=NUM_BEHAVIOR_CLONING_EPOCHS,
        per_device_train_batch_size=2,
        save_steps=500,
        save_total_limit=2,
        logging_dir="./logs_bc",
        logging_steps=10,
        report_to="none"
    )

    trainer = Trainer(
        model=base_model,
        args=training_args,
        train_dataset=tokenized_datasets,
    )
    trainer.train()
    base_model.save_pretrained(BEHAVIOR_CLONING_MODEL_PATH)
    tokenizer.save_pretrained(BEHAVIOR_CLONING_MODEL_PATH)
    print(f"Behavior cloned model saved to {BEHAVIOR_CLONING_MODEL_PATH}")
    return base_model

# --- 3. Reward Modeling Module --- #
class RewardModel(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(model_name) # Using CausalLM for simplicity
        self.score_head = torch.nn.Linear(self.base_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # Use the last hidden state of the last token as input to the scoring head
        last_hidden_state = outputs.hidden_states[-1][:, -1, :]
        return self.score_head(last_hidden_state)

def train_reward_model(tokenizer, preference_data):
    """Trains a reward model to predict preference scores."""
    print("\n--- Starting Reward Model Training ---")
    reward_model = RewardModel(MODEL_NAME)
    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reward_model.to(device)

    optimizer = torch.optim.Adam(reward_model.parameters(), lr=1e-5)
    loss_fn = torch.nn.MarginRankingLoss(margin=0.1) # Use MarginRankingLoss for preference comparisons

    def prepare_preference_batch(batch):
        # For each pair (response_A, response_B), the preferred one gets a score of 1, the other -1.
        # This is a simplification; a more robust RM would learn direct scores.
        inputs_A = [f"Customer: {q}\nAgent: {r}" for q, r in zip(batch["query"], batch["response_A"])]
        inputs_B = [f"Customer: {q}\nAgent: {r}" for q, r in zip(batch["query"], batch["response_B"])]

        tokenized_A = tokenizer(inputs_A, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
        tokenized_B = tokenizer(inputs_B, truncation=True, padding="max_length", max_length=128, return_tensors="pt")

        rewards_target = torch.tensor([1 if p == 1 else -1 for p in batch["preferred"]], dtype=torch.float32).to(device)
        return tokenized_A, tokenized_B, rewards_target

    for epoch in range(NUM_REWARD_MODEL_EPOCHS):
        for i in range(0, len(preference_data), 2): # Simple batching
            batch_data = preference_data[i:i+2]
            if not batch_data["query"]:
                continue

            tokenized_A, tokenized_B, rewards_target = prepare_preference_batch(batch_data)
            tokenized_A = {k: v.to(device) for k, v in tokenized_A.items()}
            tokenized_B = {k: v.to(device) for k, v in tokenized_B.items()}

            optimizer.zero_grad()

            score_A = reward_model(**tokenized_A).squeeze(-1)
            score_B = reward_model(**tokenized_B).squeeze(-1)

            # If preferred is 1, A is better, else B is better. So if preferred is 1, score_A > score_B
            # For MarginRankingLoss: x1, x2, y. If y=1, x1 is preferred. If y=-1, x2 is preferred.
            # So if preferred[idx] is 1, then score_B, score_A, 1. If preferred[idx] is -1, then score_A, score_B, 1.
            # A simpler way is to train it to predict the score directly, but for preference, this is common.
            loss = loss_fn(score_A, score_B, rewards_target) # If preferred=1 -> score_A preferred, else score_B preferred

            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Epoch {epoch}, Step {i}, Loss: {loss.item():.4f}")

    torch.save(reward_model.state_dict(), REWARD_MODEL_PATH)
    print(f"Reward model saved to {REWARD_MODEL_PATH}")
    return reward_model


# --- 4. Reinforcement Learning from Human Feedback (RLHF) Module --- #
def train_rlhf_model(tokenizer, behavior_cloned_model, reward_model, preference_data):
    """Fine-tune the behavior-cloned LLM using RLHF."""
    print("\n--- Starting RLHF Training ---")

    # Load the behavior cloned model with a value head for PPO
    model = AutoModelForCausalLMWithValueHead.from_pretrained(BEHAVIOR_CLONING_MODEL_PATH)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(BEHAVIOR_CLONING_MODEL_PATH)

    # Move models to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    ref_model.to(device)
    reward_model.to(device)

    ppo_config = PPOConfig(
        learning_rate=1e-5,
        batch_size=2,
        mini_batch_size=1,
        gradient_accumulation_steps=1,
        remove_unused_columns=False, # Important for PPO
        ppo_epochs=NUM_RLHF_EPOCHS,
        seed=0,
        log_with="none"
    )

    # Prepare a dataset for PPO. In a real scenario, this would be new queries.
    # For demonstration, we'll use some queries from the preference data.
    ppo_dataset_text = [f"Customer: {d['query']}\nAgent:" for d in preference_data]
    ppo_dataset = Dataset.from_dict({"query": ppo_dataset_text})

    def tokenize_ppo_query(examples):
        return tokenizer(examples["query"], truncation=True, padding="max_length", max_length=64)

    ppo_dataset = ppo_dataset.map(tokenize_ppo_query, batched=True)
    ppo_dataset.set_format(type="torch")

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=ppo_dataset,
    )

    def reward_fn(responses):
        """Calculates reward for a list of responses using the reward model."""
        # Tokenize responses and get scores
        tokenized_responses = tokenizer(responses, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
        tokenized_responses = {k: v.to(device) for k, v in tokenized_responses.items()}
        scores = reward_model(**tokenized_responses).squeeze(-1).detach()
        return scores

    for epoch in range(ppo_config.ppo_epochs):
        for batch in ppo_trainer.dataloader:
            query_tensors = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Generate responses
            response_tensors = ppo_trainer.generate(
                query_tensors,
                do_sample=True,
                max_new_tokens=64,
                eos_token_id=tokenizer.eos_token_id,
            )
            # Decode responses for reward model input
            responses = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]

            # Calculate rewards
            rewards = reward_fn(responses)

            # Train PPO step
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)

            print(f"Epoch {epoch}, PPO Loss: {stats['ppo/loss/total']:.4f}, Mean Reward: {rewards.mean().item():.4f}")

    model.save_pretrained(RLHF_MODEL_PATH)
    tokenizer.save_pretrained(RLHF_MODEL_PATH)
    print(f"RLHF-tuned model saved to {RLHF_MODEL_PATH}")
    return model

# --- 5. Rejection Sampling (Best-of-N) Module --- #
def generate_n_responses(model, tokenizer, query, n=N_REJECTION_SAMPLES):
    """Generates N candidate responses for a given query."""
    input_text = f"Customer: {query}\nAgent:"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        input_ids,
        do_sample=True,
        num_return_sequences=n,
        max_new_tokens=64,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        top_k=50, # Example sampling parameters
        top_p=0.95,
        temperature=0.7
    )
    # Decode and clean responses (remove input prompt from generated text)
    responses = []
    for gen_id in generated_ids:
        decoded_response = tokenizer.decode(gen_id, skip_special_tokens=True)
        # Remove the input query part from the generated response
        start_index = decoded_response.find("Agent:") + len("Agent:")
        responses.append(decoded_response[start_index:].strip())
    return responses

def select_best_response(model, tokenizer, reward_model, query, responses):
    """Selects the best response from candidates using the reward model."""
    if not responses:
        return ""

    # Prepare inputs for the reward model
    reward_inputs = [f"Customer: {query}\nAgent: {res}" for res in responses]
    tokenized_reward_inputs = tokenizer(reward_inputs, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
    tokenized_reward_inputs = {k: v.to(reward_model.base_model.device) for k, v in tokenized_reward_inputs.items()}

    with torch.no_grad():
        scores = reward_model(**tokenized_reward_inputs).squeeze(-1)

    best_response_index = torch.argmax(scores).item()
    return responses[best_response_index]

# --- 6. Main Orchestration Function --- #
def run_customer_support_agent_pipeline():
    print("Initializing AI Customer Support Agent Pipeline...")

    # Initialize Tokenizer and Base Model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Add a pad token if the tokenizer doesn't have one (common for GPT-2)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'}) 
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    # Resize token embeddings if pad token was added
    if '[PAD]' in tokenizer.get_added_vocab():
        base_model.resize_token_embeddings(len(tokenizer))

    # 1. Load Data
    demonstration_data = load_demonstration_data()
    preference_data = load_preference_data()

    # 2. Behavior Cloning
    behavior_cloned_model = train_behavior_cloning_model(tokenizer, base_model, demonstration_data)

    # 3. Reward Modeling
    reward_model = train_reward_model(tokenizer, preference_data)

    # 4. RLHF
    rlhf_model = train_rlhf_model(tokenizer, behavior_cloned_model, reward_model, preference_data)

    print("\n--- AI Customer Support Agent Ready for Inference (with Rejection Sampling) ---")
    # Example Inference
    test_query = "I have a billing dispute."
    print(f"Customer Query: {test_query}")

    candidate_responses = generate_n_responses(rlhf_model, tokenizer, test_query, N_REJECTION_SAMPLES)
    print("Candidate Responses:")
    for i, res in enumerate(candidate_responses):
        print(f"  {i+1}. {res}")

    final_response = select_best_response(rlhf_model, tokenizer, reward_model, test_query, candidate_responses)
    print(f"\nAI Agent (Best-of-{N_REJECTION_SAMPLES} selected): {final_response}")

    print("\n--- Pipeline Completed ---")

if __name__ == "__main__":
    run_customer_support_agent_pipeline()
