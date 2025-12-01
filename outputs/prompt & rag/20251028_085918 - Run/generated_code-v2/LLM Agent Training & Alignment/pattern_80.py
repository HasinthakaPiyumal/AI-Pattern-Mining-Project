import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from trl import PPOTrainer, PPOConfig
from datasets import Dataset
import random


# --- 1. Language Model (LLM) Setup ---
# Using a small, pre-trained model for demonstration
LLM_MODEL_NAME = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
# Add a pad token if the model doesn't have one, crucial for batching
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'}) 
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')

llm_model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME)
llm_model.resize_token_embeddings(len(tokenizer))


# --- 2. Reward Model (RM) Definition ---
# A simplified Reward Model that takes embeddings and outputs a scalar score.
# In a real scenario, this would be trained on human preference data.
class RewardModel(nn.Module):
    def __init__(self, hidden_size=768):
        super().__init__()
        self.encoder = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME)
        # Freeze encoder for simplicity, in reality it might be fine-tuned or a separate model
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Take the embedding output from the encoder's last hidden state
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_mask=None):
        # Use the encoder to get context-aware embeddings
        # Taking the hidden state of the last token as a representation
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        # Get the representation of the last token (or average pooling, etc.)
        # Simple approach: take the last non-padded token's hidden state
        sequence_output = last_hidden_state[:, -1, :]
        reward = self.reward_head(sequence_output)
        return reward

reward_model = RewardModel()


# --- 3. Dummy Human Feedback Data Generation (for RM training concept) ---
# In a real application, this data would come from human annotators.
# We simulate pairs of responses where one is preferred over the other.
def generate_dummy_feedback_data(num_samples=100):
    data = []
    prompts = [
        "What is your return policy?",
        "How do I track my order?",
        "Can I change my shipping address?",
        "What payment methods do you accept?",
        "Do you offer international shipping?"
    ]
    good_responses = [
        "Our return policy allows returns within 30 days of purchase for a full refund. Please ensure the item is in its original condition.",
        "You can track your order using the tracking number provided in your shipping confirmation email. Just enter it on our 'Track Order' page.",
        "To change your shipping address, please contact our customer support immediately with your order number. We'll do our best to update it before shipment.",
        "We accept major credit cards (Visa, MasterCard, Amex), PayPal, and Apple Pay for your convenience.",
        "Yes, we offer international shipping to many countries. Shipping costs and delivery times vary by destination. Please see our shipping policy for details."
    ]
    bad_responses = [
        "No returns after 30 days.",
        "Check your email.",
        "Maybe, if it's not shipped.",
        "Credit cards only.",
        "Sometimes we ship overseas."
    ]

    for _ in range(num_samples):
        prompt = random.choice(prompts)
        preferred = random.choice(good_responses)
        rejected = random.choice(bad_responses)
        data.append({"prompt": prompt, "chosen": preferred, "rejected": rejected})
    return data

dummy_feedback_data = generate_dummy_feedback_data()

# --- Conceptual RM Training Loop (simplified) ---
# In a real scenario, you'd load a dataset, define a loss function (e.g., cross-entropy or sigmoid loss
# for pairwise preferences), and optimize the RewardModel.
# Here, we just show the structure.
class PairwiseRewardLoss(nn.Module):
    def forward(self, chosen_rewards, rejected_rewards):
        # Binary Cross-Entropy like loss for preferences
        return -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()


# --- 4. Reinforcement Learning with Human Feedback (RLHF - PPO) ---
# Initialize the PPO Trainer
ppo_config = PPOConfig(
    model_name=LLM_MODEL_NAME,
    learning_rate=1e-5,
    batch_size=4,
    forward_batch_size=4,
    # For this example, we'll run a very short, conceptual training
    num_train_epochs=1
)

# Create a dummy dataset for PPO fine-tuning. This would normally contain prompts.
ppo_dataset_dict = Dataset.from_dict({"query": ["Tell me about product X", "How do I reset my password?"]})

# Define a reference model (a frozen copy of the initial LLM)
ref_model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME)
ref_model.resize_token_embeddings(len(tokenizer))

# The PPO trainer expects a reward function. This function will call our reward_model.
def get_rewards(texts):
    # Tokenize the texts
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Get rewards from our reward_model
    with torch.no_grad():
        rewards = reward_model(input_ids, attention_mask)
    return [r for r in rewards.squeeze().tolist()] # Convert to list of floats

# Initialize PPOTrainer
ppo_trainer = PPOTrainer(
    ppo_config,
    llm_model,
    ref_model,
    tokenizer,
    ppo_dataset_dict,
    reward_fn=get_rewards,
)

# Conceptual PPO training loop
# In a real scenario, this would involve iterative generation, reward calculation, and optimization.
# for epoch, batch in enumerate(ppo_trainer.dataloader):
#     query_tensors = batch["input_ids"]
#     response_tensors = ppo_trainer.generate(query_tensors, return_prompt=False, length_sampler=ppo_trainer.response_length_sampler)
#     texts = [tokenizer.decode(r.squeeze()) for r in response_tensors]
#     rewards = get_rewards(texts)
#     stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
#     ppo_trainer.log_stats(stats, batch, rewards)


# --- 5. Chatbot Inference using RLHF-tuned model (or Rejection Sampling) ---
# For simplicity, we'll use a text generation pipeline.
# In a full RLHF setup, `llm_model` would be the fine-tuned model after the PPO training loop.
chatbot_pipeline = pipeline(
    "text-generation",
    model=llm_model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1 # Use GPU if available
)

# Function to generate a response using the (conceptually) RLHF-tuned model
def generate_rlhf_response(prompt, max_new_tokens=50):
    # The pipeline uses the currently loaded llm_model, which would be the RLHF-tuned one
    # after actual training.
    response = chatbot_pipeline(prompt, max_new_tokens=max_new_tokens, num_return_sequences=1, 
                                 pad_token_id=tokenizer.eos_token_id)[0]['generated_text']
    # Remove the prompt from the response
    return response[len(prompt):].strip()

# Function for conceptual Rejection Sampling
def generate_rejection_sampling_response(prompt, num_samples=5, max_new_tokens=50):
    candidate_responses = []
    for _ in range(num_samples):
        response = chatbot_pipeline(prompt, max_new_tokens=max_new_tokens, num_return_sequences=1,
                                     pad_token_id=tokenizer.eos_token_id)[0]['generated_text']
        candidate_responses.append(response[len(prompt):].strip())

    # Score candidates with the reward model
    if not candidate_responses:
        return ""
    
    # Tokenize all candidate responses at once
    inputs = tokenizer(candidate_responses, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        rewards = reward_model(input_ids, attention_mask).squeeze()
    
    # Select the response with the highest predicted reward
    best_response_idx = torch.argmax(rewards)
    return candidate_responses[best_response_idx]


# --- Main Chatbot Interaction Loop --- 
if __name__ == "__main__":
    print("Welcome to the AI-Powered Customer Support Chatbot!")
    print("Type 'exit' to end the conversation.")

    # Example of conceptual RM training (only a forward pass here, no actual optimization)
    print("\n--- Reward Model (Conceptual Training/Evaluation) ---")
    dummy_prompts_for_rm = [d['prompt'] for d in dummy_feedback_data[:2]]
    dummy_chosen_responses = [d['chosen'] for d in dummy_feedback_data[:2]]
    dummy_rejected_responses = [d['rejected'] for d in dummy_feedback_data[:2]]

    chosen_inputs = tokenizer(dummy_chosen_responses, return_tensors="pt", padding=True, truncation=True)
    rejected_inputs = tokenizer(dummy_rejected_responses, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        chosen_rewards = reward_model(chosen_inputs["input_ids"], chosen_inputs["attention_mask"])
        rejected_rewards = reward_model(rejected_inputs["input_ids"], rejected_inputs["attention_mask"])
    
    print(f"Example chosen rewards: {chosen_rewards.squeeze().tolist()}")
    print(f"Example rejected rewards: {rejected_rewards.squeeze().tolist()}")
    print("Reward Model would be optimized to make chosen_rewards > rejected_rewards.")
    print("--------------------------------------------------\n")

    # This part would conceptually run the PPO training
    print("\n--- Conceptual PPO Training Initialization --- ")
    print("PPO Trainer initialized. In a real scenario, 'ppo_trainer.train()' would be called here.")
    print("--------------------------------------------------\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break

        # Choose between RLHF-tuned response or Rejection Sampling based on preference
        # For demonstration, let's always use rejection sampling after RM is 