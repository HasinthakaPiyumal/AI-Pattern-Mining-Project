import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, Trainer, TrainingArguments
from trl import PPOTrainer, PPOConfig
import pandas as pd
import numpy as np
import os

# Ensure reproducibility for demonstration
torch.manual_seed(42)
np.random.seed(42)

# --- 1. Data Pipelines and Processing ---
class CustomerSupportDataset(Dataset):
    """
    A custom Dataset for customer support interactions.
    Handles both behavior cloning data and preference data.
    """
    def __init__(self, data_type="bc", tokenizer=None, bc_data=None, preference_data=None):
        self.data_type = data_type
        self.tokenizer = tokenizer
        if data_type == "bc":
            self.data = bc_data
        elif data_type == "preference":
            self.data = preference_data
        else:
            raise ValueError("Invalid data_type. Must be 'bc' or 'preference'.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        if self.data_type == "bc":
            # For Behavior Cloning: prompt + completion
            text = item["prompt"] + self.tokenizer.eos_token + item["completion"]
            encodings = self.tokenizer(text, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
            return {key: val.squeeze() for key, val in encodings.items()}
        elif self.data_type == "preference":
            # For Preference Data: query, response_A, response_B, label
            # We want to tokenize (query + response_A) and (query + response_B) for reward model training
            combined_A_text = item["query"] + self.tokenizer.eos_token + item["response_A"]
            combined_B_text = item["query"] + self.tokenizer.eos_token + item["response_B"]

            enc_A = self.tokenizer(combined_A_text, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
            enc_B = self.tokenizer(combined_B_text, truncation=True, padding="max_length", max_length=128, return_tensors="pt")

            return {
                "input_ids_A": enc_A["input_ids"].squeeze(),
                "attention_mask_A": enc_A["attention_mask"].squeeze(),
                "input_ids_B": enc_B["input_ids"].squeeze(),
                "attention_mask_B": enc_B["attention_mask"].squeeze(),
                "label": torch.tensor(item["label"], dtype=torch.long) # 0 for A preferred, 1 for B preferred
            }

# --- Dummy Data Generation ---
def generate_dummy_data(num_samples=100):
    bc_data = []
    preference_data = []

    for i in range(num_samples):
        # Behavior Cloning data
        customer_query = f"Customer query {i}: I have an issue with order {1000+i}."
        agent_response = f"Agent response {i}: No problem, let me check order {1000+i} for you. How can I assist further?"
        bc_data.append({"prompt": customer_query, "completion": agent_response})

        # Preference data
        query = f"Query {i}: My product arrived damaged."
        response_good = f"Response Good {i}: I'm so sorry to hear that! Please provide your order number and a photo of the damage, and we'll arrange a replacement or refund immediately."
        response_bad = f"Response Bad {i}: Damaged product? What's your order number? We can't do anything without it."
        # Randomly assign which response is preferred (0 for A, 1 for B)
        if np.random.rand() > 0.5:
            preference_data.append({"query": query, "response_A": response_good, "response_B": response_bad, "label": 0})
        else:
            preference_data.append({"query": query, "response_A": response_bad, "response_B": response_good, "label": 1})

    return pd.DataFrame(bc_data), pd.DataFrame(preference_data)

# --- 2. Core Language Model (LLM) & Tokenizer setup ---
LLM_MODEL_NAME = "gpt2" # Using a small model for demonstration
REWARD_MODEL_NAME = "bert-base-uncased" # Using a small model for demonstration

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token # Or set a specific pad token

# --- 3. Reward Model ---
class RewardModel(torch.nn.Module):
    def __init__(self, model_name=REWARD_MODEL_NAME):
        super().__init__()
        # AutoModelForSequenceClassification with num_labels=1 for scalar output
        self.encoder = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

    def forward(self, input_ids, attention_mask):
        # The output of AutoModelForSequenceClassification is a sequence classifier output object
        # We take the logits as the reward score
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask).logits

# --- 4. Training Modules ---

# Helper function for data collation for BC
class DataCollatorForLanguageModeling:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        # examples is a list of dicts from __getitem__
        batch = self.tokenizer.pad(examples, return_tensors="pt")
        # Shift tokens for causal language modeling objective
        batch["labels"] = batch["input_ids"].clone()
        return batch

# --- Behavior Cloning (Initial Skill Acquisition) ---
def train_behavior_cloning(llm_model, tokenizer, bc_data, output_dir="./bc_model"):
    print("\n--- Training Behavior Cloning Model ---")
    os.makedirs(output_dir, exist_ok=True)

    bc_dataset = CustomerSupportDataset(data_type="bc", tokenizer=tokenizer, bc_data=bc_data)
    data_collator = DataCollatorForLanguageModeling(tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1, # Keep epochs low for demo
        per_device_train_batch_size=2, # Small batch size for demo
        save_steps=10_000, # Only save at the end for this short demo
        save_total_limit=1,
        logging_dir="./logs",
        logging_steps=100,
        gradient_accumulation_steps=1,
        report_to="none", # Disable reporting to external services for demo
    )

    trainer = Trainer(
        model=llm_model,
        args=training_args,
        train_dataset=bc_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    llm_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Behavior Cloning model saved to {output_dir}")
    return llm_model

# --- Reward Model Training ---
class RewardModelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Custom loss for pairwise ranking: -log(sigmoid(score(preferred) - score(rejected)))
        input_ids_A = inputs["input_ids_A"]
        attention_mask_A = inputs["attention_mask_A"]
        input_ids_B = inputs["input_ids_B"]
        attention_mask_B = inputs["attention_mask_B"]
        labels = inputs["label"] # 0 if A preferred, 1 if B preferred

        scores_A = model(input_ids=input_ids_A, attention_mask=attention_mask_A)
        scores_B = model(input_ids=input_ids_B, attention_mask=attention_mask_B)

        scores_A = scores_A.squeeze() # Ensure scalar scores per item in batch
        scores_B = scores_B.squeeze()

        loss = torch.tensor(0.0, device=scores_A.device)
        for i in range(len(labels)):
            if labels[i] == 0: # A is preferred
                preferred_score = scores_A[i]
                rejected_score = scores_B[i]
            else: # B is preferred
                preferred_score = scores_B[i]
                rejected_score = scores_A[i]

            loss += -torch.nn.functional.logsigmoid(preferred_score - rejected_score)

        return (loss, {"scores_A": scores_A, "scores_B": scores_B}) if return_outputs else loss

def train_reward_model(reward_model, tokenizer, preference_data, output_dir="./reward_model"):
    print("\n--- Training Reward Model ---")
    os.makedirs(output_dir, exist_ok=True)

    preference_dataset = CustomerSupportDataset(data_type="preference", tokenizer=tokenizer, preference_data=preference_data)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1, # Keep epochs low for demo
        per_device_train_batch_size=2,
        save_steps=10_000,
        save_total_limit=1,
        logging_dir="./logs_rm",
        logging_steps=100,
        gradient_accumulation_steps=1,
        report_to="none", # Disable reporting to external services for demo
    )

    trainer = RewardModelTrainer(
        model=reward_model,
        args=training_args,
        train_dataset=preference_dataset,
        # Tokenizer is not directly used by Trainer if compute_loss handles tokenization logic via dataset
    )
    trainer.train()
    reward_model.encoder.save_pretrained(output_dir) # Save the underlying encoder
    tokenizer.save_pretrained(output_dir) # Save tokenizer for consistent loading
    print(f"Reward Model saved to {output_dir}")
    return reward_model

# --- Reinforcement Learning from Human Feedback (RLHF) ---
def train_rlhf(llm_model, reward_model, tokenizer, bc_data, output_dir="./rlhf_model"):
    print("\n--- Training RLHF Model (Conceptual Mock) ---")
    os.makedirs(output_dir, exist_ok=True)

    ppo_config = PPOConfig(
        learning_rate=1e-5,
        log_with="tensorboard",
        batch_size=2, # For simplicity, match BC batch size
        mini_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        # Adjust these parameters for actual RLHF training
    )

    # For RLHF, we usually provide prompts for the model to generate responses.
    # Using the prompts from BC data for this demonstration.
    rlhf_prompts = bc_data["prompt"].tolist()

    # Prepare a simple dataset of tokenized prompts for PPOTrainer
    rlhf_dataset = []
    for prompt_text in rlhf_prompts:
        input_ids = tokenizer(prompt_text, return_tensors="pt", truncation=True, padding="max_length", max_length=64)["input_ids"].squeeze()
        rlhf_dataset.append({"input_ids": input_ids})

    # policy_model: The LLM we want to train
    # ref_model: A frozen copy of the LLM for KL divergence calculation
    # For simplicity, load a fresh instance for the reference model
    ref_model = AutoModelForCausalLM.from_pretrained(llm_model.config._name_or_path, torch_dtype=torch.bfloat16).to(llm_model.device)

    ppo_trainer = PPOTrainer(
        ppo_config,
        llm_model, # Policy model
        ref_model, # Reference model
        tokenizer,
        dataset=rlhf_dataset,
    )

    print("NOTE: RLHF training steps are conceptually mocked here. A full RLHF loop is more involved.")
    print("Simulating generation, reward calculation, and a few PPO-like steps...")

    for epoch in range(ppo_config.num_train_epochs):
        for i, input_batch in enumerate(ppo_trainer.dataloader):
            if i >= 5: # Limit steps for a quick demo
                break
            query_tensors = input_batch["input_ids"]

            # Step 1: Generate responses from the current policy model
            # (ppo_trainer.generate handles this in a real scenario)
            response_tensors = ppo_trainer.generate(
                query_tensors,
                max_new_tokens=20,
                do_sample=True, top_k=50, top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )

            # Step 2: Calculate rewards using the trained Reward Model
            # This requires combining the query and the generated response for reward model input
            rewards = []
            for j in range(query_tensors.shape[0]):
                query_text = tokenizer.decode(query_tensors[j], skip_special_tokens=True)
                response_text = tokenizer.decode(response_tensors[j, query_tensors.shape[1]:], skip_special_tokens=True).strip()
                combined_text_for_rm = query_text + tokenizer.eos_token + response_text
                
                rm_input = tokenizer(combined_text_for_az, return_tensors="pt", truncation=True, padding="max_length", max_length=128).to(llm_model.device)
                with torch.no_grad():
                    reward_score = reward_model(input_ids=rm_input["input_ids"], attention_mask=rm_input["attention_mask"]).squeeze().item()
                    rewards.append(torch.tensor(reward_score).to(llm_model.device))
            
            rewards = torch.stack(rewards)

            # Step 3: Perform PPO optimization step
            # (ppo_trainer.step actually does the backward pass and optimization)
            # In this mock, we just acknowledge the data flow.
            print(f"  Epoch {epoch+1}, Step {i+1}: Processed batch for RLHF (Avg. reward: {rewards.mean().item():.4f})")
            # ppo_trainer.step(query_tensors, response_tensors, rewards)
            # If uncommented, the above line would run the PPO optimization.

    # Saving the final LLM after RLHF
    llm_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"RLHF trained model saved to {output_dir}")
    return llm_model

# --- 5. Inference Engine & Rejection Sampling (Best-of-N) ---
class CustomerSupportAgent:
    def __init__(self, llm_model_path, reward_model_path, tokenizer_path, device="cpu"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Initializing CustomerSupportAgent on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(llm_model_path, torch_dtype=torch.bfloat16).to(self.device)
        self.llm.eval() # Set LLM to evaluation mode
        self.llm.config.pad_token_id = self.tokenizer.eos_token_id # Ensure pad_token_id is set for generation

        # Load the reward model's encoder, then wrap it in our RewardModel class
        reward_encoder = AutoModelForSequenceClassification.from_pretrained(reward_model_path, num_labels=1).to(self.device)
        self.reward_model = RewardModel() # Create an empty wrapper
        self.reward_model.encoder = reward_encoder # Assign the loaded encoder
        self.reward_model.eval() # Set reward model to evaluation mode
        print("Agent initialized.")

    def generate_response(self, prompt, num_candidates=5, max_new_tokens=50):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        # Generate N candidate responses
        candidate_texts = []
        with torch.no_grad():
            for _ in range(num_candidates):
                output = self.llm.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True, # Enable sampling for diverse responses
                    top_k=50, top_p=0.95, # Common sampling parameters
                    num_return_sequences=1 # Generate one sequence per loop
                )
                # Decode the generated part (excluding the prompt)
                response_ids = output[0, input_ids.shape[-1]:]
                response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
                candidate_texts.append(response_text)

        # Score each candidate response using the reward model
        scores = []
        with torch.no_grad():
            for response_text in candidate_texts:
                # Combine original prompt with generated response for consistent scoring with how RM was trained
                combined_text = prompt + self.tokenizer.eos_token + response_text
                rm_input = self.tokenizer(combined_text, return_tensors="pt", truncation=True, padding="max_length", max_length=128).to(self.device)
                score = self.reward_model(input_ids=rm_input["input_ids"], attention_mask=rm_input["attention_mask"]).squeeze().item()
                scores.append(score)

        # Select the best response based on the highest score
        best_response_idx = np.argmax(scores)
        best_response_text = candidate_texts[best_response_idx]

        print(f"\n--- Rejection Sampling Results for prompt: '{prompt}' ---")
        for i, (text, score) in enumerate(zip(candidate_texts, scores)):
            print(f"  Candidate {i+1} (Score: {score:.4f}): {text}")
        print(f"Selected Best Response (Score: {scores[best_response_idx]:.4f}): {best_response_text}")

        return best_response_text

# --- Main execution flow ---
if __name__ == "__main__":
    # Generate dummy data
    bc_data_df, preference_data_df = generate_dummy_data(num_samples=50)
    print("Generated dummy BC data samples:", len(bc_data_df))
    print("Generated dummy Preference data samples:", len(preference_data_df))

    # Initialize LLM and Tokenizer
    llm_model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm_model.config.pad_token_id = tokenizer.eos_token_id

    # 1. Behavior Cloning
    bc_model_path = "./bc_model_output"
    llm_model = train_behavior_cloning(llm_model, tokenizer, bc_data_df, output_dir=bc_model_path)

    # 2. Reward Model Training
    reward_model = RewardModel(model_name=REWARD_MODEL_NAME)
    rm_model_path = "./reward_model_output"
    # Ensure reward model is on the correct device for training
    reward_model = train_reward_model(reward_model.to("cuda" if torch.cuda.is_available() else "cpu"), 
                                    tokenizer, preference_data_df, output_dir=rm_model_path)

    # 3. RLHF (using the LLM after BC and the trained Reward Model)
    # Reload LLM from BC path to ensure it's the model trained by BC for RLHF init
    llm_for_rlhf = AutoModelForCausalLM.from_pretrained(bc_model_path, torch_dtype=torch.bfloat16)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm_for_rlhf.config.pad_token_id = tokenizer.eos_token_id

    rlhf_model_path = "./rlhf_model_output"
    # Ensure LLM and Reward Model are on the correct device for RLHF
    llm_after_rlhf = train_rlhf(llm_for_rlhf.to("cuda" if torch.cuda.is_available() else "cpu"),
                                reward_model.to("cuda" if torch.cuda.is_available() else "cpu"), 
                                tokenizer, bc_data_df, output_dir=rlhf_model_path)

    # 4. Inference with Rejection Sampling
    print("\n--- Demonstrating Inference with Rejection Sampling ---")
    # Load the final LLM (after RLHF) and the trained Reward Model
    agent = CustomerSupportAgent(
        llm_model_path=rlhf_model_path,
        reward_model_path=rm_model_path,
        tokenizer_path=rlhf_model_path, # Tokenizer should be consistent, typically saved with LLM
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    test_prompt_1 = "I need to return a product. What is your return policy?"
    agent.generate_response(test_prompt_1, num_candidates=3)

    test_prompt_2 = "My order #12345 has not arrived yet."
    agent.generate_response(test_prompt_2, num_candidates=3)

    print("\n--- End of Demo ---")