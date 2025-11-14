"""
This script implements a conceptual framework for an Intelligent Customer Support Agent for E-commerce,
integrating key AI design patterns like Behavior Cloning (SFT), Reward Modeling, and RLHF concepts.
It uses dummy data for demonstration purposes. In a real-world scenario, actual data collection
and more robust training configurations would be used.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
import random

# --- Configuration --- #
MODEL_NAME = "distilbert-base-uncased"  # Using a smaller model for demonstration
                                        # In production, use Llama-2, Mistral, etc.
SFT_MODEL_OUTPUT_DIR = "./sft_model"
REWARD_MODEL_OUTPUT_DIR = "./reward_model"

# --- 1. Data Collection and Preprocessing Module (Dummy Data) ---
def generate_dummy_demonstrations(num_samples=100):
    """Generates dummy human expert demonstrations for SFT."""
    demonstrations = []
    for i in range(num_samples):
        query = f"Customer query about product {i % 10}"
        response = f"Expert response explaining product {i % 10} features and common FAQs."
        demonstrations.append({"query": query, "response": response})
    print(f"Generated {num_samples} dummy demonstrations.")
    return pd.DataFrame(demonstrations)

def generate_dummy_preference_data(num_samples=50):
    """Generates dummy preference data for reward model training."""
    preference_data = []
    for i in range(num_samples):
        query = f"Customer asks about return policy for item {i % 5}"
        # Simulate two responses, one better than the other
        response_a = f"Please check our website for the return policy."
        response_b = f"Our return policy allows returns within 30 days of purchase for most items. Please visit our 'Returns' page for detailed instructions and exceptions."
        
        # Randomly assign which one is 'chosen' (preferred)
        if random.random() > 0.5:
            chosen = response_b
            rejected = response_a
        else:
            chosen = response_a
            rejected = response_b
            
        preference_data.append({"query": query, "chosen": chosen, "rejected": rejected})
    print(f"Generated {num_samples} dummy preference pairs.")
    return pd.DataFrame(preference_data)

def preprocess_sft_data(tokenizer, df):
    """Preprocesses demonstration data for SFT."""
    def tokenize_function(examples):
        # For SFT, we concatenate query and response
        return tokenizer([q + " " + r for q, r in zip(examples["query"], examples["response"])], truncation=True, padding="max_length")
    
    dataset = Dataset.from_pandas(df)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    # For CausalLM, labels are usually the input_ids themselves shifted
    tokenized_dataset = tokenized_dataset.map(lambda examples: {"labels": examples["input_ids"]}, batched=True)
    return tokenized_dataset.remove_columns(["query", "response", "__index__"])

def preprocess_reward_data(tokenizer, df):
    """Preprocesses preference data for Reward Model training."""
    def tokenize_function(examples):
        chosen_tokenized = tokenizer([q + " " + c for q, c in zip(examples["query"], examples["chosen"])], truncation=True, padding="max_length")
        rejected_tokenized = tokenizer([q + " " + r for q, r in zip(examples["query"], examples["rejected"])], truncation=True, padding="max_length")
        return {
            "input_ids_chosen": chosen_tokenized["input_ids"],
            "attention_mask_chosen": chosen_tokenized["attention_mask"],
            "input_ids_rejected": rejected_tokenized["input_ids"],
            "attention_mask_rejected": rejected_tokenized["attention_mask"],
        }

    dataset = Dataset.from_pandas(df)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset.remove_columns(["query", "chosen", "rejected", "__index__"])

# --- 2. Behavior Cloning for Initial Skill Acquisition (SFT Module) ---
def train_sft_model(model_name, train_dataset, output_dir):
    """Trains the base LLM using Supervised Fine-Tuning (SFT)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Pad token might be missing for some models; add if necessary
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'}) 
        
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        model.resize_token_embeddings(len(tokenizer))

    # Preprocess data
    processed_train_dataset = preprocess_sft_data(tokenizer, train_dataset)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,  # Reduced for demo
        per_device_train_batch_size=2, # Reduced for demo
        save_steps=10_000, # Large save_steps for demo
        logging_dir='./logs',
        logging_steps=100,
        learning_rate=2e-5,
        fp16=torch.cuda.is_available(), # Use mixed precision if GPU available
        gradient_accumulation_steps=1 # For demo
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_train_dataset,
        tokenizer=tokenizer,
    )

    print("\n--- Starting SFT Training ---")
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"SFT model saved to {output_dir}")
    return model, tokenizer

# --- 3. Reward Modeling Module ---
class RewardModel(torch.nn.Module):
    """A simple Reward Model head on top of a transformer base."""
    def __init__(self, base_model_name):
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        # Freeze base model layers initially (optional, but common for reward models)
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Add a regression head to predict a scalar reward
        self.reward_head = torch.nn.Linear(self.base_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        # Get the hidden states from the base model
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # Use the hidden state of the last token (or pooled output)
        last_hidden_state = outputs.hidden_states[-1]
        pooled_output = last_hidden_state[:, -1, :]
        
        # Predict the reward
        reward = self.reward_head(pooled_output)
        return reward

def train_reward_model(base_model_name, train_df, output_dir):
    """Trains a Reward Model based on human preferences."""
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'}) 

    reward_model = RewardModel(base_model_name)
    if tokenizer.pad_token is None:
        reward_model.base_model.resize_token_embeddings(len(tokenizer))
    
    # Ensure the reward head is trainable
    for param in reward_model.reward_head.parameters():
        param.requires_grad = True

    processed_train_dataset = preprocess_reward_data(tokenizer, train_df)
    
    # Custom Trainer for Reward Model (simplified)
    class RewardTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            # Extract chosen and rejected inputs
            chosen_input_ids = inputs["input_ids_chosen"]
            chosen_attention_mask = inputs["attention_mask_chosen"]
            rejected_input_ids = inputs["input_ids_rejected"]
            rejected_attention_mask = inputs["attention_mask_rejected"]

            # Get rewards for chosen and rejected responses
            rewards_chosen = model(chosen_input_ids, chosen_attention_mask)
            rewards_rejected = model(rejected_input_ids, rejected_attention_mask)

            # DPO-like or pairwise ranking loss: chosen should have higher reward
            # Using a simplified margin loss for illustration
            loss = -torch.nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
            return (loss, {'rewards_chosen': rewards_chosen, 'rewards_rejected': rewards_rejected}) if return_outputs else loss

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,  # Reduced for demo
        per_device_train_batch_size=2, # Reduced for demo
        save_steps=10_000, # Large save_steps for demo
        logging_dir='./logs_rm',
        logging_steps=100,
        learning_rate=1e-5,
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=1
    )

    trainer = RewardTrainer(
        model=reward_model,
        args=training_args,
        train_dataset=processed_train_dataset,
        tokenizer=tokenizer,
    )

    print("\n--- Starting Reward Model Training ---")
    trainer.train()
    torch.save(reward_model.state_dict(), f"{output_dir}/reward_model.pt")
    tokenizer.save_pretrained(output_dir)
    print(f"Reward model saved to {output_dir}")
    return reward_model, tokenizer

# --- 4. Inference / Agent Usage (Conceptual) ---
class CustomerSupportAgent:
    def __init__(self, sft_model_path, reward_model_path, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(sft_model_path)
        self.sft_model = AutoModelForCausalLM.from_pretrained(sft_model_path)
        self.reward_model = RewardModel(model_name)
        self.reward_model.load_state_dict(torch.load(f"{reward_model_path}/reward_model.pt"))
        self.reward_model.eval() # Set to evaluation mode
        
        if torch.cuda.is_available():
            self.sft_model.to('cuda')
            self.reward_model.to('cuda')

    def generate_response_candidates(self, query, num_candidates=3):
        input_ids = self.tokenizer.encode(query, return_tensors="pt")
        if torch.cuda.is_available():
            input_ids = input_ids.to('cuda')
            
        candidates = []
        for _ in range(num_candidates):
            output = self.sft_model.generate(
                input_ids,
                max_new_tokens=50,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id # Or tokenizer.pad_token_id
            )
            response = self.tokenizer.decode(output[0][len(input_ids[0]):], skip_special_tokens=True)
            candidates.append(response.strip())
        return candidates

    def select_best_response(self, query, candidates):
        """Uses the reward model for rejection sampling (Best-of-N selection)."""
        if not candidates:
            return "I'm sorry, I couldn't generate a response."

        best_response = None
        highest_reward = -float('inf')

        for candidate in candidates:
            full_text = query + " " + candidate
            input_ids = self.tokenizer.encode(full_text, return_tensors="pt")
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
            
            if torch.cuda.is_available():
                input_ids = input_ids.to('cuda')
                attention_mask = attention_mask.to('cuda')

            with torch.no_grad():
                reward = self.reward_model(input_ids, attention_mask).item()
            
            print(f"  Candidate: '{candidate[:50]}...' | Reward: {reward:.4f}")
            if reward > highest_reward:
                highest_reward = reward
                best_response = candidate
        
        print(f"Selected best response with reward: {highest_reward:.4f}")
        return best_response

    def chat(self, query):
        print(f"\nCustomer: {query}")
        candidates = self.generate_response_candidates(query)
        print(f"Generated {len(candidates)} response candidates.")
        best_response = self.select_best_response(query, candidates)
        print(f"Agent: {best_response}")
        return best_response

# --- Main Execution Flow --- #
if __name__ == "__main__":
    # 1. Generate Dummy Data
    sft_df = generate_dummy_demonstrations()
    rm_df = generate_dummy_preference_data()

    # 2. Train SFT Model (Behavior Cloning)
    # Note: For a real application, you'd use a larger, more capable LLM and more data.
    sft_model, sft_tokenizer = train_sft_model(MODEL_NAME, sft_df, SFT_MODEL_OUTPUT_DIR)

    # 3. Train Reward Model
    # This reward model learns to rank responses based on human preferences.
    reward_model, rm_tokenizer = train_reward_model(MODEL_NAME, rm_df, REWARD_MODEL_OUTPUT_DIR)

    # 4. Simulate Agent Interaction with Rejection Sampling
    print("\n--- Simulating Customer Support Agent ---")
    agent = CustomerSupportAgent(SFT_MODEL_OUTPUT_DIR, REWARD_MODEL_OUTPUT_DIR, MODEL_NAME)
    
    agent.chat("I want to know about the return policy for a shirt I bought last week.")
    agent.chat("Can you tell me the features of the new 'Eco-friendly water bottle'?")
    agent.chat("My order #12345 has not arrived yet. What should I do?")

    print("\n--- Framework Execution Complete ---")
