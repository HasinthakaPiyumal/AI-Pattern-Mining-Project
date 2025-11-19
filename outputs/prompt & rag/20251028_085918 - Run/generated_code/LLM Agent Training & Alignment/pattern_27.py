import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
from trl import RewardTrainer, RewardConfig

# --- 1. Behavior Cloning Module (Initial Skill Acquisition) ---

class CustomerSupportDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, data_pairs, max_length=512):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attention_mask = []

        for query, response in data_pairs:
            # Combine query and response for causal language modeling
            full_text = f"Customer: {query}\nAgent: {response}"
            encoded = tokenizer(full_text, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
            self.input_ids.append(encoded["input_ids"][0])
            self.attention_mask.append(encoded["attention_mask"][0])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx], "attention_mask": self.attention_mask[idx], "labels": self.input_ids[idx]}


def train_behavior_cloning_model(model_name="gpt2", data_pairs=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if data_pairs is None:
        data_pairs = [
            ("My internet is not working.", "Please restart your router and modem. If the issue persists, contact technical support."),
            ("How do I change my billing address?", "You can update your billing address in your account settings under the 'Billing Information' section."),
        ]

    dataset = CustomerSupportDataset(tokenizer, data_pairs)

    training_args = TrainingArguments(
        output_dir="./bc_results",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        save_steps=500,
        save_total_limit=2,
        logging_dir="./bc_logs",
        logging_steps=10,
        learning_rate=2e-5,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()
    model.save_pretrained("./bc_model")
    tokenizer.save_pretrained("./bc_model")
    return model, tokenizer

# --- 2. Reward Model Module (Subjective Quality Alignment) ---

class PreferenceDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, data_preferences, max_length=512):
        self.tokenizer = tokenizer
        self.data_preferences = data_preferences
        self.max_length = max_length

    def __len__(self):
        return len(self.data_preferences)

    def __getitem__(self, idx):
        item = self.data_preferences[idx]
        query = item["query"]
        chosen_response = item["chosen"]
        rejected_response = item["rejected"]

        chosen_encoded = self.tokenizer(
            f"Customer: {query}\nAgent: {chosen_response}",
            truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
        )
        rejected_encoded = self.tokenizer(
            f"Customer: {query}\nAgent: {rejected_response}",
            truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
        )

        return {
            "input_ids_chosen": chosen_encoded["input_ids"][0],
            "attention_mask_chosen": chosen_encoded["attention_mask"][0],
            "input_ids_rejected": rejected_encoded["input_ids"][0],
            "attention_mask_rejected": rejected_encoded["attention_mask"][0],
        }

def train_reward_model(model_name="bert-base-uncased", data_preferences=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token # Or set to a specific pad_token
    
    # Use AutoModelForSequenceClassification for a simple reward model or a small LLM with a classification head
    # For a true reward model, you might fine-tune a smaller LLM or a specific reward model architecture.
    # Here we simulate with a sequence classification head that outputs a score.
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1) 

    if data_preferences is None:
        data_preferences = [
            {"query": "My internet is slow.", "chosen": "Have you tried restarting your router?", "rejected": "Your internet speed is not guaranteed."}, 
            {"query": "How do I reset my password?", "chosen": "You can reset your password on the login page by clicking 'Forgot Password'.", "rejected": "Password reset is not possible."}, 
        ]

    dataset = PreferenceDataset(tokenizer, data_preferences)

    reward_config = RewardConfig(
        output_dir="./rm_results",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        save_steps=500,
        save_total_limit=2,
        logging_dir="./rm_logs",
        logging_steps=10,
        learning_rate=1e-5,
        fp16=torch.cuda.is_available(),
    )

    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=reward_config,
        train_dataset=dataset,
    )

    trainer.train()
    model.save_pretrained("./rm_model")
    tokenizer.save_pretrained("./rm_model")
    return model, tokenizer

# Example Usage (optional, can be commented out or run separately)
if __name__ == "__main__":
    print("Training Behavior Cloning Model...")
    bc_model, bc_tokenizer = train_behavior_cloning_model()
    print("Behavior Cloning Model trained and saved to ./bc_model")

    print("\nTraining Reward Model...")
    rm_model, rm_tokenizer = train_reward_model()
    print("Reward Model trained and saved to ./rm_model")

    # To use the trained models:
    # from transformers import pipeline
    # # For BC model (text generation)
    # generator = pipeline("text-generation", model=bc_model, tokenizer=bc_tokenizer)
    # print(generator("Customer: My printer is jammed.", max_new_tokens=50))
    # # For RM model (scoring - simplified, actual RM usually compares scores)
    # # A real RM would compare chosen vs rejected scores, higher for chosen is better.
    # # This example shows how to get a score for a single input.
    # # Input for RM should be formatted consistently, e.g., 'Customer: query\nAgent: response'
    # # score = rm_model(**rm_tokenizer(f"Customer: My internet is slow.\nAgent: Have you tried restarting your router?", return_tensors="pt")).logits.item()
    # # print(f"Score for good response: {score}")
    # # score = rm_model(**rm_tokenizer(f"Customer: My internet is slow.\nAgent: Your internet speed is not guaranteed.", return_tensors="pt")).logits.item()
    # # print(f"Score for bad response: {score}")
