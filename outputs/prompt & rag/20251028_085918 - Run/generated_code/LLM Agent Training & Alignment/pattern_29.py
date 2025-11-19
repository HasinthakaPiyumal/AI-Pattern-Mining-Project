
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, RewardTrainer, RewardConfig, DPOConfig, DPOTrainer
from datasets import Dataset
import random
import json

# --- 1. Core LLM Foundation ---
# Placeholder for a small model for demonstration. In a real scenario, this would be a large, domain-adapted model.
MODEL_NAME = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# Add a pad token if the tokenizer doesn't have one, which is common for generation models
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

llm_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# --- 2. Data Collection Modules ---
class DataCollector:
    def __init__(self):
        self.medical_prompts = [
            "What are the early symptoms of diabetes?",
            "How can I manage high blood pressure through diet?",
            "Explain the side effects of insulin.",
            "What are common exercises for heart health?",
            "How often should I check my blood sugar if I have Type 2 diabetes?"
        ]
        self.expert_responses = [
            "Early symptoms of diabetes often include frequent urination, increased thirst, and unexplained weight loss.",
            "Managing high blood pressure through diet involves reducing sodium intake, eating plenty of fruits and vegetables, and choosing lean proteins.",
            "Common side effects of insulin can include hypoglycemia (low blood sugar), weight gain, and injection site reactions.",
            "Regular aerobic exercises like brisk walking, jogging, and swimming are excellent for heart health. Aim for at least 150 minutes per week.",
            "If you have Type 2 diabetes, your doctor will advise on blood sugar monitoring frequency, but it's typically daily or several times a week, especially around meals."
        ]
        self.bad_responses = [
            "Just ignore your symptoms, they will go away.",
            "Eat more salty foods to lower blood pressure.",
            "Insulin has no side effects.",
            "Avoid all exercise to protect your heart.",
            "You never need to check your blood sugar."
        ]

    def generate_expert_demonstrations(self, num_samples=100):
        demonstrations = []
        for _ in range(num_samples):
            prompt = random.choice(self.medical_prompts)
            completion = self.expert_responses[self.medical_prompts.index(prompt)]
            demonstrations.append({"prompt": prompt, "completion": completion})
        print(f"Generated {len(demonstrations)} expert demonstrations.")
        return demonstrations

    def generate_preference_data(self, num_samples=50):
        preference_data = []
        for _ in range(num_samples):
            prompt = random.choice(self.medical_prompts)
            chosen = self.expert_responses[self.medical_prompts.index(prompt)]
            rejected = random.choice(self.bad_responses)
            preference_data.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
        print(f"Generated {len(preference_data)} preference data samples.")
        return preference_data

data_collector = DataCollector()
expert_demos = data_collector.generate_expert_demonstrations()
preference_data = data_collector.generate_preference_data()

# Convert to Hugging Face Dataset format
demo_dataset = Dataset.from_list(expert_demos)
pref_dataset = Dataset.from_list(preference_data)

# --- 3. Behavior Cloning (BC) Module ---
def tokenize_function(examples):
    return tokenizer(examples["prompt"] + tokenizer.eos_token + examples["completion"], truncation=True, max_length=128)

tokenized_demo_dataset = demo_dataset.map(tokenize_function, batched=True)

training_args_bc = TrainingArguments(
    output_dir="./results_bc",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    logging_dir="./logs_bc",
    logging_steps=10,
    report_to="none",
)

trainer_bc = Trainer(
    model=llm_model,
    args=training_args_bc,
    train_dataset=tokenized_demo_dataset,
    tokenizer=tokenizer,
)

print("Starting Behavior Cloning...")
trainer_bc.train()
print("Behavior Cloning complete.")

bc_llm_model = llm_model # The LLM after BC

# --- 4. Reward Model (RM) Training Module ---
rm_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)
rm_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if rm_tokenizer.pad_token is None:
    rm_tokenizer.pad_token = rm_tokenizer.eos_token
rm_model.config.pad_token_id = rm_tokenizer.pad_token_id

def preprocess_function_rm(examples):
    # Tokenize chosen and rejected responses
    tokenized_chosen = rm_tokenizer(examples["prompt"] + " " + examples["chosen"], truncation=True, max_length=128)
    tokenized_rejected = rm_tokenizer(examples["prompt"] + " " + examples["rejected"], truncation=True, max_length=128)

    input_ids = tokenized_chosen["input_ids"] + tokenized_rejected["input_ids"]
    attention_mask = tokenized_chosen["attention_mask"] + tokenized_rejected["attention_mask"]
    
    # The 'labels' here would indicate which is preferred. For RewardTrainer, it expects 'chosen' and 'rejected' fields directly.
    return {
        "input_ids_chosen": tokenized_chosen["input_ids"],
        "attention_mask_chosen": tokenized_chosen["attention_mask"],
        "input_ids_rejected": tokenized_rejected["input_ids"],
        "attention_mask_rejected": tokenized_rejected["attention_mask"],
    }

# For RewardTrainer, we actually need specific column names and it handles the pairing internally.
# The `trl.RewardTrainer` expects `input_ids`, `attention_mask` and `rewards` or `chosen`, `rejected`.
# Let's adjust for `RewardTrainer`'s direct input style with `chosen` and `rejected`.

# It's usually better to just pass prompt, chosen, rejected to RewardTrainer and let it handle tokenization.
# However, for a fully manual approach or if we were using a vanilla Trainer, we would preprocess like above.
# For trl.RewardTrainer, the dataset needs 'input_ids_chosen', 'attention_mask_chosen', etc. 
# The `RewardTrainer` internally creates the pairs for comparison. 
# Let's use the raw preference_data and let RewardTrainer handle the tokenization, if possible.

# A simpler way for RewardTrainer is to pass the text directly and let it tokenize.
# But for dataset compatibility, ensure the format matches expected 'chosen' and 'rejected' columns.
# Let's re-structure the dataset for RewardTrainer:
rm_dataset = pref_dataset.map(
    lambda examples: {
        "chosen": examples["chosen"],
        "rejected": examples["rejected"],
        "prompt": examples["prompt"]
    },
    batched=True
)

reward_config = RewardConfig(
    output_dir="./results_rm",
    per_device_train_batch_size=1,
    num_train_epochs=1,
    logging_steps=10,
    report_to="none",
)

reward_trainer = RewardTrainer(
    model=rm_model,
    tokenizer=rm_tokenizer,
    args=reward_config,
    train_dataset=rm_dataset,
)

print("Starting Reward Model Training...")
reward_trainer.train()
print("Reward Model Training complete.")

# --- 5. Reinforcement Learning from Human Feedback (RLHF) Module ---
# The LLM with a value head for PPO
ppo_llm_model = AutoModelForCausalLMWithValueHead.from_pretrained(bc_llm_model.pretrained_model_name_or_path, value_head_type="linear")
ppo_tokenizer = tokenizer

# For DPO, we use the original LLM and the reference model
dpo_llm_model = bc_llm_model
dpo_ref_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME) # A frozen version of the BC model

# Prepare dataset for PPO/DPO
def ppo_dpo_tokenize(sample):
    return {"input_ids": ppo_tokenizer.encode(sample["prompt"], return_tensors="pt")[0]}

ppo_dpo_dataset = demo_dataset.map(ppo_dpo_tokenize, batched=False)
ppo_dpo_dataset.set_format(type="torch")

# PPO Configuration
ppo_config = PPOConfig(
    learning_rate=1e-5,
    batch_size=1,
    mini_batch_size=1,
    gradient_accumulation_steps=1,
    log_with="none",
    ppo_epochs=1,
    tracker_project_name="medical_assistant_ppo",
    output_dir="./results_ppo"
)

# DPOTrainer needs `chosen` and `rejected` responses, like the RewardTrainer.
# Let's use the preference_data for DPO.
dpo_config = DPOConfig(
    output_dir="./results_dpo",
    learning_rate=1e-5,
    per_device_train_batch_size=1,
    num_train_epochs=1,
    logging_steps=10,
    report_to="none",
)

print("Starting DPO Training...")
dpo_trainer = DPOTrainer(
    model=dpo_llm_model,
    ref_model=dpo_ref_model,
    args=dpo_config,
    tokenizer=ppo_tokenizer,
    train_dataset=rm_dataset, # Using the same preference dataset for DPO
)
dpo_trainer.train()
print("DPO Training complete.")

# For simplicity, we'll use the DPO trained model for inference.
final_llm_model = dpo_llm_model

# --- 6. Inference and Rejection Sampling Module ---
def generate_and_score_responses(prompt, llm_model, reward_model, tokenizer, num_candidates=3):
    candidates = []
    for _ in range(num_candidates):
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=128)
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=50,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        candidates.append(response)

    scored_candidates = []
    for candidate in candidates:
        # The reward model expects an input that represents the prompt + response
        # Tokenize as a single sequence for the RM
        rm_inputs = reward_model_tokenizer(prompt + " " + candidate, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            score = reward_model(**rm_inputs).logits.squeeze().item()
        scored_candidates.append((candidate, score))
    
    # Sort by score in descending order
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    return scored_candidates

def get_best_response_with_rejection_sampling(prompt, llm_model, reward_model, tokenizer, num_candidates=5):
    scored_candidates = generate_and_score_responses(prompt, llm_model, reward_model, tokenizer, num_candidates)
    if scored_candidates:
        return scored_candidates[0][0] # Return the response with the highest score
    return "I'm sorry, I couldn't generate a helpful response at this moment."

# --- 7. User Interface (Conceptual) ---
print("\n--- Personalized Medical Assistant ---\n")
print("Type your medical query or 'quit' to exit.")

while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break

    # Use the DPO-trained LLM and the trained Reward Model
    response = get_best_response_with_rejection_sampling(
        user_input,
        final_llm_model,
        rm_model,
        tokenizer # Using the same tokenizer for both LLM and RM for simplicity
    )
    print(f"Assistant: {response}")

print("Thank you for using the Medical Assistant. Goodbye!")
