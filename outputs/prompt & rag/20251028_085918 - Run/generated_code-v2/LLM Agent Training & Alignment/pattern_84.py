import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from trl import RewardTrainer, PPOTrainer, PPOConfig
from trl.core import LengthSampler
import gradio as gr
import random
import pandas as pd

# --- 1. Data Collection & Preprocessing Modules (Simulated) ---

class DataStorage:
    def __init__(self):
        self.demonstrations = []
        self.comparisons = []
        self.medical_knowledge = [
            "Symptoms of Marfan syndrome include tall and slender build, disproportionately long arms, legs and fingers.",
            "Ehlers-Danlos syndromes are a group of inherited disorders that affect connective tissues, primarily skin, joints, and blood vessel walls.",
            "Fibrodysplasia Ossificans Progressiva (FOP) causes fibrous tissues (muscles, tendons, ligaments) to progressively turn into bone (ossify)."
        ]

    def add_demonstration(self, state, action):
        self.demonstrations.append({"state": state, "action": action})

    def add_comparison(self, query, chosen_output, rejected_output):
        self.comparisons.append({"query": query, "chosen": chosen_output, "rejected": rejected_output})

    def get_demonstrations(self):
        return self.demonstrations

    def get_comparisons(self):
        return self.comparisons

    def get_knowledge(self):
        return self.medical_knowledge

data_store = DataStorage()

# Simulate Demonstration Data Collection
def collect_demonstration_data():
    data_store.add_demonstration(
        "Patient presents with unusually long limbs and flexible joints.",
        "Search for 'syndromes with hypermobility and dolichostenomelia'"
    )
    data_store.add_demonstration(
        "Search results suggest Marfan syndrome and Ehlers-Danlos syndrome.",
        "Review patient's family history for similar conditions and cardiovascular issues."
    )
    data_store.add_demonstration(
        "Patient has progressive stiffness in neck and back, no clear injury.",
        "Consider rare genetic disorders affecting connective tissue, specifically FOP."
    )

# Simulate Comparison Data Collection
def collect_comparison_data():
    data_store.add_comparison(
        "What is the likely diagnosis for a patient with extreme joint hypermobility, fragile skin, and easy bruising?",
        "Ehlers-Danlos Syndrome, hypermobile type, due to classic triad of symptoms.",
        "Marfan Syndrome, given the joint hypermobility, but fragile skin is more indicative of EDS."
    )
    data_store.add_comparison(
        "Suggest a preliminary diagnostic plan for suspected Marfan syndrome.",
        "Echocardiogram to assess aortic root dilation, ophthalmologic exam for lens dislocation, and genetic testing for FBN1 gene mutation.",
        "Order general blood tests and a full body X-ray to look for bone abnormalities."
    )

collect_demonstration_data()
collect_comparison_data()

# --- 2. Agentic LLM Training Pipeline ---

# Base Model and Tokenizer (using a small, generic model for demonstration)
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def preprocess_demonstrations(demonstrations, tokenizer):
    inputs = []
    for item in demonstrations:
        # Format as a conversation for BC
        prompt = f"[STATE] {item['state']} [ACTION] {item['action']}"
        inputs.append(prompt)
    return tokenizer(inputs, truncation=True, padding=True, return_tensors="pt")

def preprocess_comparisons(comparisons, tokenizer):
    # For RewardTrainer, we need text_j (rejected) and text_k (chosen)
    # And ideally, a query.
    formatted_data = []
    for item in comparisons:
        formatted_data.append({
            "query": item['query'],
            "response_j": item['rejected'],  # Rejected output
            "response_k": item['chosen']      # Chosen (preferred) output
        })
    return pd.DataFrame(formatted_data)

class DemonstrationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return self.encodings.input_ids.shape[0]

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

# Behavior Cloning (BC) Module
def train_behavior_cloning_llm(demonstrations, model_name, tokenizer):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    processed_data = preprocess_demonstrations(demonstrations, tokenizer)
    dataset = DemonstrationDataset(processed_data)

    # For causal language modeling, labels are usually the input_ids themselves shifted
    def collate_fn(batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = input_ids.clone()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    training_args = TrainingArguments(
        output_dir="./bc_model",
        per_device_train_batch_size=2, # Small batch for demo
        num_train_epochs=1, # Single epoch for demo
        logging_dir="./bc_logs",
        learning_rate=2e-5,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=collate_fn
    )
    trainer.train()
    return model

# Reward Modeling (RM) Module
def train_reward_model(comparisons, model_name, tokenizer):
    # Use a pre-trained language model head as the base for the reward model
    rm_model = AutoModelForCausalLM.from_pretrained(model_name)
    # A small linear layer on top for reward prediction
    rm_model.config.num_labels = 1 # For scalar reward
    rm_model.score = torch.nn.Linear(rm_model.config.hidden_size, 1)

    processed_data = preprocess_comparisons(comparisons, tokenizer)

    training_args = TrainingArguments(
        output_dir="./rm_model",
        per_device_train_batch_size=2, # Small batch for demo
        num_train_epochs=1, # Single epoch for demo
        logging_dir="./rm_logs",
        learning_rate=1e-5,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
    )

    reward_trainer = RewardTrainer(
        model=rm_model, # Pass the model to RewardTrainer
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=processed_data,
    )
    reward_trainer.train()
    return rm_model

# Reinforcement Learning from Human Feedback (RLHF) Module
def finetune_rlhf_llm(bc_model, reward_model, tokenizer):
    ppo_config = PPOConfig(
        learning_rate=1e-5,
        batch_size=2,
        mini_batch_size=1,
        gradient_accumulation_steps=1,
        seed=0,
        init_kl_coef=0.2,
        target_kl=0.1,
        log_with="none", # No logging for simple demo
        ppo_epochs=1,
    )

    # Reference model for PPO (often a frozen copy of the BC model)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name)

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=bc_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        reward_model=reward_model, # Pass the reward model to the trainer
    )

    # Generate dummy query data for PPO training (simulating new patient cases)
    query_data = [
        "Patient has joint pain and skin issues, what's a rare diagnosis?",
        "How to confirm suspected FOP?",
        "Initial management for Marfan syndrome?"
    ]
    query_tensors = [tokenizer(q, return_tensors="pt").input_ids[0] for q in query_data]

    # PPO training loop (simplified)
    for epoch in range(1):
        for query in query_tensors:
            # Generate response from current LLM
            generation_kwargs = {
                "min_length": -1,
                "top_k": 0.0,
                "top_p": 1.0,
                "do_sample": True,
                "pad_token_id": tokenizer.pad_token_id,
                "max_new_tokens": 50,
            }
            response_tensors = ppo_trainer.generate(
                query,
                return_prompt=False,
                **generation_kwargs
            )
            # Decode and get reward
            responses = [tokenizer.decode(r.squeeze()) for r in response_tensors]

            # Simulate reward calculation using the reward_model
            # In a real scenario, this would involve the reward_model's forward pass
            # For this demo, we'll assign a mock reward based on length (very simplistic)
            rewards = [torch.tensor(len(r) / 100.0).float() for r in responses]

            # Train PPO
            ppo_trainer.step([query], response_tensors, rewards)

    return bc_model # Returns the fine-tuned model

print("Starting Behavior Cloning training...")
bc_llm = train_behavior_cloning_llm(data_store.get_demonstrations(), model_name, tokenizer)
print("Behavior Cloning training complete.")

print("Starting Reward Model training...")
reward_model = train_reward_model(data_store.get_comparisons(), model_name, tokenizer)
print("Reward Model training complete.")

print("Starting RLHF fine-tuning...")
final_llm = finetune_rlhf_llm(bc_llm, reward_model, tokenizer)
print("RLHF fine-tuning complete.")

# --- 3. Diagnostic Assistant Agent (Inference) ---

def get_diagnostic_assistance(patient_data):
    prompt = f"Patient data: {patient_data}\nBased on this, consider rare diseases and provide initial diagnostic steps and potential diagnoses."
    inputs = tokenizer(prompt, return_tensors="pt").to(final_llm.device)

    # Integrate medical knowledge (simple retrieval for demo)
    relevant_knowledge = []
    for fact in data_store.get_knowledge():
        if any(keyword in patient_data.lower() for keyword in fact.lower().split()):
            relevant_knowledge.append(fact)
    
    if relevant_knowledge:
        prompt += "\nRelevant Medical Knowledge: " + " ".join(relevant_knowledge)
        inputs = tokenizer(prompt, return_tensors="pt").to(final_llm.device)

    output_sequences = final_llm.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=200,
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=1,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )

    response = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    return response

# Gradio Interface
iface = gr.Interface(
    fn=get_diagnostic_assistance,
    inputs=gr.Textbox(lines=5, label="Enter Patient Data (Symptoms, History, Lab Results, etc.)"),
    outputs=gr.Textbox(label="Diagnostic Assistant's Response"),
    title="AI-Powered Rare Disease Diagnostic Assistant",
    description="This assistant leverages an agentic LLM trained with expert demonstrations and human preferences to help clinicians identify and manage rare diseases."
)

iface.launch()