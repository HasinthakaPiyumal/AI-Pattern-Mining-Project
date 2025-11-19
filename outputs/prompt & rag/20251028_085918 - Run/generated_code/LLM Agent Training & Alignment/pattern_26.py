import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, Trainer, TrainingArguments
from trl import PPOTrainer, PPOConfig
from datasets import Dataset
import pandas as pd
import random

# --- 1. Data Simulation --- 

def generate_dummy_data():
    # Behavior Cloning Data
    bc_data = []
    for i in range(100):
        query = f"User query {i}: How do I reset my password?"
        response = f"Expert response {i}: You can reset your password by visiting the 'Forgot Password' link on the login page and following the instructions."
        bc_data.append({"query": query, "response": response})
    bc_df = pd.DataFrame(bc_data)

    # Reward Model Data (comparison pairs)
    rm_data = []
    for i in range(50):
        query = f"User query for RM {i}: My internet is not working."
        response_a = f"Response A {i}: Please restart your router and modem. If that doesn't work, contact support."
        response_b = f"Response B {i}: Have you tried turning it off and on again?"
        preferred = random.choice([0, 1]) # 0 for A, 1 for B
        rm_data.append({"query": query, "response_a": response_a, "response_b": response_b, "preferred": preferred})
    rm_df = pd.DataFrame(rm_data)
    
    return bc_df, rm_df

bc_df, rm_df = generate_dummy_data()
bc_dataset = Dataset.from_pandas(bc_df)
rm_dataset = Dataset.from_pandas(rm_df)

# --- 2. Behavior Cloning Module ---

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token # gpt2 doesn't have a pad token by default

bc_model = AutoModelForCausalLM.from_pretrained(model_name)

def preprocess_bc_function(examples):
    inputs = [f"Query: {q}\nResponse: " for q in examples["query"]]
    targets = examples["response"]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

bc_tokenized_dataset = bc_dataset.map(preprocess_bc_function, batched=True)
bc_tokenized_dataset = bc_tokenized_dataset.remove_columns(["query", "response"])

bc_training_args = TrainingArguments(
    output_dir="./results_bc",
    per_device_train_batch_size=4,
    num_train_epochs=1,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=50,
    overwrite_output_dir=True,
)

bc_trainer = Trainer(
    model=bc_model,
    args=bc_training_args,
    train_dataset=bc_tokenized_dataset,
)

bc_trainer.train()

print("Behavior Cloning training complete.")

# --- 3. Reward Model Module ---

# A simpler RM model for demonstration: a classification model that takes two responses and a query,
# and learns to prefer one over the other. Output 1 for preferred A, 0 for preferred B. This is a simplification.
# A more robust RM would output a scalar score for a single response.

rm_model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", 
    num_labels=1 # For regression output
)
rm_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def preprocess_rm_function(examples):
    # Concatenate query and responses for a single input to predict preference score
    # Here, we'll create an input for response A and another for response B, then compare scores
    inputs_a = [f"Query: {q} Response: {r_a}" for q, r_a in zip(examples["query"], examples["response_a"])]
    inputs_b = [f"Query: {q} Response: {r_b}" for q, r_b in zip(examples["query"], examples["response_b"])]
    
    tokenized_a = rm_tokenizer(inputs_a, max_length=256, truncation=True, padding="max_length", return_tensors="pt")
    tokenized_b = rm_tokenizer(inputs_b, max_length=256, truncation=True, padding="max_length", return_tensors="pt")
    
    # For simplicity, let's create a single 'text' input and a 'labels' indicating the preferred score
    # This is not a direct comparison model, but rather a preference learning setup where we teach the model
    # to score preferred responses higher.
    all_texts = []
    all_labels = [] # 1.0 for preferred, 0.0 for not preferred in this simplified setup

    for i in range(len(examples["query"])):
        if examples["preferred"][i] == 0: # A is preferred
            all_texts.append(f"Query: {examples['query'][i]} Response: {examples['response_a'][i]}")
            all_labels.append(1.0)
            all_texts.append(f"Query: {examples['query'][i]} Response: {examples['response_b'][i]}")
            all_labels.append(0.0)
        else: # B is preferred
            all_texts.append(f"Query: {examples['query'][i]} Response: {examples['response_b'][i]}")
            all_labels.append(1.0)
            all_texts.append(f"Query: {examples['query'][i]} Response: {examples['response_a'][i]}")
            all_labels.append(0.0)

    tokenized_inputs = rm_tokenizer(all_texts, max_length=256, truncation=True, padding="max_length")
    tokenized_inputs["labels"] = torch.tensor(all_labels, dtype=torch.float32).unsqueeze(-1)
    return tokenized_inputs

rm_tokenized_dataset = rm_dataset.map(preprocess_rm_function, batched=True, remove_columns=rm_dataset.column_names)

rm_training_args = TrainingArguments(
    output_dir="./results_rm",
    per_device_train_batch_size=4,
    num_train_epochs=1,
    logging_dir="./logs_rm",
    logging_steps=10,
    save_steps=50,
    overwrite_output_dir=True,
    learning_rate=2e-5,
    prediction_loss_only=False,
    evaluation_strategy="no"
)

rm_trainer = Trainer(
    model=rm_model,
    args=rm_training_args,
    train_dataset=rm_tokenized_dataset,
)

rm_trainer.train()

print("Reward Model training complete.")

# --- 4. RLHF/Rejection Sampling Module (using PPO) ---

# Load the BC model as the policy model for PPO
ppo_model = AutoModelForCausalLM.from_pretrained(bc_model.config.name_or_path)
ppo_model.load_state_dict(bc_model.state_dict()) # Initialize with BC weights

# Create a reference model (frozen copy of the initial BC model)
ref_model = AutoModelForCausalLM.from_pretrained(bc_model.config.name_or_path)
ref_model.load_state_dict(bc_model.state_dict())

ppo_config = PPOConfig(
    model_name=model_name,
    learning_rate=1e-5,
    ppo_epochs=4,
    mini_batch_size=1,
    batch_size=4,
    gradient_accumulation_steps=1,
    target_kl=0.1,
    init_kl_coef=0.2,
    adap_kl_ctrl=True,
)

ppo_tokenizer = AutoTokenizer.from_pretrained(model_name)
ppo_tokenizer.pad_token = ppo_tokenizer.eos_token

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=ppo_model,
    ref_model=ref_model,
    tokenizer=ppo_tokenizer,
)

# Simulate some data for RLHF (e.g., new user queries)
rlhf_queries = [f"RLHF query {i}: I have a billing issue." for i in range(20)]
rlhf_query_dataset = Dataset.from_dict({"query": rlhf_queries})

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


for epoch in range(1):
    for batch in ppo_trainer.dataloader:
        query_texts = batch["query"]
        query_tensors = [ppo_tokenizer(q, return_tensors="pt").input_ids.squeeze(0) for q in query_texts]
        
        # Generate responses using the current policy
        response_tensors = []
        for query_tensor in query_tensors:
            response = ppo_trainer.generate(
                query_tensor.to(ppo_trainer.current_device),
                max_new_tokens=50,
                do_sample=True,
                top_k=0, # disable top_k
                top_p=1.0, # disable top_p
                pad_token_id=ppo_tokenizer.eos_token_id
            )
            response_tensors.append(response.squeeze(0))

        # Decode responses for RM
        responses = [ppo_tokenizer.decode(r[len(q):], skip_special_tokens=True) for r, q in zip(response_tensors, query_tensors)]

        # Get rewards from the Reward Model
        rewards = []
        for q_text, r_text in zip(query_texts, responses):
            input_text = f"Query: {q_text} Response: {r_text}"
            inputs = rm_tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=256).to(rm_model.device)
            with torch.no_grad():
                # The RM model outputs logits, which we use as the reward score
                reward_score = rm_model(**inputs).logits.squeeze().item()
            rewards.append(torch.tensor(reward_score))
        
        rewards = torch.tensor(rewards).to(ppo_trainer.current_device)
        
        # PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

print("RLHF training complete.")

# --- 5. Inference Module ---

def generate_agent_response(query_text):
    inputs = ppo_tokenizer(query_text, return_tensors="pt").to(ppo_trainer.current_device)
    response_tensor = ppo_trainer.generate(
        inputs.input_ids,
        max_new_tokens=100,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        pad_token_id=ppo_tokenizer.eos_token_id
    )
    # Decode the generated response, excluding the prompt part
    decoded_response = ppo_tokenizer.decode(response_tensor[0, len(inputs.input_ids[0]):], skip_special_tokens=True)
    return decoded_response

# --- Example Usage ---

print("\n--- Agent in Action ---")
user_query = "I need help setting up my new account."
agent_response = generate_agent_response(user_query)
print(f"User: {user_query}")
print(f"Agent: {agent_response}")

user_query_2 = "My payment failed, what should I do?"
agent_response_2 = generate_agent_response(user_query_2)
print(f"User: {user_query_2}")
print(f"Agent: {agent_response_2}")
