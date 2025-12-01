import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, pipeline, BitsAndBytesConfig
from transformers import AdamW, get_scheduler
from datasets import Dataset as HFDataset
import pandas as pd
import random
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# Try to import trl, if not available, provide a mock for demonstration
try:
    from trl import PPOTrainer, PPOConfig
except ImportError:
    print("TRL not found. Mocking PPOConfig and PPOTrainer for demonstration.")
    class PPOConfig:
        def __init__(self, **kwargs):
            self.seed = 42
            self.learning_rate = 1e-5
            self.mini_batch_size = 4
            self.batch_size = 16
            self.gradient_accumulation_steps = 1
            self.log_with = None
            self.ppo_epochs = 4
            self.target_kl = 0.1
            for k, v in kwargs.items():
                setattr(self, k, v)

    class PPOTrainer:
        def __init__(self, config, model, ref_model, tokenizer, dataset, data_collator):
            self.config = config
            self.model = model
            self.ref_model = ref_model
            self.tokenizer = tokenizer
            self.dataset = dataset
            self.data_collator = data_collator
            print("PPOTrainer mocked: No actual RLHF training will occur.")

        def generate(self, query_tensors, **kwargs):
            return [torch.randint(0, self.tokenizer.vocab_size, (10,)) for _ in query_tensors]

        def learn(self): 
            print("PPOTrainer.learn() mocked: Model is not actually trained.")


# Configuration
BASE_LM_MODEL_NAME = "distilgpt2"
REWARD_MODEL_NAME = "distilbert-base-uncased"

# --- 1. Data Collection & Preprocessing Module ---
def generate_synthetic_preference_data(num_samples=100):
    queries = [
        "What is your return policy?",
        "How can I track my order?",
        "I need help with a billing issue.",
        "Can I change my shipping address?",
        "What are your operating hours?",
        "How do I reset my password?",
        "Tell me about your product X.",
        "Do you offer international shipping?",
        "My product arrived damaged.",
        "I want to cancel my subscription."
    ]

    data = []
    for i in range(num_samples):
        query = random.choice(queries)
        if i % 2 == 0:
            chosen_response = f"Of course! To track your order, please visit our website and enter your order number {random.randint(10000, 99999)}."
            rejected_response = f"You can track it on the site. Use your order ID."
        else:
            chosen_response = f"Our return policy allows returns within 30 days of purchase, provided the item is unused and in its original packaging."
            rejected_response = f"Returns are 30 days. Unused condition."

        data.append({"query": query, "chosen": chosen_response, "rejected": rejected_response})
    
    return HFDataset.from_pandas(pd.DataFrame(data))

# --- 3. Reward Model (RM) ---
class RewardModel(torch.nn.Module):
    def __init__(self, reward_model_name):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(reward_model_name, num_labels=1)
        self.model.config.problem_type = "regression"
        
    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

def train_reward_model(preference_data, reward_model, epochs=3, learning_rate=1e-5, batch_size=8):
    optimizer = AdamW(reward_model.parameters(), lr=learning_rate)
    
    def collate_fn(batch):
        queries = [item["query"] for item in batch]
        chosen = [item["chosen"] for item in batch]
        rejected = [item["rejected"] for item in batch]
        
        tokenized_chosen = reward_model.tokenizer(queries, chosen, padding=True, truncation=True, return_tensors="pt", max_length=128)
        tokenized_rejected = reward_model.tokenizer(queries, rejected, padding=True, truncation=True, return_tensors="pt", max_length=128)
        return tokenized_chosen, tokenized_rejected

    dataloader = DataLoader(preference_data, batch_size=batch_size, collate_fn=collate_fn)

    reward_model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_chosen, batch_rejected in dataloader:
            optimizer.zero_grad()

            chosen_rewards = reward_model(batch_chosen["input_ids"], batch_chosen["attention_mask"])
            rejected_rewards = reward_model(batch_rejected["input_ids"], batch_rejected["attention_mask"])

            loss = -torch.nn.functional.logsigmoid(chosen_rewards - rejected_rewards).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Reward Model Loss: {total_loss / len(dataloader)}")
    reward_model.eval()
    return reward_model

# --- 2. Base Language Model (Chatbot LM) & 4. RLHF Training Module ---

def train_rlhf_model(base_lm_model, base_lm_tokenizer, reward_model, preference_data, ppo_epochs=4, learning_rate=1e-5):
    ppo_config = PPOConfig(
        ppo_epochs=ppo_epochs,
        learning_rate=learning_rate,
        mini_batch_size=4,
        batch_size=16,
        gradient_accumulation_steps=1,
        target_kl=0.1,
        seed=42
    )

    # Use the same tokenizer as the base LM for PPO
    ppo_tokenizer = base_lm_tokenizer
    ppo_tokenizer.pad_token = ppo_tokenizer.eos_token
    
    def build_dataset_for_ppo(data):
        formatted_data = []
        for item in data:
            formatted_data.append({"query": item["query"], "query_tensors": ppo_tokenizer(item["query"], return_tensors="pt").input_ids[0]})
        return HFDataset.from_pandas(pd.DataFrame(formatted_data))

    ppo_dataset = build_dataset_for_ppo(preference_data)

    # Reference model is a frozen copy of the base LM
    ref_model = type(base_lm_model)(base_lm_model.config).to(base_lm_model.device)
    ref_model.load_state_dict(base_lm_model.state_dict())
    for param in ref_model.parameters():
        param.requires_grad = False

    # PPO Trainer expects a collator function
    def ppo_data_collator(batch):
        return {"query_tensors": [item["query_tensors"] for item in batch]}

    ppo_trainer = PPOTrainer(
        ppo_config,
        base_lm_model,
        ref_model,
        ppo_tokenizer,
        ppo_dataset,
        data_collator=ppo_data_collator,
    )
    
    # Simulation of RLHF training loop
    print("Starting mock RLHF training...")
    for epoch in range(ppo_config.ppo_epochs):
        for batch in ppo_trainer.dataloader:
            query_tensors = batch["query_tensors"]
            
            # Generate responses
            generation_kwargs = {
                "min_length": -1,
                "top_k": 0.0,
                "top_p": 1.0,
                "do_sample": True,
                "pad_token_id": ppo_tokenizer.eos_token_id,
                "max_new_tokens": 50
            }
            response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
            
            # Decode for reward model input
            responses = [ppo_tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors]
            queries_decoded = [ppo_tokenizer.decode(q.squeeze(), skip_special_tokens=True) for q in query_tensors]
            
            # Compute rewards using the trained reward model
            rewards = []
            for q, r in zip(queries_decoded, responses):
                tokenized_qr = reward_model.tokenizer(q, r, padding=True, truncation=True, return_tensors="pt", max_length=128)
                with torch.no_grad():
                    reward = reward_model(tokenized_qr["input_ids"], tokenized_qr["attention_mask"]).squeeze().item()
                rewards.append(torch.tensor(reward))
            rewards = torch.tensor(rewards).to(base_lm_model.device) # Ensure rewards are tensors
            
            # Train PPO (mocked if TRL not installed)
            train_stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            print(f"Epoch: {epoch}, Batch trained. Stats (mock): {train_stats}")
    
    return base_lm_model

# --- 5. Chatbot API & Interface ---
app = FastAPI()

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str

# Global variables for models (loaded once)
chatbot_lm = None
chatbot_tokenizer = None

@app.on_event("startup")
async def startup_event():
    global chatbot_lm, chatbot_tokenizer
    print("Loading chatbot model for API...")
    # This should load the RLHF-tuned model if training happened
    # For demonstration, we'll load the base LM for now
    chatbot_lm = AutoModelForCausalLM.from_pretrained(BASE_LM_MODEL_NAME)
    chatbot_tokenizer = AutoTokenizer.from_pretrained(BASE_LM_MODEL_NAME)
    chatbot_tokenizer.pad_token = chatbot_tokenizer.eos_token
    chatbot_lm.eval()
    print("Chatbot model loaded.")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    inputs = chatbot_tokenizer(request.query, return_tensors="pt", padding=True, truncation=True)
    
    # Generate a response
    outputs = chatbot_lm.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=50,
        num_return_sequences=1,
        pad_token_id=chatbot_tokenizer.eos_token_id,
        do_sample=True,  # For more varied responses
        top_k=50,        # Top-k sampling
        top_p=0.95       # Nucleus sampling
    )
    
    # Decode the generated response, excluding the input query part
    generated_text = chatbot_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the input query from the generated text if it's prepended
    if generated_text.startswith(request.query):
        response_text = generated_text[len(request.query):].strip()
    else:
        response_text = generated_text.strip()
        
    return ChatResponse(response=response_text)

# --- Main Execution Block ---
if __name__ == "__main__":
    # Simulate Data Collection
    print("Generating synthetic human preference data...")
    preference_data = generate_synthetic_preference_data(num_samples=200)
    print(f"Generated {len(preference_data)} samples.")
    print("Sample preference data:", preference_data[0])

    # Initialize Base Language Model
    print("Loading Base Language Model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_lm_tokenizer = AutoTokenizer.from_pretrained(BASE_LM_MODEL_NAME)
    base_lm_model = AutoModelForCausalLM.from_pretrained(BASE_LM_MODEL_NAME).to(device)
    base_lm_tokenizer.pad_token = base_lm_tokenizer.eos_token

    # Initialize and Train Reward Model
    print("Initializing and training Reward Model...")
    reward_model = RewardModel(REWARD_MODEL_NAME).to(device)
    reward_model = train_reward_model(preference_data, reward_model)
    print("Reward Model training complete.")

    # Perform RLHF Training (mocked if trl not installed)
    print("Starting RLHF training for Chatbot LM...")
    # The actual base_lm_model will be updated in-place by PPOTrainer if trl is installed
    fine_tuned_chatbot_lm = train_rlhf_model(base_lm_model, base_lm_tokenizer, reward_model, preference_data)
    print("RLHF training (or mock) complete.")

    # Deploy Chatbot API
    print("Starting FastAPI application...")
    # Uvicorn will load the global chatbot_lm and chatbot_tokenizer on startup
    # In a real scenario, you'd save fine_tuned_chatbot_lm and load it here
    # For this consolidated script, we rely on the global assignment within startup_event
    uvicorn.run(app, host="0.0.0.0", port=8000)
