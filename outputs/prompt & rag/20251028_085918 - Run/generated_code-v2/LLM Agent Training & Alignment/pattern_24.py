from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import torch
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from datasets import Dataset
import random

# --- Database Setup ---
DATABASE_URL = "sqlite:///./sql_app.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Preference(Base):
    __tablename__ = "preferences"
    id = Column(Integer, primary_key=True, index=True)
    prompt = Column(Text, nullable=False)
    response_a = Column(Text, nullable=False)
    response_b = Column(Text, nullable=False)
    preferred_response = Column(String, nullable=False) # 'A' or 'B'

Base.metadata.create_all(bind=engine)

# --- Core Language Model (Chatbot LLM) ---
# Using a small, pre-trained model for demonstration. In a real scenario, this would be a large generative LLM.
# For actual generation, you'd use a model like 'gpt2', 'microsoft/DialoGPT-small', or a fine-tuned Llama-2 variant.
# For this example, we'll mock the generation or use a very basic text generation pipeline.

try:
    chatbot_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    # Using a causal LM for generation if possible, otherwise mock
    chatbot_model = AutoModelForCausalLM.from_pretrained("gpt2") # Placeholder, will use dummy generate for simplicity
except Exception:
    print("Could not load gpt2 or distilbert-base-uncased. Using dummy models.")
    class DummyTokenizer:
        def __call__(self, text, return_tensors=None, padding=True, truncation=True): return {"input_ids": torch.tensor([[1,2,3]]), "attention_mask": torch.tensor([[1,1,1]])}
        def decode(self, tokens): return "Dummy response"
    class DummyModel:
        def generate(self, input_ids, max_new_tokens=50, **kwargs): return input_ids # Mock generation
    chatbot_tokenizer = DummyTokenizer()
    chatbot_model = DummyModel()


def generate_chatbot_response(prompt: str) -> str:
    if isinstance(chatbot_model, AutoModelForCausalLM):
        inputs = chatbot_tokenizer(prompt, return_tensors="pt")
        outputs = chatbot_model.generate(inputs["input_ids"], max_new_tokens=50, pad_token_id=chatbot_tokenizer.eos_token_id)
        return chatbot_tokenizer.decode(outputs[0], skip_special_tokens=True)
    else:
        return f"Chatbot response to: {prompt} (DUMMY GENERATION)"

# --- Reward Model (RM) ---
# Using a small sequence classification model for demonstration of RM structure.
# In a real scenario, this would be trained to output a scalar reward based on prompt+response pair.

try:
    reward_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    reward_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=1) # One label for reward score
except Exception:
    print("Could not load distilbert-base-uncased for reward model. Using dummy models.")
    class DummyRewardTokenizer:
        def __call__(self, text_pair, return_tensors=None, padding=True, truncation=True): return {"input_ids": torch.tensor([[1,2,3,4,5]]), "attention_mask": torch.tensor([[1,1,1,1,1]])}
    class DummyRewardModel(torch.nn.Module):
        def __init__(self): super().__init__(); self.linear = torch.nn.Linear(5, 1)
        def forward(self, input_ids, attention_mask=None): return torch.tensor([[random.uniform(-1, 1)]]) # Mock reward
    reward_tokenizer = DummyRewardTokenizer()
    reward_model = DummyRewardModel()

def get_reward_score(prompt: str, response: str) -> float:
    inputs = reward_tokenizer(prompt, response, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = reward_model(**inputs)
        # Assuming a single scalar output for reward
        return outputs.logits.squeeze().item() if hasattr(outputs, 'logits') else outputs.squeeze().item()

# --- FastAPI Application ---
app = FastAPI()

class ChatRequest(BaseModel):
    prompt: str

class FeedbackRequest(BaseModel):
    prompt: str
    response_a: str
    response_b: str
    preferred_response: str # 'A' or 'B'

@app.post("/chat")
async def chat_with_bot(request: ChatRequest):
    response = generate_chatbot_response(request.prompt)
    return {"response": response}

@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    db = SessionLocal()
    new_preference = Preference(
        prompt=request.prompt,
        response_a=request.response_a,
        response_b=request.response_b,
        preferred_response=request.preferred_response
    )
    db.add(new_preference)
    db.commit()
    db.refresh(new_preference)
    db.close()
    return {"message": "Feedback submitted successfully", "feedback_id": new_preference.id}

@app.post("/train_reward_model")
async def train_reward_model():
    db = SessionLocal()
    preferences = db.query(Preference).all()
    db.close()

    if not preferences:
        raise HTTPException(status_code=400, detail="No feedback data available for training reward model.")

    # Simulate RM training. In a real scenario, this would involve a full training loop with PyTorch/TRL.
    # For simplicity, we just print a message.
    print(f"Simulating Reward Model training with {len(preferences)} samples...")
    # Example of how you might prepare data for TRL's RewardTrainer (conceptual)
    # This part is highly simplified and not executable without a proper TRL setup for reward modeling.
    # dummy_dataset = Dataset.from_dict({
    #     "prompt": [p.prompt for p in preferences],
    #     "response_j": [p.response_a if p.preferred_response == 'A' else p.response_b for p in preferences],
    #     "response_k": [p.response_b if p.preferred_response == 'A' else p.response_a for p in preferences],
    # })
    # reward_trainer = RewardTrainer(...)
    # reward_trainer.train()
    return {"message": f"Reward Model training simulated with {len(preferences)} feedback entries."}

@app.post("/finetune_llm_rlhf")
async def finetune_llm_rlhf():
    db = SessionLocal()
    preferences = db.query(Preference).all()
    db.close()

    if not preferences:
        raise HTTPException(status_code=400, detail="No feedback data available for RLHF finetuning.")

    # Simulate RLHF finetuning (PPO). This is highly conceptual.
    # In a real scenario, this would involve a PPO trainer from TRL, interacting with the RM and LLM.
    print(f"Simulating RLHF finetuning with {len(preferences)} feedback entries...")

    # Conceptual setup for TRL's PPOTrainer
    # This requires an SFT-trained model as a base and the trained reward model.
    # ppo_config = PPOConfig(
    #     model_name="gpt2", # Or the path to your SFT-tuned model
    #     learning_rate=1e-5,
    #     log_with="wandb", # If wandb is configured
    # )
    # model = AutoModelForCausalLMWithValueHead.from_pretrained(chatbot_model_path) # Assumes base LLM has a value head
    # ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(chatbot_model_path)
    # ppo_trainer = PPOTrainer(
    #     config=ppo_config,
    #     model=model,
    #     ref_model=ref_model,
    #     tokenizer=chatbot_tokenizer,
    #     dataset=dummy_dataset_for_ppo, # Dataset of prompts for RL
    #     reward_model=reward_model, # The actual trained reward model
    # )
    # ppo_trainer.train()
    return {"message": f"LLM finetuning with RLHF simulated using {len(preferences)} feedback entries."}

# To run this application:
# 1. Save the code as `main.py` (or any other Python file name).
# 2. Install required libraries: `pip install fastapi uvicorn sqlalchemy pydantic transformers torch trl datasets`
# 3. Run from your terminal: `uvicorn main:app --reload`
# 4. Access the API at http://127.0.0.1:8000
# You can use tools like curl or Postman to send requests to /chat, /feedback, /train_reward_model, /finetune_llm_rlhf.

# Example usage (after running uvicorn):
# Chat:
# curl -X POST -H "Content-Type: application/json" -d '{"prompt": "What is the capital of France?"}' http://127.0.0.1:8000/chat

# Feedback:
# curl -X POST -H "Content-Type: application/json" -d '{"prompt": "How to reset my password?", "response_a": "Go to settings and click reset.", "response_b": "Visit our support page and follow the 'forgot password' link.", "preferred_response": "B"}' http://127.0.0.1:8000/feedback

# Train RM (after some feedback):
# curl -X POST http://127.0.0.1:8000/train_reward_model

# Finetune LLM (after RM training and more feedback):
# curl -X POST http://127.0.0.1:8000/finetune_llm_rlhf
