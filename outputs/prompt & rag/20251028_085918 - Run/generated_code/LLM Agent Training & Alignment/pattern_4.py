
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI
from pydantic import BaseModel
import random

# --- 1. Core Language Model (LLM) and Tokenizer Loading ---
# In a real-world scenario, you would load your fine-tuned SFT/RLHF model here.
# For demonstration, we'll use a small pre-trained model.
MODEL_NAME = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Add a pad token if the tokenizer doesn't have one (common for generative models)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

# --- 2. Data Handling and Preprocessing (Conceptual) ---
# This section would typically involve loading and preparing datasets
# for SFT, reward modeling, and RLHF. For this code, we assume data
# has been processed and models are ready.
# Example: datasets.load_dataset("path/to/my_demonstrations")
# Example: pandas.read_csv("preference_comparisons.csv")

# --- 3. Behavior Cloning (SFT) - Conceptual ---
# In a real application, you would fine-tune the base LLM here
# using expert human demonstrations. Example with `trl`:
# from trl import SFTTrainer
# trainer = SFTTrainer(model, tokenizer, train_dataset=sft_dataset, ...)
# trainer.train()
# sft_model = model

# --- 4. Reward Model Training - Conceptual ---
# Train a separate reward model based on human preference comparisons.
# from trl import RewardTrainer
# reward_model = AutoModelForSequenceClassification.from_pretrained("path/to/reward_model_base")
# reward_tokenizer = AutoTokenizer.from_pretrained("path/to/reward_model_base")
# reward_trainer = RewardTrainer(reward_model, reward_tokenizer, train_dataset=reward_dataset, ...)
# reward_trainer.train()

# For this demo, we'll simulate a reward model using a simple heuristic.
class SimulatedRewardModel:
    def __init__(self):
        pass

    def get_reward(self, query: str, response: str) -> float:
        """
        Simulates a reward score for a given response. Higher is better.
        In a real scenario, this would be a fine-tuned model predicting human preference.
        """
        # Simple heuristic: longer responses are better, and responses containing
        # "thank you" or "resolved" get a bonus.
        reward = len(response) * 0.1
        if "thank you" in response.lower() or "resolved" in response.lower():
            reward += 1.0
        if "i cannot help" in response.lower() or "error" in response.lower():
            reward -= 2.0
        return reward + random.uniform(-0.5, 0.5) # Add some noise

reward_model = SimulatedRewardModel()

# --- 5. Reinforcement Learning from Human Feedback (RLHF) - Conceptual ---
# Further fine-tune the SFT model using the reward model.
# from trl import PPOTrainer, DPOptimizer
# ppo_trainer = PPOTrainer(model, ref_model, tokenizer, dataset, reward_model=reward_model, ...)
# ppo_trainer.train()
# rlhf_model = model

# --- 6. Rejection Sampling (Best-of-N at Inference) ---
def generate_and_rank_responses(query: str, num_candidates: int = 5, max_length: int = 100) -> str:
    """
    Generates N candidate responses and selects the best one using the reward model.
    """
    input_ids = tokenizer.encode(query, return_tensors="pt")
    best_response = ""
    highest_reward = -float('inf')

    print(f"Generating {num_candidates} candidates for query: '{query}'")

    for i in range(num_candidates):
        # Generate a candidate response
        # We add top_k and do_sample for more diverse outputs for rejection sampling
        output_ids = model.generate(
            input_ids, 
            max_length=max_length + len(input_ids[0]), 
            num_return_sequences=1, 
            do_sample=True, 
            top_k=50, 
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id
        )
        candidate_response = tokenizer.decode(output_ids[0][len(input_ids[0]):], skip_special_tokens=True)
        
        # Get reward for the candidate
        current_reward = reward_model.get_reward(query, candidate_response)
        print(f"  Candidate {i+1} (Reward: {current_reward:.2f}): {candidate_response}")

        if current_reward > highest_reward:
            highest_reward = current_reward
            best_response = candidate_response
    
    print(f"Selected best response (Reward: {highest_reward:.2f}): {best_response}")
    return best_response

# --- 7. Sample-Efficient RL with Reference Reuse (Conceptual) ---
# This is primarily an optimization strategy within the RLHF training loop.
# It would involve careful design of replay buffers and sampling strategies
# within the `trl` PPOTrainer or similar frameworks.

# --- 8. Dual Data Collection & Continuous Learning Pipeline (Conceptual) ---
# This involves infrastructure for collecting new human demonstrations and preferences
# and periodically retraining/updating the models. Not directly implemented in code.

# --- 9. Deployment and API with FastAPI ---
app = FastAPI(
    title="Smart Customer Support Agent API",
    description="An API for a customer support agent leveraging advanced AI patterns.",
    version="1.0.0",
)

class QueryRequest(BaseModel):
    query: str
    num_candidates: int = 5 # Number of responses to generate for rejection sampling
    max_response_length: int = 100 # Maximum length of the generated response

class AgentResponse(BaseModel):
    response: str
    model_used: str
    technique_applied: str = "Rejection Sampling (Best-of-N)"

@app.post("/chat", response_model=AgentResponse, summary="Get a smart response from the agent")
async def chat_with_agent(request: QueryRequest):
    """
    Sends a customer query to the Smart Customer Support Agent and gets a response
    optimized using Rejection Sampling.
    """
    # Use the rejection sampling function to get the best response
    best_response = generate_and_rank_responses(
        query=request.query,
        num_candidates=request.num_candidates,
        max_length=request.max_response_length
    )

    return AgentResponse(
        response=best_response,
        model_used=MODEL_NAME
    )

@app.get("/health", summary="Check API health")
async def health_check():
    return {"status": "ok", "model_loaded": MODEL_NAME}

# To run this API:
# 1. Save the code as `smart_customer_support_agent.py`
# 2. Install dependencies: `pip install transformers fastapi uvicorn torch`
# 3. Run from your terminal: `uvicorn smart_customer_support_agent:app --reload`
# 4. Access the API at http://127.0.0.1:8000/docs for interactive documentation.
