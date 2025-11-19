import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import random
import os

# --- Configuration and Model Loading ---
# In a real application, models would be loaded from persistent storage after training
# For demonstration, we'll use a small pre-trained model and simulate fine-tuning.

# Set environment variable to disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Define the base LLM model
MODEL_NAME = "distilgpt2"
try:
    llm_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    llm_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    # Add a pad token if it doesn't exist, common for GPT-like models in generation scenarios
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
        llm_model.config.pad_token_id = llm_tokenizer.eos_token_id
except Exception as e:
    print(f"Error loading LLM: {e}. Please ensure you have an internet connection or the model is cached.")
    llm_tokenizer, llm_model = None, None

# --- 1. Core LLM Component ---
class LLMGenerator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        if model and tokenizer:
            self.generator_pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
        else:
            self.generator_pipeline = None

    def generate_responses(self, query: str, num_returns: int = 3, max_length: int = 100) -> list[str]:
        if not self.generator_pipeline:
            return ["Error: LLM not loaded properly."] * num_returns

        prompt = f"Customer query: {query}\nAgent response:"
        try:
            outputs = self.generator_pipeline(
                prompt,
                num_return_sequences=num_returns,
                max_length=max_length,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
            responses = [output['generated_text'][len(prompt):].strip() for output in outputs]
            return [resp.split('\n')[0].strip() for resp in responses] # Take only the first line as response
        except Exception as e:
            print(f"Error during LLM generation: {e}")
            return [f"Error generating response: {e}"] * num_returns

# --- 2. Behavior Cloning (BC) Module ---
# In a real scenario, this would involve loading a dataset and fine-tuning `llm_model`.
# For this demonstration, we'll assume the `llm_model` has already been 'cloned' or is a good base.
class BehaviorCloningTrainer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def fine_tune_bc(self, demonstrations: list[dict], epochs: int = 1):
        if not self.model or not self.tokenizer:
            print("Cannot fine-tune: LLM not loaded.")
            return
        print(f"Simulating Behavior Cloning fine-tuning with {len(demonstrations)} examples for {epochs} epochs.")
        print("In a real setup, this would involve actual model training (e.g., using `Trainer` from Hugging Face).")
        # Placeholder for actual fine-tuning logic
        # e.g., using `transformers.Trainer` with a `Dataset` from `demonstrations`
        # self.model.save_pretrained("bc_fine_tuned_model")
        # self.tokenizer.save_pretrained("bc_fine_tuned_model")
        self.is_trained = True
        print("Behavior Cloning simulation complete.")

# --- 3. Reward Model (RM) Module ---
# This is a simplified Reward Model that assigns a heuristic score.
# A real RM would be a trained model that outputs a score based on quality.
class RewardModel:
    def __init__(self):
        print("Initializing placeholder Reward Model.")

    def get_reward(self, query: str, response: str) -> float:
        # Simulate a reward based on length and keywords (very simplistic)
        score = 0.0
        if "thank you" in response.lower() or "apologize" in response.lower():
            score += 0.5
        if len(response) > 20: # Encourage more detailed responses
            score += 0.2
        if "error" in response.lower() or "sorry" in response.lower(): # Penalize generic errors
            score -= 0.3
        
        # Further refine based on query-response match (placeholder)
        if any(keyword in response.lower() for keyword in query.lower().split()):
             score += 0.1

        return max(0.0, score + random.uniform(-0.1, 0.2)) # Add some randomness

# --- 4. Reinforcement Learning (RL) / Rejection Sampling Module ---
class RejectionSampler:
    def __init__(self, llm_generator: LLMGenerator, reward_model: RewardModel):
        self.llm_generator = llm_generator
        self.reward_model = reward_model

    def select_best_response(self, query: str, num_candidates: int = 5) -> str:
        candidate_responses = self.llm_generator.generate_responses(query, num_candidates)
        if not candidate_responses or "Error" in candidate_responses[0]:
            return candidate_responses[0] if candidate_responses else "No response generated."

        scored_responses = []
        for resp in candidate_responses:
            reward = self.reward_model.get_reward(query, resp)
            scored_responses.append((resp, reward))
        
        # Sort by reward and return the best one
        best_response = max(scored_responses, key=lambda x: x[1])
        print(f"Selected response (reward: {best_response[1]:.2f}): {best_response[0]}")
        return best_response[0]

# --- 5. Sample-Efficient RL with Reference Reuse Module ---
class ReferenceBuffer:
    def __init__(self):
        self.high_quality_references = [] # Stores {'query': ..., 'response': ..., 'reward': ...}

    def add_reference(self, query: str, response: str, reward: float):
        # Only add if it meets a certain quality threshold (e.g., high reward)
        if reward > 0.6: # Arbitrary threshold for 'high quality'
            self.high_quality_references.append({'query': query, 'response': response, 'reward': reward})
            print(f"Added high-quality reference. Buffer size: {len(self.high_quality_references)}")

    def get_relevant_references(self, query: str, top_k: int = 2) -> list[dict]:
        # In a real system, this would involve embedding and similarity search
        # For simplicity, we return random or all references
        if not self.high_quality_references:
            return []
        
        # Simple heuristic: return random references
        return random.sample(self.high_quality_references, min(top_k, len(self.high_quality_references)))

# --- 6. Dual Data Collection Module ---
class DataCollector:
    def __init__(self):
        self.demonstrations = [] # For Behavior Cloning: {'query': ..., 'expert_response': ...}
        self.comparisons = []    # For Reward Model: {'query': ..., 'response_a': ..., 'response_b': ..., 'preferred': ...}

    def log_demonstration(self, query: str, expert_response: str):
        self.demonstrations.append({'query': query, 'expert_response': expert_response})
        print(f"Logged demonstration. Total: {len(self.demonstrations)}")

    def log_comparison(self, query: str, response_a: str, response_b: str, preferred: str):
        self.comparisons.append({
            'query': query,
            'response_a': response_a,
            'response_b': response_b,
            'preferred': preferred
        })
        print(f"Logged comparison. Total: {len(self.comparisons)}")

# --- 7. Agent Orchestration / API Layer (FastAPI) ---
app = FastAPI(title="Intelligent Customer Support Agent")

# Initialize components
llm_generator = LLMGenerator(llm_model, llm_tokenizer)
reward_model = RewardModel()
rejection_sampler = RejectionSampler(llm_generator, reward_model)
bc_trainer = BehaviorCloningTrainer(llm_model, llm_tokenizer)
data_collector = DataCollector()
reference_buffer = ReferenceBuffer()

# Pydantic models for API requests and responses
class QueryRequest(BaseModel):
    customer_query: str

class QueryResponse(BaseModel):
    agent_response: str

class LogDemonstrationRequest(BaseModel):
    query: str
    expert_response: str

class LogComparisonRequest(BaseModel):
    query: str
    response_a: str
    response_b: str
    preferred_response: str # Should be either response_a or response_b

@app.on_event("startup")
async def startup_event():
    if not llm_model or not llm_tokenizer:
        print("CRITICAL: LLM model or tokenizer failed to load at startup. Check internet/model path.")
        # Optionally exit or disable query functionality

@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    if not llm_generator.generator_pipeline:
        raise HTTPException(status_code=500, detail="LLM service not ready. Model failed to load.")

    print(f"Received query: {request.customer_query}")
    agent_response = rejection_sampler.select_best_response(request.customer_query)
    
    # In a real system, you might log the interaction and reward here for continuous improvement
    # For simplicity, we'll log successful responses to the reference buffer if they seem good.
    final_reward = reward_model.get_reward(request.customer_query, agent_response)
    reference_buffer.add_reference(request.customer_query, agent_response, final_reward)

    return QueryResponse(agent_response=agent_response)

@app.post("/log_demonstration")
async def log_demo(request: LogDemonstrationRequest):
    data_collector.log_demonstration(request.query, request.expert_response)
    # After collecting enough demonstrations, one might trigger a BC retraining
    # bc_trainer.fine_tune_bc(data_collector.demonstrations)
    return {"message": "Demonstration logged successfully"}

@app.post("/log_comparison")
async def log_comp(request: LogComparisonRequest):
    if request.preferred_response not in [request.response_a, request.response_b]:
        raise HTTPException(status_code=400, detail="Preferred response must be one of response_a or response_b.")
    data_collector.log_comparison(request.query, request.response_a, request.response_b, request.preferred_response)
    # After collecting enough comparisons, one might trigger Reward Model retraining
    return {"message": "Comparison logged successfully"}

@app.get("/status")
async def get_status():
    return {
        "llm_loaded": llm_model is not None,
        "demonstrations_count": len(data_collector.demonstrations),
        "comparisons_count": len(data_collector.comparisons),
        "reference_buffer_count": len(reference_buffer.high_quality_references)
    }


if __name__ == "__main__":
    import uvicorn
    print(f"Starting FastAPI app. Model used: {MODEL_NAME}")
    print("To run, save this code as customer_support_agent.py and execute: uvicorn customer_support_agent:app --reload")
    print("Then access the API at http://127.0.0.1:8000/docs for interactive documentation.")
    # For local development, you might want to call bc_trainer.fine_tune_bc() here with some dummy data
    # Example dummy demonstrations for BC
    dummy_demonstrations = [
        {"query": "My order is late.", "expert_response": "I apologize for the delay. Could you please provide your order number?"},
        {"query": "How do I return an item?", "expert_response": "You can initiate a return from your order history page. Please note our return policy allows returns within 30 days of purchase."},
    ]
    # bc_trainer.fine_tune_bc(dummy_demonstrations)

    uvicorn.run(app, host="0.0.0.0", port=8000)
