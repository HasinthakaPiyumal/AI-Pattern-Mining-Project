from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import uvicorn
import os

# Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "facebook/opt-125m") # A small model for demonstration
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "2048"))

# Initialize vLLM engine
# vLLM inherently uses PagedAttention for efficient KV cache management.
# This setup allows for dynamic batching and reduces memory fragmentation.
print(f"Initializing LLM with model: {MODEL_NAME}...")
llm = LLM(
    model=MODEL_NAME,
    max_model_len=MAX_MODEL_LEN,
    # gpu_memory_utilization=0.9, # Adjust as needed for your GPU
    # enforce_eager=True, # For debugging/development, might slow down inference
    # trust_remote_code=False, # Set to True if using custom model code
)
print("LLM initialized successfully.")

# Initialize FastAPI app
app = FastAPI(
    title="High-Throughput AI Customer Support Chatbot Platform",
    description="Leveraging vLLM with PagedAttention for efficient LLM serving."
)

# Pydantic model for request body
class ChatRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.95

@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Chat endpoint to generate responses from the LLM.
    The vLLM engine efficiently handles concurrent requests using PagedAttention.
    """
    try:
        sampling_params = SamplingParams(
            n=1,  # Number of output sequences to return for the given prompt
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            # stop=["\n"], # Example stop sequence
        )

        # Generate response using vLLM
        # vLLM takes care of batching and PagedAttention internally
        outputs = llm.generate(
            prompts=[request.prompt],
            sampling_params=sampling_params,
            # If you had multiple prompts to process in a single batch, you'd pass them here:
            # prompts=["What is your refund policy?", "How do I reset my password?"]
        )

        # Extract the generated text
        generated_text = outputs[0].outputs[0].text

        return {"response": generated_text.strip()}
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "ok", "model_loaded": MODEL_NAME}

# To run the application:
# 1. Install dependencies: pip install fastapi uvicorn "vllm[cuda]" pydantic
# 2. Run: uvicorn main:app --host 0.0.0.0 --port 8000
# You can then send POST requests to http://localhost:8000/chat
# Example using curl:
# curl -X POST "http://localhost:8000/chat" \
#      -H "Content-Type: application/json" \
#      -d '{"prompt": "Hello, how can I help you today?"}'

# For local development without uvicorn command (less common for production):
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
