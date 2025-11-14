from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import os

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str

# --- vLLM Model Initialization ---
# It's recommended to set environment variables or pass arguments for model path.
# For demonstration, we'll use a small, commonly available model.
# In a production environment, you'd specify your fine-tuned or chosen LLM.

# Example: os.environ["VLLM_MODEL"] = "mistralai/Mistral-7B-Instruct-v0.1"
# Or pass directly to LLM constructor

try:
    # Using a small model for demonstration. Replace with a larger model as needed.
    # Ensure the model is downloaded or accessible by vLLM.
    llm_model_name = os.getenv("VLLM_MODEL", "gpt2") # Default to gpt2 for quick demo
    print(f"Initializing vLLM with model: {llm_model_name}")
    llm = LLM(model=llm_model_name, 
              trust_remote_code=True, # Set to True if using models with custom code
              gpu_memory_utilization=0.9, # Adjust based on your GPU memory
              enforce_eager=False # Usually False for better performance
             )
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=256)
except Exception as e:
    print(f"Error initializing vLLM: {e}")
    print("Please ensure the specified model is valid and accessible, and you have sufficient GPU resources.")
    llm = None # Set to None to indicate failure
    sampling_params = None

# --- FastAPI Application ---
app = FastAPI(
    title="Intelligent Customer Support Chatbot",
    description="A live customer support chatbot platform leveraging vLLM for efficient LLM inference."
)

@app.get("/health", summary="Health Check")
async def health_check():
    """Checks the health of the application."""
    if llm is None:
        raise HTTPException(status_code=503, detail="LLM model not initialized.")
    return {"status": "healthy", "message": "Chatbot is ready."}

@app.post("/chat", response_model=ChatResponse, summary="Chat with the AI Assistant")
async def chat_with_assistant(request: QueryRequest):
    """Processes a customer query and returns an AI-generated response."""
    if llm is None or sampling_params is None:
        raise HTTPException(status_code=503, detail="LLM service is not available. Model failed to load.")
    
    try:
        # vLLM handles batching internally if multiple requests come in concurrently
        # For a single request, we pass a list containing one prompt.
        outputs = llm.generate([request.query], sampling_params)
        
        # Extract the generated text from the first (and only) output sequence
        if outputs and outputs[0].outputs:
            generated_text = outputs[0].outputs[0].text.strip()
            return ChatResponse(response=generated_text)
        else:
            raise HTTPException(status_code=500, detail="Failed to generate a response from the LLM.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during LLM inference: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # To run:
    # 1. Ensure you have vLLM installed with GPU support (`pip install vllm[cuda]`) 
    # 2. Set the VLLM_MODEL environment variable if you want to use a specific model (e.g., `export VLLM_MODEL="mistralai/Mistral-7B-Instruct-v0.1"`)
    # 3. Run this script: `python chatbot_platform.py` or `uvicorn chatbot_platform:app --host 0.0.0.0 --port 8000`
    # Access the API at http://0.0.0.0:8000/docs
    uvicorn.run(app, host="0.0.0.0", port=8000)
