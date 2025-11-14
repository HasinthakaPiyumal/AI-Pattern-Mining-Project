
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

from vllm import LLM, SamplingParams

# Assuming llm_config.py exists and defines MODEL_NAME and GPU_MEMORY_UTILIZATION
from .llm_config import MODEL_NAME, GPU_MEMORY_UTILIZATION

app = FastAPI(
    title="LLM Customer Support Co-pilot API",
    description="API for fault-tolerant, high-performance LLM inference with advanced KV cache strategies."
)

# Initialize vLLM model globally to avoid reloading on each request
try:
    llm = LLM(model=MODEL_NAME, 
              gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
              enforce_eager=True) # enforce_eager can sometimes help with debugging or specific scenarios
    print(f"Successfully loaded LLM model: {MODEL_NAME}")
except Exception as e:
    print(f"Error loading LLM model: {e}")
    llm = None # Set to None if loading fails, and handle in endpoint

class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    n: int = 1 # Number of output sequences to generate
    stop: Optional[List[str]] = None

class GenerationResponse(BaseModel):
    text: List[str]
    
@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: PromptRequest):
    if llm is None:
        raise HTTPException(status_code=503, detail="LLM model not loaded or failed to initialize.")

    sampling_params = SamplingParams(
        n=request.n,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        stop=request.stop
    )

    try:
        # vLLM automatically handles PagedAttention and KV Cache Reuse
        outputs = llm.generate([request.prompt], sampling_params)
        
        generated_texts = []
        for output in outputs:
            for completion in output.outputs:
                generated_texts.append(completion.text)
        
        return GenerationResponse(text=generated_texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {e}")

@app.get("/health")
async def health_check():
    if llm is not None:
        return {"status": "ok", "model_loaded": True}
    return {"status": "error", "model_loaded": False, "message": "LLM not initialized"}

if __name__ == "__main__":
    # To run this file directly:
    # uvicorn app:app --host 0.0.0.0 --port 8000 --reload
    # The --reload flag is useful for development but should be removed in production.
    uvicorn.run(app, host="0.0.0.0", port=8000)
