from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import uvicorn
import os

# --- Configuration --- #
# Model to use for the LLM inference. Ensure this model is supported by vLLM and available.
# For demonstration, we use a Mistral-7B instruct model.
LLM_MODEL = os.getenv("LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.1")
# Adjust GPU memory utilization as needed. 0.9 (90%) is a common starting point.
GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEM_UTIL", "0.9"))

# --- FastAPI App Initialization --- #
app = FastAPI(
    title="Intelligent Customer Support LLM Inference Service",
    description="Highly efficient, fault-tolerant, and performant LLM inference for customer support applications, leveraging vLLM's KV Cache Reuse, PagedAttention, and an efficient cache strategy."
)

# --- vLLM Engine Initialization --- #
# The LLM engine is initialized globally to load the model once at startup.
# This will load the model onto the GPU.
print(f"Initializing vLLM with model: {LLM_MODEL} and GPU memory utilization: {GPU_MEMORY_UTILIZATION}")
llm = LLM(model=LLM_MODEL, gpu_memory_utilization=GPU_MEMORY_UTILIZATION)
print("vLLM engine initialized successfully.")

# --- Request Body Pydantic Model --- #
class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    n: int = 1 # Number of output sequences to generate for the given prompt.

# --- API Endpoint --- #
@app.post("/generate")
async def generate_response(request: PromptRequest):
    """
    Generates an LLM response for a given customer support prompt.

    Leverages vLLM for efficient inference, including KV Cache Reuse and PagedAttention.
    """
    print(f"Received prompt: {request.prompt[:100]}...")

    # Configure sampling parameters for vLLM
    sampling_params = SamplingParams(
        n=request.n,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        stop=["\nCustomer:", "\nAgent:"] # Example stop sequences for chat-like interaction
    )

    try:
        # Generate response using vLLM
        # vLLM automatically handles KV Cache Reuse and PagedAttention internally.
        outputs = llm.generate(request.prompt, sampling_params)

        generated_texts = []
        for output in outputs:
            # Assuming a single prompt for now, taking the first generated text
            for generation in output.outputs:
                generated_texts.append(generation.text.strip())
        
        # For a single prompt, we expect a list of generated texts (if n > 1) or a single text.
        # If n=1, return the first and only generated text.
        response_text = generated_texts[0] if request.n == 1 else generated_texts

        print(f"Generated response: {str(response_text)[:100]}...")
        return {"response": response_text}

    except Exception as e:
        print(f"Error during LLM inference: {e}")
        return Response(content=f"Internal Server Error: {e}", status_code=500)

# --- Health Check Endpoint --- #
@app.get("/health")
async def health_check():
    """
    Health check endpoint to ensure the service is running.
    """
    return {"status": "ok", "model": LLM_MODEL}

# --- Instructions to run the service --- #
if __name__ == "__main__":
    print("\nTo run the service, use uvicorn:")
    print(f"uvicorn customer_support_llm_service:app --host 0.0.0.0 --port 8000 --workers 1")
    print("\nMake sure you have vLLM and FastAPI installed:")
    print("pip install vllm fastapi uvicorn")
    print("\nEnvironment variables can be set for model and GPU memory utilization:")
    print("export LLM_MODEL=\"meta-llama/Llama-2-7b-chat-hf\"")
    print("export GPU_MEM_UTIL=\"0.8\"")

