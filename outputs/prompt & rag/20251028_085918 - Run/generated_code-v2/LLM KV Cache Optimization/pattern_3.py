from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import uvicorn
import asyncio

app = FastAPI()

# Initialize vLLM engine
# This assumes you have a compatible model available locally or accessible
# vLLM internally uses PagedAttention for efficient KV cache management
llm = LLM(model="lmsys/vicuna-7b-v1.5", 
          trust_remote_code=True, 
          tensor_parallel_size=1, 
          gpu_memory_utilization=0.9)

class QueryRequest(BaseModel):
    query: str
    session_id: str

@app.post("/chat")
async def chat_with_bot(request: QueryRequest):
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=256)
    
    # For a real application, you might manage conversation history per session_id
    # and pass it as part of the prompt.
    # For this example, we'll just process the current query.
    
    prompt = f"[INST] You are an e-commerce customer support assistant. Answer the following question concisely: {request.query} [/INST]"
    
    # vLLM can handle multiple prompts concurrently, leveraging PagedAttention
    # to efficiently manage KV cache across varying sequence lengths.
    outputs = llm.generate(prompt, sampling_params)
    
    response_text = outputs[0].outputs[0].text
    
    return {"session_id": request.session_id, "response": response_text}

@app.get("/health")
async def health_check():
    return {"status": "ok", "llm_model": llm.llm_engine.model_executor.driver_worker.model_name}


# To run this application:
# 1. Make sure you have vLLM and FastAPI installed: pip install vllm fastapi uvicorn
# 2. Save the code as chatbot_platform.py
# 3. Run from your terminal: uvicorn chatbot_platform:app --host 0.0.0.0 --port 8000
# 4. Access the API at http://localhost:8000/docs for Swagger UI.
