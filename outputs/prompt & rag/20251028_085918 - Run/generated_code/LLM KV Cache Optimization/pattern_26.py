from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import asyncio

# Initialize vLLM engine
llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.2")

# Initialize FastAPI app
app = FastAPI()

class ChatRequest(BaseModel):
    prompt: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512)
    
    # Generate response using vLLM
    outputs = await asyncio.to_thread(llm.generate, [request.prompt], sampling_params)
    
    # Extract the generated text
    generated_text = outputs[0].outputs[0].text
    
    return {"response": generated_text}