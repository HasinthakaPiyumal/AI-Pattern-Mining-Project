from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams

app = FastAPI()

class ChatRequest(BaseModel):
    query: str

# Initialize vLLM LLM instance globally
llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.2")

@app.post("/chat")
async def chat(request: ChatRequest):
    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=256)
    
    # vLLM expects a list of prompts
    outputs = llm.generate([request.query], sampling_params)
    
    # Extract the generated text from the first output
    generated_text = outputs[0].outputs[0].text
    
    return {"response": generated_text}