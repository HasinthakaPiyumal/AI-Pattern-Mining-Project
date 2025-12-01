from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams

class ChatRequest(BaseModel):
    prompt: str

class ChatResponse(BaseModel):
    response: str

app = FastAPI()

# Initialize vLLM with a pre-trained model
# Using a small model for demonstration purposes. Replace with a larger model like Mistral-7B-Instruct-v0.2 for production.
llm = LLM(model="facebook/opt-125m")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=256)
    
    # vLLM expects a list of prompts
    outputs = llm.generate(prompts=[request.prompt], sampling_params=sampling_params)
    
    generated_text = outputs[0].outputs[0].text
    
    return ChatResponse(response=generated_text)

# To run this application:
# 1. Make sure you have vLLM and FastAPI installed: pip install vllm fastapi uvicorn
# 2. Save the code as chatbot_service.py
# 3. Run from your terminal: uvicorn chatbot_service:app --host 0.0.0.0 --port 8000
# 4. You can then send POST requests to http://0.0.0.0:8000/chat with a JSON body like: {"prompt": "Hello, how are you?"}
