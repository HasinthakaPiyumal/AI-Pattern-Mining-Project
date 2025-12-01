from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams

# 1. Pydantic Models for Request and Response
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

# 2. FastAPI App Initialization
app = FastAPI()

# 3. vLLM Initialization (Load the Hugging Face Model)
# For a real-world scenario, you might want to load a larger model
# and consider GPU resources. facebook/opt-125m is used for demonstration.
llm = LLM(model="facebook/opt-125m")
sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=100)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # 4. Construct a prompt
    prompt = f"Customer: {request.message}\nAgent:"

    # 5. Send the prompt to vLLM for generation
    # vLLM handles PagedAttention internally for efficient KV cache management
    outputs = llm.generate([prompt], sampling_params)

    generated_text = "Error generating response."
    if outputs and outputs[0].outputs:
        generated_text = outputs[0].outputs[0].text.strip()
    
    # 6. Return the LLM's response
    return ChatResponse(response=generated_text)

# To run this application, save it as main.py and execute:
# uvicorn main:app --host 0.0.0.0 --port 8000
