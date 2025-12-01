from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import uvicorn

class ChatRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7

class ChatResponse(BaseModel):
    response: str

app = FastAPI()

# Initialize vLLM engine
# Using a small model for demonstration. Replace with your desired model.
llm = LLM(model="facebook/opt-125m")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    sampling_params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        stop=["\n", "<|im_end|>"], # Example stop sequences
    )
    
    outputs = llm.generate(request.prompt, sampling_params)
    generated_text = outputs[0].outputs[0].text
    
    return ChatResponse(response=generated_text)

if __name__ == "__main__":
    # To run this application:
    # 1. Save the code as main.py
    # 2. Install dependencies: pip install fastapi uvicorn vllm torch transformers
    # 3. Run from your terminal: uvicorn main:app --host 0.0.0.0 --port 8000
    # Make sure you have a GPU and CUDA installed for vLLM to work efficiently.
    uvicorn.run(app, host="0.0.0.0", port=8000)
