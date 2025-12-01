from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import uvicorn

app = FastAPI()

llm = LLM(model="HuggingFaceH4/zephyr-7b-beta")

class ChatRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 512
    temperature: float = 0.7

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    sampling_params = SamplingParams(
        temperature=request.temperature,
        max_new_tokens=request.max_new_tokens
    )
    outputs = llm.generate([request.prompt], sampling_params)
    generated_text = outputs[0].outputs[0].text
    return {"response": generated_text}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)