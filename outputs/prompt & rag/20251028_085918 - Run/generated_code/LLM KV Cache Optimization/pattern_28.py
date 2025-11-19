import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
from transformers import AutoTokenizer

VLLM_API_URL = os.getenv("VLLM_API_URL", "http://localhost:8000/generate")
LLM_MODEL_FOR_TOKENIZER = os.getenv("LLM_MODEL_FOR_TOKENIZER", "meta-llama/Llama-2-7b-hf")

app = FastAPI()

try:
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_FOR_TOKENIZER)
except Exception:
    tokenizer = None

class ChatRequest(BaseModel):
    query: str
    max_tokens: int = 128
    temperature: float = 0.7

@app.post("/chat")
async def chat_with_bot(request: ChatRequest):
    if not tokenizer:
        raise HTTPException(status_code=500, detail="Tokenizer not initialized. Check server configuration and logs.")

    payload = {
        "prompt": request.query,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(VLLM_API_URL, json=payload, timeout=60)
            response.raise_for_status()
            vllm_response = response.json()

            if vllm_response and "text" in vllm_response and len(vllm_response["text"]) > 0:
                generated_text = vllm_response["text"][0]
                return {"response": generated_text}
            else:
                raise HTTPException(status_code=500, detail="Invalid or empty response from vLLM server.")

    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"Could not connect to vLLM server: {exc}. Ensure vLLM is running at {VLLM_API_URL}.")
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=f"vLLM server error: {exc.response.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
