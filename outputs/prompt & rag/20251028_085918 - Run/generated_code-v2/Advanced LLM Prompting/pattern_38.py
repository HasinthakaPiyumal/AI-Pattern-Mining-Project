from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ChatRequest(BaseModel):
    text: str
    lang: str

class ChatResponse(BaseModel):
    response: str

def translate_to_english(text: str, lang: str) -> str:
    return f"[Translated to English from {lang}: {text}]"

def get_genai_response(english_query: str) -> str:
    return f"[GenAI English Response to: {english_query} - This is a helpful answer.]"

def translate_from_english(english_response: str, lang: str) -> str:
    return f"[Translated from English to {lang}: {english_response}]"

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    english_query = translate_to_english(request.text, request.lang)
    english_response = get_genai_response(english_query)
    final_response = translate_from_english(english_response, request.lang)
    return ChatResponse(response=final_response)