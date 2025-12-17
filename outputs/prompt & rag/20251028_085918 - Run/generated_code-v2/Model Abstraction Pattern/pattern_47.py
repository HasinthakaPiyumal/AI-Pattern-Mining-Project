from abc import ABC, abstractmethod
from typing import Dict
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi import FastAPI
import openai
import google.generativeai as genai

# --- llm_interface.py ---
class LLMProvider(ABC):
    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        pass

# --- gpt_llm.py ---
class GPTLLM(LLMProvider):
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")
        openai.api_key = self.api_key

    def generate_response(self, prompt: str) -> str:
        try:
            response = openai.Completion.create(
                engine="gpt-3.5-turbo-instruct",
                prompt=prompt,
                max_tokens=150
            )
            return response.choices[0].text.strip()
        except Exception as e:
            return f"Error with GPT: {e}"

# --- gemini_llm.py ---
class GeminiLLM(LLMProvider):
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-pro')

    def generate_response(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error with Gemini: {e}"

# --- llm_manager.py ---
class LLMManager:
    def __init__(self):
        self.providers: Dict[str, LLMProvider] = {
            "gpt": GPTLLM(),
            "gemini": GeminiLLM()
        }

    def get_llm_provider(self, query_complexity: str = "simple") -> LLMProvider:
        if query_complexity == "complex":
            return self.providers["gpt"]
        else:
            return self.providers["gemini"]

# --- models.py ---
class ChatRequest(BaseModel):
    query: str
    query_complexity: str = "simple"

class ChatResponse(BaseModel):
    response: str
    provider: str

# --- main.py ---
load_dotenv()

app = FastAPI()
llm_manager = LLMManager()

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        provider_instance = llm_manager.get_llm_provider(request.query_complexity)
        response_text = provider_instance.generate_response(request.query)
        provider_name = "gpt" if isinstance(provider_instance, GPTLLM) else "gemini"
        return ChatResponse(response=response_text, provider=provider_name)
    except Exception as e:
        return ChatResponse(response=f"An error occurred: {e}", provider="error")

# --- .env file content (for demonstration) ---
# OPENAI_API_KEY="your_openai_api_key_here"
# GEMINI_API_KEY="your_gemini_api_key_here"
