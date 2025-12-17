import os
from abc import ABC, abstractmethod
from typing import Dict, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from openai import OpenAI
import google.generativeai as genai
from fastapi import FastAPI

# --- config.py ---
load_dotenv()

class Settings(BaseModel):
    openai_api_key: str = Field(os.getenv("OPENAI_API_KEY", ""), env="OPENAI_API_KEY")
    gemini_api_key: str = Field(os.getenv("GEMINI_API_KEY", ""), env="GEMINI_API_KEY")
    openai_model_name: str = Field(os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo"), env="OPENAI_MODEL_NAME")
    gemini_model_name: str = Field(os.getenv("GEMINI_MODEL_NAME", "gemini-pro"), env="GEMINI_MODEL_NAME")

settings = Settings()

# --- llm_abstractor.py ---
class AbstractLLM(ABC):
    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        pass

class OpenAILLM(AbstractLLM):
    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo"):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def generate_response(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error from OpenAI: {e}"

class GeminiLLM(AbstractLLM):
    def __init__(self, api_key: str, model_name: str = "gemini-pro"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def generate_response(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error from Gemini: {e}"

class MockLLM(AbstractLLM):
    def __init__(self, model_name: str = "mock-llama"):
        self.model_name = model_name

    def generate_response(self, prompt: str) -> str:
        return f"MockLLM ({self.model_name}) received prompt: {prompt}. This is a simulated response."

# --- llm_router.py ---
class LLMManager:
    def __init__(self, llm_providers: Dict[str, AbstractLLM]):
        self.llm_providers = llm_providers
        self.default_llm = next(iter(llm_providers.values())) if llm_providers else MockLLM()

    def get_llm(self, criteria: str = "default") -> AbstractLLM:
        criteria_lower = criteria.lower()
        if criteria_lower == "cost-effective":
            return self.llm_providers.get("openai", self.default_llm) # Example: OpenAI might be considered more cost-effective for some use cases
        elif criteria_lower == "fast":
            return self.llm_providers.get("gemini", self.default_llm) # Example: Gemini might be considered faster for some use cases
        elif criteria_lower in self.llm_providers:
            return self.llm_providers[criteria_lower]
        else:
            return self.default_llm

# --- main.py ---
app = FastAPI()

class ChatRequest(BaseModel):
    query: str
    llm_preference: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    llm_used: str

# Initialize LLM providers
llm_providers_map = {}
if settings.openai_api_key:
    llm_providers_map["openai"] = OpenAILLM(settings.openai_api_key, settings.openai_model_name)
if settings.gemini_api_key:
    llm_providers_map["gemini"] = GeminiLLM(settings.gemini_api_key, settings.gemini_model_name)
llm_providers_map["mock"] = MockLLM()

llm_manager = LLMManager(llm_providers_map)

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    selected_llm = llm_manager.get_llm(request.llm_preference or "default")
    response_text = selected_llm.generate_response(request.query)
    return ChatResponse(response=response_text, llm_used=selected_llm.model_name)

# To run the app:
# 1. Save this code as multi_llm_chatbot.py
# 2. Create a .env file in the same directory with your API keys:
#    OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
#    GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
# 3. Install dependencies: pip install fastapi uvicorn pydantic python-dotenv openai google-generativeai
# 4. Run: uvicorn multi_llm_chatbot:app --reload
