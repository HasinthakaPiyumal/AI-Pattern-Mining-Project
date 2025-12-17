import os
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
import openai
import google.generativeai as genai

# --- config.py ---
load_dotenv()

class Config:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    DEFAULT_LLM: str = os.getenv("DEFAULT_LLM", "gemini") # 'gemini' or 'openai' or 'llama'

# --- models.py ---
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    model_used: str

# --- llm_abstraction.py ---
class BaseLLMProvider(ABC):
    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        pass

class OpenAIAdapter(BaseLLMProvider):
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        openai.api_key = api_key
        self.model = model

    def generate_response(self, prompt: str) -> str:
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150
            )
            return response.choices[0].message['content'].strip()
        except Exception as e:
            return f"Error from OpenAI: {e}"

class GeminiAdapter(BaseLLMProvider):
    def __init__(self, api_key: str, model: str = "gemini-pro"):
        genai.configure(api_key=api_key)
        self.model = model

    def generate_response(self, prompt: str) -> str:
        try:
            model = genai.GenerativeModel(self.model)
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error from Gemini: {e}"

class MockLlamaAdapter(BaseLLMProvider):
    def __init__(self, model: str = "mock-llama"):
        self.model = model

    def generate_response(self, prompt: str) -> str:
        return f"[Mock Llama Response for {self.model}]: Processing '{prompt}' - This is a simulated response."

# --- llm_factory.py ---
class LLMProviderFactory:
    def get_provider(self, provider_name: str) -> BaseLLMProvider:
        if provider_name.lower() == "openai":
            if not Config.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY not set in environment variables.")
            return OpenAIAdapter(api_key=Config.OPENAI_API_KEY)
        elif provider_name.lower() == "gemini":
            if not Config.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY not set in environment variables.")
            return GeminiAdapter(api_key=Config.GEMINI_API_KEY)
        elif provider_name.lower() == "llama":
            return MockLlamaAdapter()
        else:
            raise ValueError(f"Unknown LLM provider: {provider_name}")

# --- llm_service.py ---
class LLMService:
    def __init__(self):
        self.factory = LLMProviderFactory()

    def _route_llm(self, message: str) -> str:
        # Simple routing logic:
        # - Use Gemini for general queries
        # - Use OpenAI for more analytical/complex queries (example keywords)
        # - Use Llama (mock) if specific keyword is present
        if "complex analysis" in message.lower() or "technical details" in message.lower():
            return "openai"
        elif "creative story" in message.lower() or "poem" in message.lower():
            return "gemini"
        elif "internal info" in message.lower(): # Example for a specific internal model
            return "llama"
        else:
            return Config.DEFAULT_LLM # Default as per config

    def chat(self, message: str) -> tuple[str, str]:
        provider_name = self._route_llm(message)
        try:
            llm_provider = self.factory.get_provider(provider_name)
            response = llm_provider.generate_response(message)
            return response, provider_name
        except ValueError as e:
            return f"Error: {e}. Please ensure API keys are set or valid provider is selected.", "error"
        except Exception as e:
            return f"An unexpected error occurred: {e}", "error"

# --- main.py ---
app = FastAPI()
llm_service = LLMService()

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    response_content, model_used = llm_service.chat(request.message)
    return ChatResponse(response=response_content, model_used=model_used)

# --- .env.example (for reference) ---
# OPENAI_API_KEY=your_openai_api_key_here
# GEMINI_API_KEY=your_gemini_api_key_here
# DEFAULT_LLM=gemini # or openai or llama
