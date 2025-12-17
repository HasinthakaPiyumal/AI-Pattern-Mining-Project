from abc import ABC, abstractmethod
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 1. LLM Abstraction Layer
class AbstractLLM(ABC):
    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        pass

class GPTLLM(AbstractLLM):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def generate_response(self, prompt: str) -> str:
        # Simulate OpenAI GPT API call
        return f"GPT Response (using key: {self.api_key[:5]}...): {prompt[::-1]}"

class GeminiLLM(AbstractLLM):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def generate_response(self, prompt: str) -> str:
        # Simulate Google Gemini API call
        return f"Gemini Response (using key: {self.api_key[:5]}...): {prompt.upper()}"

class LlamaLLM(AbstractLLM):
    def __init__(self, model_endpoint: str = "http://localhost:8000/llama/inference"):
        self.model_endpoint = model_endpoint

    def generate_response(self, prompt: str) -> str:
        # Simulate Llama API call (e.g., to a self-hosted or Hugging Face endpoint)
        return f"Llama Response (via {self.model_endpoint}): {prompt.lower()}"

# 2. LLM Manager
class LLMManager:
    def __init__(self, default_provider: str = "GPT"):
        self.default_provider = default_provider.upper()

    def get_llm_instance(self, provider: str = None) -> AbstractLLM:
        provider_to_use = (provider or self.default_provider).upper()

        if provider_to_use == "GPT":
            gpt_api_key = os.getenv("GPT_API_KEY", "sk-mock-gpt-key")
            return GPTLLM(api_key=gpt_api_key)
        elif provider_to_use == "GEMINI":
            gemini_api_key = os.getenv("GEMINI_API_KEY", "mock-gemini-key")
            return GeminiLLM(api_key=gemini_api_key)
        elif provider_to_use == "LLAMA":
            llama_endpoint = os.getenv("LLAMA_MODEL_ENDPOINT", "http://localhost:8000/llama/inference")
            return LlamaLLM(model_endpoint=llama_endpoint)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider_to_use}")

# 3. Chatbot Service
class ChatbotService:
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager

    def get_chatbot_response(self, query: str, llm_provider: str = None) -> str:
        llm = self.llm_manager.get_llm_instance(provider=llm_provider)
        response = llm.generate_response(query)
        return response

# 4. API Layer (FastAPI)
app = FastAPI(title="Intelligent Customer Support Chatbot")

class ChatRequest(BaseModel):
    query: str
    llm_provider: str = None  # Optional: to dynamically select LLM

# Initialize LLMManager and ChatbotService
# Default LLM provider can be set via environment variable or directly here
default_llm = os.getenv("DEFAULT_LLM_PROVIDER", "GPT")
llm_manager_instance = LLMManager(default_provider=default_llm)
chatbot_service_instance = ChatbotService(llm_manager=llm_manager_instance)

@app.post("/chat", response_model=dict)
async def chat_endpoint(request: ChatRequest):
    try:
        response_text = chatbot_service_instance.get_chatbot_response(request.query, request.llm_provider)
        return {"response": response_text, "llm_provider_used": request.llm_provider or default_llm}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

# To run this application:
# 1. Save the code as main.py
# 2. Install dependencies: pip install fastapi uvicorn pydantic
# 3. Run from your terminal: uvicorn main:app --reload
# You can set environment variables like:
# export GPT_API_KEY="your_openai_key"
# export GEMINI_API_KEY="your_gemini_key"
# export DEFAULT_LLM_PROVIDER="GEMINI" (or GPT, LLAMA)
# To test with Llama, you might need to mock or run a local Llama inference endpoint. For this example, it's just a placeholder string.
