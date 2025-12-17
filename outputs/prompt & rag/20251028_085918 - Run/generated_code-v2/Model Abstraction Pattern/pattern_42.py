import os
from abc import ABC, abstractmethod
from typing import List, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from loguru import logger

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI


# --- config.py ---
class Settings(BaseSettings):
    OPENAI_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    DEFAULT_LLM_PROVIDER: str = "gemini"
    FALLBACK_LLM_PROVIDER: str = "openai"

    class Config:
        env_file = ".env"
        extra = "ignore"

load_dotenv()
settings = Settings()

logger.info("Settings loaded.")


# --- llm_providers.py ---
class AbstractLLMProvider(ABC):
    @abstractmethod
    def generate_response(self, prompt: str, history: List[Dict]) -> str:
        pass

    def _convert_history_to_langchain_format(self, history: List[Dict]) -> List[BaseMessage]:
        langchain_history = []
        for msg in history:
            if msg["role"] == "user":
                langchain_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_history.append(AIMessage(content=msg["content"]))
        return langchain_history


class OpenAIChatProvider(AbstractLLMProvider):
    def __init__(self):
        if not settings.OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY not found. OpenAI provider will not be available.")
            self.client = None
        else:
            self.client = ChatOpenAI(api_key=settings.OPENAI_API_KEY, model="gpt-3.5-turbo")
            logger.info("OpenAIChatProvider initialized.")

    def generate_response(self, prompt: str, history: List[Dict]) -> str:
        if not self.client:
            raise ValueError("OpenAI provider is not configured.")
        try:
            langchain_history = self._convert_history_to_langchain_format(history)
            messages = langchain_history + [HumanMessage(content=prompt)]
            response = self.client.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Error with OpenAI provider: {e}")
            raise


class GoogleGeminiChatProvider(AbstractLLMProvider):
    def __init__(self):
        if not settings.GOOGLE_API_KEY:
            logger.warning("GOOGLE_API_KEY not found. Google Gemini provider will not be available.")
            self.client = None
        else:
            self.client = ChatGoogleGenerativeAI(google_api_key=settings.GOOGLE_API_KEY, model="gemini-pro")
            logger.info("GoogleGeminiChatProvider initialized.")

    def generate_response(self, prompt: str, history: List[Dict]) -> str:
        if not self.client:
            raise ValueError("Google Gemini provider is not configured.")
        try:
            langchain_history = self._convert_history_to_langchain_format(history)
            messages = langchain_history + [HumanMessage(content=prompt)]
            response = self.client.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Error with Google Gemini provider: {e}")
            raise


# --- llm_router.py ---
class LLMRouter:
    def __init__(self):
        self.providers = {
            "openai": OpenAIChatProvider(),
            "gemini": GoogleGeminiChatProvider(),
        }
        logger.info("LLMRouter initialized with available providers.")

    def select_llm_provider(self, query: str, history: List[Dict]) -> AbstractLLMProvider:
        # Basic routing logic: Try default, then fallback
        selected_provider_name = settings.DEFAULT_LLM_PROVIDER.lower()
        provider = self.providers.get(selected_provider_name)

        if provider and provider.client:
            logger.info(f"Attempting to use default provider: {selected_provider_name}")
            return provider
        else:
            logger.warning(f"Default provider '{selected_provider_name}' not available or configured. Attempting fallback.")
            fallback_provider_name = settings.FALLBACK_LLM_PROVIDER.lower()
            fallback_provider = self.providers.get(fallback_provider_name)

            if fallback_provider and fallback_provider.client:
                logger.info(f"Using fallback provider: {fallback_provider_name}")
                return fallback_provider
            else:
                logger.error(f"Fallback provider '{fallback_provider_name}' not available or configured. No LLM provider found.")
                raise ValueError("No functional LLM provider available.")


# --- main.py ---
app = FastAPI()
llm_router = LLMRouter()


class ChatRequest(BaseModel):
    message: str
    conversation_history: List[Dict] = []


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        logger.info(f"Received chat request: {request.message}")

        provider = llm_router.select_llm_provider(request.message, request.conversation_history)
        
        response_content = provider.generate_response(request.message, request.conversation_history)
        logger.info(f"LLM response generated successfully.")
        return {"response": response_content}
    except ValueError as e:
        logger.error(f"Configuration Error: {e}")
        raise HTTPException(status_code=503, detail=f"Service Unavailable: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
