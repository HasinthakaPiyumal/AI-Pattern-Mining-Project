from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from enum import Enum
from abc import ABC, abstractmethod
from typing import Dict
import os
from dotenv import load_dotenv
import openai
import google.generativeai as genai

# --- 1. llm_abstraction.py (Core Abstraction Layer) ---

class LLMProvider(str, Enum):
    GPT = "GPT"
    GEMINI = "GEMINI"
    MOCK = "MOCK"

class LLMConfig(BaseModel):
    api_key: str
    model_name: str

class BaseLLMAdapter(ABC):
    @abstractmethod
    def generate_text(self, prompt: str) -> str:
        pass

    @abstractmethod
    def summarize_text(self, text: str) -> str:
        pass

    @abstractmethod
    def analyze_sentiment(self, text: str) -> str:
        pass

class GPTAdapter(BaseLLMAdapter):
    def __init__(self, config: LLMConfig):
        self.client = openai.OpenAI(api_key=config.api_key)
        self.model_name = config.model_name

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"GPT API error: {e}")

    def generate_text(self, prompt: str) -> str:
        return self._call_llm("You are a helpful assistant.", prompt)

    def summarize_text(self, text: str) -> str:
        return self._call_llm("Summarize the following text concisely.", text)

    def analyze_sentiment(self, text: str) -> str:
        return self._call_llm("Analyze the sentiment of the following text. Respond with POSITIVE, NEGATIVE, or NEUTRAL.", text)

class GeminiAdapter(BaseLLMAdapter):
    def __init__(self, config: LLMConfig):
        genai.configure(api_key=config.api_key)
        self.model = genai.GenerativeModel(config.model_name)

    def _call_llm(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Gemini API error: {e}")

    def generate_text(self, prompt: str) -> str:
        return self._call_llm(f"You are a helpful assistant. {prompt}")

    def summarize_text(self, text: str) -> str:
        return self._call_llm(f"Summarize the following text concisely: {text}")

    def analyze_sentiment(self, text: str) -> str:
        return self._call_llm(f"Analyze the sentiment of the following text. Respond with POSITIVE, NEGATIVE, or NEUTRAL: {text}")

class MockAdapter(BaseLLMAdapter):
    def __init__(self, config: LLMConfig):
        self.config = config # config might be used for logging/mocking specific responses

    def generate_text(self, prompt: str) -> str:
        return f"Mock Response for: {prompt[:50]}... (using {self.config.model_name})"

    def summarize_text(self, text: str) -> str:
        return f"Mock Summary for: {text[:50]}... (using {self.config.model_name})"

    def analyze_sentiment(self, text: str) -> str:
        return f"Mock Sentiment for: {text[:50]}... (using {self.config.model_name}) - NEUTRAL"

class LLMFactory:
    @staticmethod
    def get_adapter(provider: LLMProvider, config: LLMConfig) -> BaseLLMAdapter:
        if provider == LLMProvider.GPT:
            return GPTAdapter(config)
        elif provider == LLMProvider.GEMINI:
            return GeminiAdapter(config)
        elif provider == LLMProvider.MOCK:
            return MockAdapter(config)
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

class LLMAgnosticService:
    def __init__(self, llm_configs: Dict[LLMProvider, LLMConfig], default_provider: LLMProvider):
        self.llm_configs = llm_configs
        self._current_adapter: BaseLLMAdapter = None
        self.set_provider(default_provider)

    def set_provider(self, provider: LLMProvider):
        if provider not in self.llm_configs:
            raise ValueError(f"Configuration for provider {provider} not found.")
        self.current_llm_provider = provider
        self._current_adapter = LLMFactory.get_adapter(provider, self.llm_configs[provider])

    def chat(self, prompt: str) -> str:
        return self._current_adapter.generate_text(prompt)

    def summarize(self, text: str) -> str:
        return self._current_adapter.summarize_text(text)

    def sentiment_analysis(self, text: str) -> str:
        return self._current_adapter.analyze_sentiment(text)

# --- 2. config.py (Configuration Management) ---

load_dotenv()

DEFAULT_LLM_PROVIDER = LLMProvider.MOCK # Default to MOCK for initial setup safety

def get_llm_configs() -> Dict[LLMProvider, LLMConfig]:
    configs = {}
    if os.getenv("OPENAI_API_KEY") and os.getenv("GPT_MODEL_NAME"):
        configs[LLMProvider.GPT] = LLMConfig(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name=os.getenv("GPT_MODEL_NAME", "gpt-3.5-turbo")
        )
    if os.getenv("GEMINI_API_KEY") and os.getenv("GEMINI_MODEL_NAME"):
        configs[LLMProvider.GEMINI] = LLMConfig(
            api_key=os.getenv("GEMINI_API_KEY"),
            model_name=os.getenv("GEMINI_MODEL_NAME", "gemini-pro")
        )
    # Always include a mock config for robustness
    configs[LLMProvider.MOCK] = LLMConfig(
        api_key="mock_key",
        model_name="mock-model"
    )
    return configs

# --- 3. models.py (Pydantic Data Models) ---

class ChatRequest(BaseModel):
    prompt: str

class ChatResponse(BaseModel):
    response: str
    provider: LLMProvider

class SummarizeRequest(BaseModel):
    text: str

class SummarizeResponse(BaseModel):
    summary: str
    provider: LLMProvider

class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    sentiment: str
    provider: LLMProvider

class SetProviderRequest(BaseModel):
    provider: LLMProvider

class SetProviderResponse(BaseModel):
    message: str
    current_provider: LLMProvider

# --- 4. main.py (FastAPI Application) ---

app = FastAPI(
    title="LLM-Agnostic Customer Support Assistant",
    description="API for a customer support system that can dynamically switch between different LLM providers."
)

# Load configurations and initialize service
llm_configurations = get_llm_configs()
llm_service = LLMAgnosticService(llm_configurations, DEFAULT_LLM_PROVIDER)

@app.on_event("startup")
def startup_event():
    if not llm_configurations:
        print("WARNING: No LLM configurations loaded from environment variables. Only MockAdapter is available.")
    print(f"Initial LLM Provider: {llm_service.current_llm_provider.value}")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        response_text = llm_service.chat(request.prompt)
        return ChatResponse(response=response_text, provider=llm_service.current_llm_provider)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize", response_model=SummarizeResponse)
async def summarize_endpoint(request: SummarizeRequest):
    try:
        summary_text = llm_service.summarize(request.text)
        return SummarizeResponse(summary=summary_text, provider=llm_service.current_llm_provider)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sentiment", response_model=SentimentResponse)
async def sentiment_endpoint(request: SentimentRequest):
    try:
        sentiment_result = llm_service.analyze_sentiment(request.text)
        return SentimentResponse(sentiment=sentiment_result, provider=llm_service.current_llm_provider)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/set-provider", response_model=SetProviderResponse)
async def set_provider_endpoint(request: SetProviderRequest):
    try:
        llm_service.set_provider(request.provider)
        return SetProviderResponse(
            message=f"LLM provider switched to {request.provider.value}",
            current_provider=llm_service.current_llm_provider
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

