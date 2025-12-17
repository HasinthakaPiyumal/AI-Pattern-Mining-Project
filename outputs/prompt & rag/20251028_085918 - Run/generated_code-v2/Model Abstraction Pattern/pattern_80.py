import os
import abc
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import google.generativeai as genai

load_dotenv()

class AbstractLLMProvider(abc.ABC):
    @abc.abstractmethod
    def generate_response(self, prompt: str) -> str:
        pass

class OpenAIProvider(AbstractLLMProvider):
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        openai.api_key = self.api_key

    def generate_response(self, prompt: str) -> str:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message["content"]
        except Exception as e:
            return f"Error with OpenAI: {e}"

class GeminiProvider(AbstractLLMProvider):
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-pro")

    def generate_response(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error with Gemini: {e}"

class MockLlamaProvider(AbstractLLMProvider):
    def generate_response(self, prompt: str) -> str:
        return f"Mock Llama response for: '{prompt}'. This is a simulated response."

class LLMManager:
    _instance = None
    _llm_provider: AbstractLLMProvider = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMManager, cls).__new__(cls)
            cls._instance._initialize_provider()
        return cls._instance

    def _initialize_provider(self):
        active_provider = os.getenv("ACTIVE_LLM_PROVIDER", "mock_llama").lower()
        if active_provider == "openai":
            self._llm_provider = OpenAIProvider()
        elif active_provider == "gemini":
            self._llm_provider = GeminiProvider()
        elif active_provider == "mock_llama":
            self._llm_provider = MockLlamaProvider()
        else:
            raise ValueError(f"Unknown LLM provider: {active_provider}. Valid options: openai, gemini, mock_llama.")

    def get_provider(self) -> AbstractLLMProvider:
        return self._llm_provider

class ChatbotCore:
    def __init__(self):
        self.llm_manager = LLMManager()

    def get_chatbot_response(self, user_query: str) -> str:
        llm_provider = self.llm_manager.get_provider()
        raw_response = llm_provider.generate_response(user_query)
        return f"Chatbot: {raw_response}"

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

chatbot_core = ChatbotCore()

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        response_text = chatbot_core.get_chatbot_response(request.message)
        return ChatResponse(response=response_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
