import os
from abc import ABC, abstractmethod

# Try to import specific LLM integrations, handle gracefully if not installed
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    OPENAI_AVAILABLE = True
except ImportError:
    ChatOpenAI = None
    HumanMessage = None
    OPENAI_AVAILABLE = False

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage as GeminiHumanMessage # Use alias to avoid conflict if both are present
    GEMINI_AVAILABLE = True
except ImportError:
    ChatGoogleGenerativeAI = None
    GeminiHumanMessage = None
    GEMINI_AVAILABLE = False

class AbstractLLMProvider(ABC):
    """Abstract base class for all LLM providers."""

    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        """Generates a response from the LLM based on the given prompt."""
        pass

class OpenAIProvider(AbstractLLMProvider):
    """Concrete implementation for OpenAI LLM provider."""
    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.7):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAIProvider requires langchain-openai to be installed.")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature, openai_api_key=api_key)

    def generate_response(self, prompt: str) -> str:
        messages = [HumanMessage(content=prompt)]
        response = self.llm.invoke(messages)
        return response.content

class GeminiProvider(AbstractLLMProvider):
    """Concrete implementation for Google Gemini LLM provider."""
    def __init__(self, model_name: str = "gemini-pro", temperature: float = 0.7):
        if not GEMINI_AVAILABLE:
            raise ImportError("GeminiProvider requires langchain-google-genai to be installed.")
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature, google_api_key=api_key)

    def generate_response(self, prompt: str) -> str:
        messages = [GeminiHumanMessage(content=prompt)]
        response = self.llm.invoke(messages)
        return response.content

class DummyLLMProvider(AbstractLLMProvider):
    """A dummy LLM provider for testing purposes."""
    def __init__(self, response_prefix: str = "Dummy Response from {model_name}: "):
        self.response_prefix = response_prefix

    def generate_response(self, prompt: str) -> str:
        return f"{self.response_prefix}{prompt}"
