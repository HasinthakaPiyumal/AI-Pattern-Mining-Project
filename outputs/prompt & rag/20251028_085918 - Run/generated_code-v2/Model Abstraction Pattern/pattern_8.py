from llm_providers import AbstractLLMProvider, OpenAILLMProvider, GoogleGeminiLLMProvider, HuggingFaceHubLLMProvider
from config import OPENAI_API_KEY, GOOGLE_API_KEY, HUGGINGFACEHUB_API_TOKEN

class LLMManager:
    """Manages and provides access to different LLM providers."""

    def __init__(self):
        self._providers = {
            "openai": OpenAILLMProvider(api_key=OPENAI_API_KEY),
            "gemini": GoogleGeminiLLMProvider(api_key=GOOGLE_API_KEY),
            "huggingface": HuggingFaceLLMProvider(api_token=HUGGINGFACEHUB_API_TOKEN)
        }
        self._current_provider_name = "openai"  # Default provider

    def set_provider(self, provider_name: str):
        """Sets the active LLM provider."""
        if provider_name not in self._providers:
            raise ValueError(f"Unknown LLM provider: {provider_name}. Available providers: {list(self._providers.keys())}")
        self._current_provider_name = provider_name
        print(f"LLM provider set to: {self._current_provider_name}")

    def get_current_provider(self) -> AbstractLLMProvider:
        """Returns the current active LLM provider instance."""
        return self._providers[self._current_provider_name]

    def generate_response(self, prompt: str) -> str:
        """Generates a response using the current active LLM provider."""
        return self.get_current_provider().generate_response(prompt)