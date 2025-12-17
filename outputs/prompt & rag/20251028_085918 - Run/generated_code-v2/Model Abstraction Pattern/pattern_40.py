from abc import ABC, abstractmethod
from typing import Dict, Type

# 1. LLM Abstraction Layer
class AbstractLLM(ABC):
    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        pass

class GPTModel(AbstractLLM):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def generate_response(self, prompt: str) -> str:
        # Simulate OpenAI GPT API call
        return f"GPT Response to '{prompt}' using key: {self.api_key[:5]}..."

class GeminiModel(AbstractLLM):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def generate_response(self, prompt: str) -> str:
        # Simulate Google Gemini API call
        return f"Gemini Response to '{prompt}' using key: {self.api_key[:5]}..."

class LlamaModel(AbstractLLM):
    def __init__(self, model_path: str):
        self.model_path = model_path

    def generate_response(self, prompt: str) -> str:
        # Simulate Llama model inference (e.g., via Hugging Face or local server)
        return f"Llama Response to '{prompt}' from model: {self.model_path}"

# 2. LLM Provider Manager
class LLMProviderManager:
    def __init__(self, config: Dict):
        self.config = config
        self._providers: Dict[str, Type[AbstractLLM]] = {
            "gpt": GPTModel,
            "gemini": GeminiModel,
            "llama": LlamaModel,
        }

    def get_llm(self, provider_name: str) -> AbstractLLM:
        provider_name = provider_name.lower()
        if provider_name not in self._providers:
            raise ValueError(f"Unsupported LLM provider: {provider_name}")

        llm_class = self._providers[provider_name]
        provider_config = self.config.get(provider_name, {})

        if provider_name == "gpt":
            api_key = provider_config.get("api_key", "DUMMY_GPT_KEY")
            return llm_class(api_key=api_key)
        elif provider_name == "gemini":
            api_key = provider_config.get("api_key", "DUMMY_GEMINI_KEY")
            return llm_class(api_key=api_key)
        elif provider_name == "llama":
            model_path = provider_config.get("model_path", "local/llama2-7b-chat")
            return llm_class(model_path=model_path)
        else:
            # Fallback for future providers, though current check handles it
            raise NotImplementedError(f"Configuration for {provider_name} not implemented")

# 3. Chatbot Core Logic
class OmniBot:
    def __init__(self, provider_manager: LLMProviderManager, default_provider: str = "gpt"):
        self.provider_manager = provider_manager
        self._current_llm: AbstractLLM = self.provider_manager.get_llm(default_provider)
        self.current_provider_name = default_provider

    def set_provider(self, provider_name: str):
        try:
            self._current_llm = self.provider_manager.get_llm(provider_name)
            self.current_provider_name = provider_name
            print(f"OmniBot switched to {provider_name} provider.")
        except ValueError as e:
            print(f"Error switching provider: {e}")

    def chat(self, user_query: str) -> str:
        print(f"\nUser ({self.current_provider_name}): {user_query}")
        response = self._current_llm.generate_response(user_query)
        print(f"OmniBot ({self.current_provider_name}): {response}")
        return response

# 4. Configuration
APP_CONFIG = {
    "gpt": {
        "api_key": "sk-YOUR_OPENAI_KEY_HERE"
    },
    "gemini": {
        "api_key": "YOUR_GEMINI_KEY_HERE"
    },
    "llama": {
        "model_path": "models/llama2-7b-chat-hf"
    }
}

if __name__ == "__main__":
    print("Initializing Omni-Bot...")
    provider_manager = LLMProviderManager(APP_CONFIG)
    
    # Initialize with a default provider (e.g., GPT)
    omnibot = OmniBot(provider_manager, default_provider="gpt")

    # Simulate conversation with GPT
    omnibot.chat("Hello, how can I get help with my order?")
    omnibot.chat("What is your return policy?")

    # Switch to Gemini provider dynamically
    omnibot.set_provider("gemini")
    omnibot.chat("Can you tell me about the latest product updates?")
    omnibot.chat("What are your operating hours?")

    # Switch to Llama provider dynamically
    omnibot.set_provider("llama")
    omnibot.chat("I have a technical question about product X.")
    omnibot.chat("How do I reset my password?")

    # Try to switch to an unsupported provider
    omnibot.set_provider("unsupported_llm")

    omnibot.chat("Give me a summary of my last interaction.")

    print("\nOmni-Bot demonstration complete.")