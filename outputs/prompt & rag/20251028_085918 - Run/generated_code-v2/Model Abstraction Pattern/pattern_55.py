import os
from abc import ABC, abstractmethod

# 1. Configuration Management (simulated)
class Config:
    def __init__(self):
        self.GPT_API_KEY = os.getenv("GPT_API_KEY", "mock_gpt_key")
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "mock_gemini_key")
        self.LLAMA_API_KEY = os.getenv("LLAMA_API_KEY", "mock_llama_key")

# 2. LLM Abstraction Layer
class LLMProvider(ABC):
    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        pass

# 3. Concrete LLM Providers
class GPTProvider(LLMProvider):
    def __init__(self, api_key: str):
        self._api_key = api_key

    def generate_response(self, prompt: str) -> str:
        # In a real application, this would use the OpenAI client library
        # For demonstration, we'll return a mocked response.
        return f"[GPT] Processing: '{prompt}' - This is a GPT-generated response."

class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str):
        self._api_key = api_key

    def generate_response(self, prompt: str) -> str:
        # In a real application, this would use the Google Generative AI client library
        # For demonstration, we'll return a mocked response.
        return f"[Gemini] Processing: '{prompt}' - This is a Gemini-generated response."

class LlamaProvider(LLMProvider):
    def __init__(self, api_key: str):
        self._api_key = api_key

    def generate_response(self, prompt: str) -> str:
        # In a real application, this would use the transformers library for a Llama model
        # For demonstration, we'll return a mocked response.
        return f"[Llama] Processing: '{prompt}' - This is a Llama-generated response."

# 4. LLM Router
class LLMRouter:
    def __init__(self, providers: dict[str, LLMProvider]):
        self._providers = providers
        self._current_provider_name = list(providers.keys())[0] if providers else None

    def set_provider(self, provider_name: str):
        if provider_name in self._providers:
            self._current_provider_name = provider_name
            print(f"Active LLM provider switched to: {provider_name}")
        else:
            print(f"Error: Provider '{provider_name}' not found.")

    def route_and_generate(self, prompt: str) -> str:
        # Simple routing logic: use the currently set provider.
        # More sophisticated logic could be added here (e.g., keyword-based, complexity-based).
        if self._current_provider_name:
            provider = self._providers[self._current_provider_name]
            return provider.generate_response(prompt)
        return "Error: No LLM provider is active."

# 5. Chatbot Core Service
class CustomerChatbot:
    def __init__(self, router: LLMRouter):
        self._router = router

    def get_chatbot_response(self, user_query: str) -> str:
        return self._router.route_and_generate(user_query)

    def switch_model(self, model_name: str):
        self._router.set_provider(model_name)

# 6. Main Application Entry Point
def main():
    config = Config()

    gpt_provider = GPTProvider(config.GPT_API_KEY)
    gemini_provider = GeminiProvider(config.GEMINI_API_KEY)
    llama_provider = LlamaProvider(config.LLAMA_API_KEY)

    providers = {
        "GPT": gpt_provider,
        "Gemini": gemini_provider,
        "Llama": llama_provider,
    }

    router = LLMRouter(providers)
    chatbot = CustomerChatbot(router)

    print("Intelligent Customer Support Chatbot Started. Type 'exit' to quit.")
    print("You can switch models by typing: /switch <GPT|Gemini|Llama>")
    print(f"Current active model: {router._current_provider_name}")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        elif user_input.lower().startswith('/switch '):
            parts = user_input.split(' ', 1)
            if len(parts) == 2:
                model_name = parts[1].strip()
                chatbot.switch_model(model_name)
                print(f"Current active model: {router._current_provider_name}")
            else:
                print("Invalid switch command. Usage: /switch <GPT|Gemini|Llama>")
        else:
            response = chatbot.get_chatbot_response(user_input)
            print(f"Chatbot: {response}")

if __name__ == "__main__":
    main()