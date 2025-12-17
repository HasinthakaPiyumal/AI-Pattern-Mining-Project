from abc import ABC, abstractmethod

class LLMProvider(ABC):
    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        pass

class GPTProvider(LLMProvider):
    def generate_response(self, prompt: str) -> str:
        return f"GPT's response to: {prompt}"

class GeminiProvider(LLMProvider):
    def generate_response(self, prompt: str) -> str:
        return f"Gemini's response to: {prompt}"

class LlamaProvider(LLMProvider):
    def generate_response(self, prompt: str) -> str:
        return f"Llama's response to: {prompt}"

class LLMService:
    def __init__(self):
        self._providers = {
            "GPT": GPTProvider(),
            "Gemini": GeminiProvider(),
            "Llama": LlamaProvider(),
        }

    def get_llm_provider(self, provider_name: str) -> LLMProvider:
        if provider_name not in self._providers:
            raise ValueError(f"Unknown LLM provider: {provider_name}")
        return self._providers[provider_name]

class CustomerSupportAssistant:
    def __init__(self, default_provider: str = "GPT"):
        self._llm_service = LLMService()
        self._current_provider_name = default_provider
        self._current_provider = self._llm_service.get_llm_provider(default_provider)

    def set_provider(self, provider_name: str):
        self._current_provider_name = provider_name
        self._current_provider = self._llm_service.get_llm_provider(provider_name)
        print(f"Switched to {provider_name} provider.")

    def handle_query(self, query: str) -> str:
        print(f"\nHandling query with {self._current_provider_name} provider...")
        response = self._current_provider.generate_response(query)
        return response

if __name__ == "__main__":
    assistant = CustomerSupportAssistant("GPT")

    query1 = "What is your refund policy?"
    print(assistant.handle_query(query1))

    assistant.set_provider("Gemini")
    query2 = "How do I reset my password?"
    print(assistant.handle_query(query2))

    assistant.set_provider("Llama")
    query3 = "Can you help me with a product recommendation?"
    print(assistant.handle_query(query3))

    assistant.set_provider("GPT")
    query4 = "Tell me about your latest features."
    print(assistant.handle_query(query4))
