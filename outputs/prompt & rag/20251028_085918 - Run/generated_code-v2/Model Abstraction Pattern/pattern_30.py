import abc
import os
import time

# config.py
class Config:
    GPT_API_KEY = os.getenv("GPT_API_KEY", "sk-mock-gpt-key")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "mock-gemini-key")
    LLAMA_API_URL = os.getenv("LLAMA_API_URL", "http://localhost:8000/mock_llama")

# llm_abstract_interface.py
class LLMProvider(abc.ABC):
    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass

    @abc.abstractmethod
    def generate_response(self, prompt: str) -> str:
        pass

# gpt_provider.py (Mocked)
class GPTProvider(LLMProvider):
    def __init__(self):
        self._name = "GPT"
        self.api_key = Config.GPT_API_KEY

    @property
    def name(self) -> str:
        return self._name

    def generate_response(self, prompt: str) -> str:
        print(f"[GPTProvider] Simulating API call for: {prompt[:50]}...")
        time.sleep(1.5)  # Simulate network latency
        # In a real scenario, this would call OpenAI API
        return f"GPT Response to '{prompt[:30]}...': This is a comprehensive answer from GPT."

# gemini_provider.py (Mocked)
class GeminiProvider(LLMProvider):
    def __init__(self):
        self._name = "Gemini"
        self.api_key = Config.GEMINI_API_KEY

    @property
    def name(self) -> str:
        return self._name

    def generate_response(self, prompt: str) -> str:
        print(f"[GeminiProvider] Simulating API call for: {prompt[:50]}...")
        time.sleep(0.8)  # Simulate network latency
        # In a real scenario, this would call Google Gemini API
        return f"Gemini Response to '{prompt[:30]}...': Here is a concise answer from Gemini."

# llama_provider.py (Mocked)
class LlamaProvider(LLMProvider):
    def __init__(self):
        self._name = "Llama"
        self.api_url = Config.LLAMA_API_URL

    @property
    def name(self) -> str:
        return self._name

    def generate_response(self, prompt: str) -> str:
        print(f"[LlamaProvider] Simulating API call for: {prompt[:50]}...")
        time.sleep(2.0)  # Simulate network latency
        # In a real scenario, this would call a local Llama server or another API
        return f"Llama Response to '{prompt[:30]}...': A detailed explanation from Llama."

# llm_strategy_selector.py
class LLMStrategySelector:
    def __init__(self, providers: list[LLMProvider]):
        self._providers = {p.name: p for p in providers}

    def select_provider(self, task_complexity: str = "medium") -> LLMProvider:
        # Simple selection logic for demonstration
        if task_complexity == "high":
            print("Selecting Llama for high complexity tasks.")
            return self._providers["Llama"]
        elif task_complexity == "low":
            print("Selecting Gemini for low complexity tasks.")
            return self._providers["Gemini"]
        else:  # medium
            print("Selecting GPT for medium complexity tasks.")
            return self._providers["GPT"]

    def get_provider(self, provider_name: str) -> LLMProvider:
        if provider_name not in self._providers:
            raise ValueError(f"Provider '{provider_name}' not registered.")
        return self._providers[provider_name]

# chatbot.py
class CustomerSupportChatbot:
    def __init__(self, selector: LLMStrategySelector):
        self.selector = selector

    def ask(self, query: str, complexity: str = "medium") -> str:
        selected_provider = self.selector.select_provider(complexity)
        print(f"Chatbot using {selected_provider.name} for query.")
        response = selected_provider.generate_response(query)
        return response

if __name__ == "__main__":
    # Initialize providers
    gpt = GPTProvider()
    gemini = GeminiProvider()
    llama = LlamaProvider()

    # Initialize strategy selector with all providers
    selector = LLMStrategySelector(providers=[gpt, gemini, llama])

    # Initialize chatbot
    chatbot = CustomerSupportChatbot(selector)

    print("\n--- Testing Low Complexity Query ---")
    response_low = chatbot.ask("What are your operating hours?", complexity="low")
    print(f"Chatbot Says: {response_low}")

    print("\n--- Testing Medium Complexity Query ---")
    response_medium = chatbot.ask("How do I reset my password if I forgot it?", complexity="medium")
    print(f"Chatbot Says: {response_medium}")

    print("\n--- Testing High Complexity Query ---")
    response_high = chatbot.ask("I need help troubleshooting a complex software issue with error code XYZ-123.", complexity="high")
    print(f"Chatbot Says: {response_high}")

    print("\n--- Dynamically Switching to a specific provider (e.g., GPT) ---")
    specific_gpt_provider = selector.get_provider("GPT")
    print(f"Chatbot manually switching to {specific_gpt_provider.name} for query.")
    manual_response_gpt = specific_gpt_provider.generate_response("Can you summarize my last order?")
    print(f"Chatbot Says: {manual_response_gpt}")
