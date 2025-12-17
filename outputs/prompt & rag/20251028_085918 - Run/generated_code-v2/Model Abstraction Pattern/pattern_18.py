import os
from abc import ABC, abstractmethod

# 1. Abstract LLM Provider Interface
class AbstractLLMProvider(ABC):
    """Abstract base class for all LLM providers."""

    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        """Generates a response using the specific LLM provider."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Returns the name of the LLM provider."""
        pass

# 2. Concrete LLM Provider Implementations
class GPTProvider(AbstractLLMProvider):
    """Concrete implementation for a GPT-based LLM provider."""
    def __init__(self, api_key: str):
        self.api_key = api_key
        # In a real scenario, initialize OpenAI client here
        # self.client = OpenAI(api_key=api_key)

    def generate_response(self, prompt: str) -> str:
        # Simulate API call to GPT
        # In a real app: response = self.client.chat.completions.create(...)
        print(f"[GPTProvider] Sending prompt to GPT: '{prompt}'")
        return f"GPT's answer to: '{prompt}' (using key: {self.api_key[:5]}...)"

    def get_name(self) -> str:
        return "GPT"

class GeminiProvider(AbstractLLMProvider):
    """Concrete implementation for a Gemini-based LLM provider."""
    def __init__(self, api_key: str):
        self.api_key = api_key
        # In a real scenario, initialize Gemini client here
        # import google.generativeai as genai
        # genai.configure(api_key=api_key)
        # self.model = genai.GenerativeModel('gemini-pro')

    def generate_response(self, prompt: str) -> str:
        # Simulate API call to Gemini
        # In a real app: response = self.model.generate_content(prompt)
        print(f"[GeminiProvider] Sending prompt to Gemini: '{prompt}'")
        return f"Gemini's answer to: '{prompt}' (using key: {self.api_key[:5]}...)"

    def get_name(self) -> str:
        return "Gemini"

class LlamaProvider(AbstractLLMProvider):
    """Concrete implementation for a Llama-based LLM provider."""
    def __init__(self, model_path: str = "./llama_model"): # model_path could be an API endpoint too
        self.model_path = model_path
        # In a real scenario, load Llama model or initialize client for local/cloud inference
        # from transformers import pipeline
        # self.generator = pipeline('text-generation', model=model_path)

    def generate_response(self, prompt: str) -> str:
        # Simulate API call to Llama
        # In a real app: response = self.generator(prompt, ...)
        print(f"[LlamaProvider] Sending prompt to Llama (from {self.model_path}): '{prompt}'")
        return f"Llama's answer to: '{prompt}' (using model from {self.model_path})"

    def get_name(self) -> str:
        return "Llama"


# 3. LLM Provider Factory/Manager
class LLMManager:
    """Manages and provides LLM instances based on configuration."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMManager, cls).__new__(cls)
            cls._instance._providers = {}
            cls._instance._current_provider_name = None
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        # Load API keys and default provider from environment variables
        # In a real app, use python-dotenv or a dedicated config system
        self._openai_api_key = os.getenv("OPENAI_API_KEY", "sk-mockopenaiapikey")
        self._gemini_api_key = os.getenv("GEMINI_API_KEY", "mockgeminiapikey")
        self._llama_model_path = os.getenv("LLAMA_MODEL_PATH", "./mock_llama_model")
        self._default_provider = os.getenv("DEFAULT_LLM_PROVIDER", "gemini").lower()

        self.register_provider("gpt", GPTProvider(api_key=self._openai_api_key))
        self.register_provider("gemini", GeminiProvider(api_key=self._gemini_api_key))
        self.register_provider("llama", LlamaProvider(model_path=self._llama_model_path))

        if self._default_provider in self._providers:
            self._current_provider_name = self._default_provider
            print(f"LLMManager initialized with default provider: {self._current_provider_name.upper()}")
        else:
            raise ValueError(f"Default provider '{self._default_provider}' not recognized.")

    def register_provider(self, name: str, provider: AbstractLLMProvider):
        """Registers an LLM provider with a given name."""
        self._providers[name.lower()] = provider

    def set_current_provider(self, provider_name: str):
        """Sets the active LLM provider by name."""
        name = provider_name.lower()
        if name in self._providers:
            self._current_provider_name = name
            print(f"Switched active LLM provider to: {self._current_provider_name.upper()}")
        else:
            raise ValueError(f"Provider '{provider_name}' not registered. Available: {list(self._providers.keys())}")

    def get_current_provider(self) -> AbstractLLMProvider:
        """Returns the currently active LLM provider instance."""
        if self._current_provider_name is None:
            raise RuntimeError("No LLM provider is set. Please set a provider first.")
        return self._providers[self._current_provider_name]

    def get_provider(self, provider_name: str) -> AbstractLLMProvider:
        """Returns a specific LLM provider instance by name."""
        name = provider_name.lower()
        if name in self._providers:
            return self._providers[name]
        else:
            raise ValueError(f"Provider '{provider_name}' not registered. Available: {list(self._providers.keys())}")


# 4. Customer Support Assistant Application Logic
class CustomerSupportAssistant:
    """Handles customer queries using the configured LLM provider."""

    def __init__(self):
        self.llm_manager = LLMManager()

    def answer_query(self, query: str) -> str:
        """Answers a customer query using the current LLM provider."""
        provider = self.llm_manager.get_current_provider()
        print(f"\n[CustomerSupportAssistant] Answering query using {provider.get_name()} provider...")
        response = provider.generate_response(query)
        return response

    def switch_llm_provider(self, provider_name: str):
        """Switches the underlying LLM provider."""
        self.llm_manager.set_current_provider(provider_name)


# 5. Demonstration and Configuration Example
if __name__ == "__main__":
    # --- Setup Environment Variables (simulate with os.environ for demonstration) ---
    # In a real application, you would use a .env file and `from dotenv import load_dotenv; load_dotenv()`
    os.environ["OPENAI_API_KEY"] = "sk-realgptapikey123"
    os.environ["GEMINI_API_KEY"] = "realgeminiapikey456"
    os.environ["LLAMA_MODEL_PATH"] = "./fine_tuned_llama_7b"
    # os.environ["DEFAULT_LLM_PROVIDER"] = "llama" # Uncomment to test changing default

    print("--- Initializing Customer Support Assistant ---")
    assistant = CustomerSupportAssistant()

    # --- Test with Default Provider (Gemini in this case, unless overridden by env) ---
    query1 = "What is your return policy?"
    response1 = assistant.answer_query(query1)
    print(f"Customer: '{query1}'")
    print(f"Assistant: {response1}")

    # --- Switch to GPT Provider ---
    print("\n--- Switching to GPT Provider ---")
    assistant.switch_llm_provider("gpt")
    query2 = "How do I track my order?"
    response2 = assistant.answer_query(query2)
    print(f"Customer: '{query2}'")
    print(f"Assistant: {response2}")

    # --- Switch to Llama Provider ---
    print("\n--- Switching to Llama Provider ---")
    assistant.switch_llm_provider("llama")
    query3 = "Can I change my shipping address after purchase?"
    response3 = assistant.answer_query(query3)
    print(f"Customer: '{query3}'")
    print(f"Assistant: {response3}")

    # --- Test with the default provider again after switching ---
    print("\n--- Switching back to Gemini Provider (default in absence of env var) ---")
    assistant.switch_llm_provider("gemini")
    query4 = "What payment methods do you accept?"
    response4 = assistant.answer_query(query4)
    print(f"Customer: '{query4}'")
    print(f"Assistant: {response4}")

    # --- Demonstrate directly getting a specific provider (e.g., for benchmarking) ---
    print("\n--- Directly getting a specific provider (e.g., for benchmarking) ---")
    llama_benchmark_provider = assistant.llm_manager.get_provider("llama")
    benchmark_query = "Summarize recent product launches."
    benchmark_response = llama_benchmark_provider.generate_response(benchmark_query)
    print(f"[Benchmarking with {llama_benchmark_provider.get_name()}] Query: '{benchmark_query}'")
    print(f"[Benchmarking Response]: {benchmark_response}")

    print("\n--- End of Demonstration ---")