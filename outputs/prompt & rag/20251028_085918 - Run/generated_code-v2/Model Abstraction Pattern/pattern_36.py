import os
from abc import ABC, abstractmethod
from typing import Optional

# --- llm_abstract_interface.py ---
class LLMAbstractInterface(ABC):
    """
    Abstract base class for all LLM providers.
    Defines the unified interface for interacting with any LLM.
    """
    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        """
        Generates a response from the LLM based on the given prompt.
        """
        pass

# --- gpt_adapter.py ---
class GPTAdapter(LLMAbstractInterface):
    """
    Adapter for interacting with a GPT-like LLM provider.
    """
    def __init__(self, api_key: str):
        self.api_key = api_key
        print(f"GPTAdapter initialized with API key: {self.api_key[:5]}...")

    def generate_response(self, prompt: str) -> str:
        """
        Simulates generating a response using a GPT model.
        """
        print(f"[GPT] Processing prompt: {prompt}")
        return f"GPT says: \"I understand your request regarding \'{prompt}\'. How can I further assist you?\""

# --- gemini_adapter.py ---
class GeminiAdapter(LLMAbstractInterface):
    """
    Adapter for interacting with a Gemini-like LLM provider.
    """
    def __init__(self, api_key: str, model_name: str = "gemini-pro"):
        self.api_key = api_key
        self.model_name = model_name
        print(f"GeminiAdapter initialized with API key: {self.api_key[:5]}... and model: {self.model_name}")

    def generate_response(self, prompt: str) -> str:
        """
        Simulates generating a response using a Gemini model.
        """
        print(f"[Gemini] Processing prompt: {prompt}")
        return f"Gemini responds: \"Your query about \'{prompt}\' is clear. Let me provide more details.\""

# --- llama_adapter.py ---
class LlamaAdapter(LLMAbstractInterface):
    """
    Adapter for interacting with a Llama-like LLM provider (e.g., through an API or local inference).
    """
    def __init__(self, model_path: str = "./models/llama-7b"):
        self.model_path = model_path
        print(f"LlamaAdapter initialized, simulating model loaded from: {self.model_path}")

    def generate_response(self, prompt: str) -> str:
        """
        Simulates generating a response using a Llama model.
        """
        print(f"[Llama] Processing prompt: {prompt}")
        return f"Llama AI states: \"Regarding \'{prompt}\', I have processed the information and am ready with an answer.\""

# --- llm_factory.py ---
class LLMFactory:
    """
    Factory class to create instances of different LLM provider adapters.
    This centralizes the creation logic and allows easy switching of providers.
    """
    @staticmethod
    def get_llm_provider(
        provider_name: str,
        api_key: Optional[str] = None,
        **kwargs
    ) -> LLMAbstractInterface:
        """
        Returns an instance of the specified LLM provider adapter.
        """
        if provider_name.lower() == "gpt":
            if not api_key: raise ValueError("API key is required for GPT provider.")
            return GPTAdapter(api_key=api_key)
        elif provider_name.lower() == "gemini":
            if not api_key: raise ValueError("API key is required for Gemini provider.")
            return GeminiAdapter(api_key=api_key, **kwargs)
        elif provider_name.lower() == "llama":
            return LlamaAdapter(**kwargs)
        else:
            raise ValueError(f"Unknown LLM provider: {provider_name}")

# --- chatbot_platform.py (Main Application Logic) ---
class CustomerSupportChatbot:
    """
    A dynamic customer support chatbot platform that uses the Model Abstraction Layer
    to switch between different LLM providers based on query characteristics.
    """
    def __init__(self):
        self.current_llm: Optional[LLMAbstractInterface] = None
        self.llm_provider_config = {
            "gpt": {"api_key": os.getenv("GPT_API_KEY", "sk-YOUR_GPT_KEY")},
            "gemini": {"api_key": os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_KEY"), "model_name": "gemini-pro"},
            "llama": {"model_path": "./models/llama-7b-tuned"}
        }
        print("Chatbot initialized. Ready to handle queries.")

    def _select_llm_provider(self, query: str) -> str:
        """
        A simple routing mechanism to select an LLM provider based on query characteristics.
        """
        query_length = len(query.split())
        if "urgent" in query.lower() or query_length > 20:
            print("Routing to GPT (simulated for complex queries)...")
            return "gpt"
        elif "billing" in query.lower() or query_length > 10:
            print("Routing to Gemini (simulated for billing/medium queries)...")
            return "gemini"
        else:
            print("Routing to Llama (simulated for simple queries)...")
            return "llama"

    def handle_query(self, query: str) -> str:
        """
        Handles a customer query by dynamically selecting an LLM provider and generating a response.
        """
        selected_provider_name = self._select_llm_provider(query)

        config = self.llm_provider_config.get(selected_provider_name, {})
        try:
            self.current_llm = LLMFactory.get_llm_provider(selected_provider_name, **config)
        except ValueError as e:
            return f"Error: Cannot load LLM provider - {e}"

        if self.current_llm:
            response = self.current_llm.generate_response(query)
            return response
        else:
            return "Error: No LLM provider could be loaded."

if __name__ == "__main__":
    os.environ.setdefault("GPT_API_KEY", "sk-dummy-gpt-key")
    os.environ.setdefault("GEMINI_API_KEY", "dummy-gemini-key")

    chatbot = CustomerSupportChatbot()

    print("\n--- Testing simple query ---")
    simple_query = "What are your operating hours?"
    print(f"User: {simple_query}")
    response = chatbot.handle_query(simple_query)
    print(f"Chatbot: {response}")

    print("\n--- Testing medium complexity query (billing) ---")
    billing_query = "I have a question about my last month's billing statement and a specific charge of $49.99."
    print(f"User: {billing_query}")
    response = chatbot.handle_query(billing_query)
    print(f"Chatbot: {response}")

    print("\n--- Testing complex/urgent query ---")
    complex_query = "URGENT: My service is completely down and I need immediate assistance! This is a critical issue impacting my business operations and requires a fast resolution. Please connect me to a human representative or provide advanced troubleshooting steps."
    print(f"User: {complex_query}")
    response = chatbot.handle_query(complex_query)
    print(f"Chatbot: {response}")

    print("\n--- Testing another simple query (should re-route) ---")
    another_simple_query = "Hello, how are you?"
    print(f"User: {another_simple_query}")
    response = chatbot.handle_query(another_simple_query)
    print(f"Chatbot: {response}")