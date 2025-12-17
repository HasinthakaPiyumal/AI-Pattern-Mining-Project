from abc import ABC, abstractmethod
import os

class AbstractLLMClient(ABC):
    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        pass

    @abstractmethod
    def get_model_info(self) -> dict:
        pass

class GeminiClient(AbstractLLMClient):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model_name = "Gemini Pro"

    def generate_response(self, prompt: str) -> str:
        # Simulate API call to Gemini
        if "billing" in prompt.lower():
            return f"Gemini response for '{prompt}': Please check our billing FAQ at example.com/billing. (using API key: {self.api_key[-4:]})"
        return f"Gemini response for '{prompt}': How can I help you further? (using API key: {self.api_key[-4:]})"

    def get_model_info(self) -> dict:
        return {"name": self.model_name, "provider": "Google", "cost_effective": True}

class GPTClient(AbstractLLMClient):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model_name = "GPT-4"

    def generate_response(self, prompt: str) -> str:
        # Simulate API call to GPT
        if "technical issue" in prompt.lower() or "troubleshoot" in prompt.lower():
            return f"GPT response for '{prompt}': Let's troubleshoot this technical issue step by step. (using API key: {self.api_key[-4:]})"
        return f"GPT response for '{prompt}': I'm a powerful AI assistant ready to help. (using API key: {self.api_key[-4:]})"

    def get_model_info(self) -> dict:
        return {"name": self.model_name, "provider": "OpenAI", "powerful": True}

class LlamaClient(AbstractLLMClient):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model_name = "Llama-2-70b"

    def generate_response(self, prompt: str) -> str:
        # Simulate API call to Llama (e.g., local or custom API)
        if "sensitive data" in prompt.lower() or "private information" in prompt.lower():
            return f"Llama response for '{prompt}': For sensitive inquiries, please contact our dedicated support line. (using API key: {self.api_key[-4:]})"
        return f"Llama response for '{prompt}': I am an open-source model, how can I assist you? (using API key: {self.api_key[-4:]})"

    def get_model_info(self) -> dict:
        return {"name": self.model_name, "provider": "Meta/Open-source", "privacy_focused": True}

class LLMFactory:
    def __init__(self):
        self.api_keys = {
            "gemini": os.getenv("GEMINI_API_KEY", "mock_gemini_key"),
            "gpt": os.getenv("GPT_API_KEY", "mock_gpt_key"),
            "llama": os.getenv("LLAMA_API_KEY", "mock_llama_key"),
        }

    def get_client(self, model_type: str) -> AbstractLLMClient:
        if model_type.lower() == "gemini":
            return GeminiClient(self.api_keys["gemini"])
        elif model_type.lower() == "gpt":
            return GPTClient(self.api_keys["gpt"])
        elif model_type.lower() == "llama":
            return LlamaClient(self.api_keys["llama"])
        else:
            raise ValueError(f"Unknown LLM type: {model_type}")

class SmartChatbot:
    def __init__(self, llm_factory: LLMFactory):
        self.llm_factory = llm_factory

    def _analyze_query(self, query: str) -> str:
        query_lower = query.lower()
        if "billing" in query_lower or "invoice" in query_lower or "payment" in query_lower:
            return "FAQ"
        elif "technical issue" in query_lower or "error" in query_lower or "troubleshoot" in query_lower:
            return "Technical Support"
        elif "sensitive data" in query_lower or "private info" in query_lower or "account security" in query_lower:
            return "Sensitive Data Request"
        else:
            return "General"

    def _route_llm(self, query_category: str) -> str:
        if query_category == "FAQ":
            return "gemini"  # Cost-effective for simple queries
        elif query_category == "Technical Support":
            return "gpt"     # Powerful for complex issues
        elif query_category == "Sensitive Data Request":
            return "llama"   # Potentially for self-hosted/privacy-focused (mocked here)
        else:
            return "gemini"  # Default to Gemini for general queries

    def get_chat_response(self, customer_query: str) -> str:
        query_category = self._analyze_query(customer_query)
        selected_llm_type = self._route_llm(query_category)

        try:
            llm_client = self.llm_factory.get_client(selected_llm_type)
            response = llm_client.generate_response(customer_query)
            return f"[Using {llm_client.get_model_info()['name']}] {response}"
        except ValueError as e:
            return f"Error: {e}. Could not process the request."

if __name__ == "__main__":
    # Example Usage
    llm_factory = LLMFactory()
    chatbot = SmartChatbot(llm_factory)

    print("--- Chatbot Interactions ---")

    query1 = "I have a question about my latest invoice."
    print(f"Customer: {query1}")
    print(f"Chatbot: {chatbot.get_chat_response(query1)}\n")

    query2 = "I'm encountering a technical issue with my software. Can you help me troubleshoot?"
    print(f"Customer: {query2}")
    print(f"Chatbot: {chatbot.get_chat_response(query2)}\n")

    query3 = "What about my account security and private information?"
    print(f"Customer: {query3}")
    print(f"Chatbot: {chatbot.get_chat_response(query3)}\n")

    query4 = "Tell me about your product features."
    print(f"Customer: {query4}")
    print(f"Chatbot: {chatbot.get_chat_response(query4)}\n")

    query5 = "This is a general query."
    print(f"Customer: {query5}")
    print(f"Chatbot: {chatbot.get_chat_response(query5)}\n")

    print("\n--- Demonstrating direct client usage (for testing/benchmarking) ---")
    gemini_client = llm_factory.get_client("gemini")
    gpt_client = llm_factory.get_client("gpt")

    print(f"Gemini directly: {gemini_client.generate_response('Hello Gemini!')}")
    print(f"GPT directly: {gpt_client.generate_response('Hello GPT!')}")
    print(f"Gemini Info: {gemini_client.get_model_info()}")
    print(f"GPT Info: {gpt_client.get_model_info()}")
