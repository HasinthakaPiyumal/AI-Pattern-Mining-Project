import os
from abc import ABC, abstractmethod
from dotenv import load_dotenv

# Try to import openai, google.generativeai. If not available, handle gracefully.
try:
    import openai
except ImportError:
    openai = None
    print("Warning: 'openai' library not found. OpenAI LLM will not be available.")

try:
    import google.generativeai as genai
except ImportError:
    genai = None
    print("Warning: 'google.generativeai' library not found. Gemini LLM will not be available.")

load_dotenv()

# LLM Abstraction Layer
class LLMProvider(ABC):
    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        pass

class OpenAILLM(LLMProvider):
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        if openai is None:
            raise RuntimeError("OpenAI library not loaded. Cannot initialize OpenAILLM.")
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def generate_response(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful customer support assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error from OpenAI: {e}"

class GeminiLLM(LLMProvider):
    def __init__(self, api_key: str, model: str = "gemini-pro"):
        if genai is None:
            raise RuntimeError("Google Generative AI library not loaded. Cannot initialize GeminiLLM.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)

    def generate_response(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error from Gemini: {e}"

class MockLlamaLLM(LLMProvider):
    def generate_response(self, prompt: str) -> str:
        # Simulate a response from a Llama-like model
        return f"[MockLlama Response for '{prompt[:50]}...']: I understand you're looking for assistance. How can I help further?"

# LLM Manager
class LLMManager:
    def __init__(self):
        self._providers = {}

    def register_llm_provider(self, name: str, provider: LLMProvider):
        self._providers[name] = provider

    def select_llm_provider(self, query_type: str) -> LLMProvider:
        # Simple logic to select an LLM based on query type
        if query_type == "billing" and "gemini" in self._providers:
            return self._providers["gemini"]
        elif query_type == "product_info" and "openai" in self._providers:
            return self._providers["openai"]
        elif "llama" in self._providers:
            return self._providers["llama"]
        elif "openai" in self._providers:
            return self._providers["openai"]
        elif "gemini" in self._providers:
            return self._providers["gemini"]
        else:
            raise ValueError("No suitable LLM provider found for the query type.")

    def get_llm_response(self, prompt: str, query_type: str) -> str:
        provider = self.select_llm_provider(query_type)
        print(f"\n>>> Using {provider.__class__.__name__} for '{query_type}' query. <<<\n")
        return provider.generate_response(prompt)

# Core Chatbot Logic
class CustomerSupportChatbot:
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager

    def _categorize_query(self, user_query: str) -> str:
        # Simplified query categorization for demonstration
        user_query_lower = user_query.lower()
        if "bill" in user_query_lower or "payment" in user_query_lower or "invoice" in user_query_lower:
            return "billing"
        elif "product" in user_query_lower or "item" in user_query_lower or "feature" in user_query_lower:
            return "product_info"
        else:
            return "general"

    def handle_query(self, user_query: str) -> str:
        query_type = self._categorize_query(user_query)
        print(f"Categorized query as: {query_type}")
        return self.llm_manager.get_llm_response(user_query, query_type)

if __name__ == "__main__":
    print("Initializing Dynamic Customer Support Chatbot...")

    # Initialize LLM Manager
    llm_manager = LLMManager()

    # Load API keys from environment variables
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    # Register available LLM providers
    if OPENAI_API_KEY and openai is not None:
        try:
            openai_llm = OpenAILLM(api_key=OPENAI_API_KEY)
            llm_manager.register_llm_provider("openai", openai_llm)
            print("OpenAI LLM registered.")
        except Exception as e:
            print(f"Failed to register OpenAI LLM: {e}")
    else:
        print("OpenAI API key or library not found. OpenAI LLM not registered.")

    if GEMINI_API_KEY and genai is not None:
        try:
            gemini_llm = GeminiLLM(api_key=GEMINI_API_KEY)
            llm_manager.register_llm_provider("gemini", gemini_llm)
            print("Gemini LLM registered.")
        except Exception as e:
            print(f"Failed to register Gemini LLM: {e}")
    else:
        print("Gemini API key or library not found. Gemini LLM not registered.")

    # Always register a mock Llama for demonstration
    mock_llama_llm = MockLlamaLLM()
    llm_manager.register_llm_provider("llama", mock_llama_llm)
    print("Mock Llama LLM registered.")

    # Initialize Chatbot
    chatbot = CustomerSupportChatbot(llm_manager)

    print("\n--- Chatbot Ready! ---\n")

    # Example Usage
    queries = [
        "What is the status of my last payment?",
        "Tell me more about the features of your new smartphone model.",
        "I have a general question about shipping.",
        "Can I get an invoice for my subscription?",
        "How do I return a faulty product?"
    ]

    for i, query in enumerate(queries):
        print(f"\n--- User Query {i+1} ---")
        print(f"User: {query}")
        response = chatbot.handle_query(query)
        print(f"Chatbot: {response}")
        print("----------------------")
