from abc import ABC, abstractmethod
import os
from openai import OpenAI
import google.generativeai as genai

class AbstractLLM(ABC):
    @abstractmethod
    def generate_response(self, prompt: str, **kwargs) -> str:
        pass

class OpenAIGPTModel(AbstractLLM):
    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo"):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def generate_response(self, prompt: str, **kwargs) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error from OpenAI: {e}"

class GoogleGeminiModel(AbstractLLM):
    def __init__(self, api_key: str, model_name: str = "gemini-pro"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def generate_response(self, prompt: str, **kwargs) -> str:
        try:
            response = self.model.generate_content(prompt, **kwargs)
            return response.text
        except Exception as e:
            return f"Error from Gemini: {e}"

class LLMFactory:
    @staticmethod
    def get_llm(provider_name: str, api_key: str, model_name: str = None) -> AbstractLLM:
        if provider_name.lower() == "openai":
            if not api_key:
                raise ValueError("OpenAI API key not found.")
            return OpenAIGPTModel(api_key, model_name or "gpt-3.5-turbo")
        elif provider_name.lower() == "gemini":
            if not api_key:
                raise ValueError("Google Gemini API key not found.")
            return GoogleGeminiModel(api_key, model_name or "gemini-pro")
        else:
            raise ValueError(f"Unknown LLM provider: {provider_name}")

class LLMRouter:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")

    def get_optimal_llm(self, query: str, cost_priority: bool = False, latency_priority: bool = False) -> AbstractLLM:
        query_length = len(query.split())

        # Simple routing logic:
        # Prioritize OpenAI for shorter, general queries (simulating cost/latency preference)
        # Prioritize Gemini for longer, potentially more complex queries

        if cost_priority or latency_priority or query_length < 10:
            if self.openai_api_key:
                return LLMFactory.get_llm("openai", self.openai_api_key)
            elif self.gemini_api_key:
                return LLMFactory.get_llm("gemini", self.gemini_api_key)
        
        if self.gemini_api_key:
            return LLMFactory.get_llm("gemini", self.gemini_api_key)
        elif self.openai_api_key:
            return LLMFactory.get_llm("openai", self.openai_api_key)
            
        raise RuntimeError("No LLM provider configured or available.")

class CustomerSupportChatbot:
    def __init__(self):
        self.llm_router = LLMRouter()

    def get_response(self, query: str, cost_priority: bool = False, latency_priority: bool = False) -> str:
        selected_llm = self.llm_router.get_optimal_llm(query, cost_priority, latency_priority)
        response = selected_llm.generate_response(prompt=query)
        return response

if __name__ == "__main__":
    # Make sure to set your API keys as environment variables:
    # export OPENAI_API_KEY="your_openai_api_key_here"
    # export GEMINI_API_KEY="your_gemini_api_key_here"

    chatbot = CustomerSupportChatbot()

    print("\n--- Simple Query (should lean towards OpenAI if available) ---")
    response1 = chatbot.get_response("What is the return policy?")
    print(f"Chatbot: {response1}")

    print("\n--- Complex Query (should lean towards Gemini if available) ---")
    response2 = chatbot.get_response("I received a damaged item, order number #12345. It was a red dress, size M. What are my options for a full refund or exchange, and how do I initiate the process? Please provide detailed steps.")
    print(f"Chatbot: {response2}")

    print("\n--- Query with cost priority (should lean towards cheaper/faster if available) ---")
    response3 = chatbot.get_response("Where is my order?", cost_priority=True)
    print(f"Chatbot: {response3}")

    print("\n--- Another simple query ---")
    response4 = chatbot.get_response("Do you offer international shipping?")
    print(f"Chatbot: {response4}")
