import os
from abc import ABC, abstractmethod
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

class AbstractLLMProvider(ABC):
    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        pass

class OpenAILLMAdapter(AbstractLLMProvider):
    def __init__(self):
        self.llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate_response(self, prompt: str) -> str:
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content if hasattr(response, 'content') else str(response)

class GoogleGeminiAdapter(AbstractLLMProvider):
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(google_api_key=os.getenv("GOOGLE_API_KEY"))

    def generate_response(self, prompt: str) -> str:
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content if hasattr(response, 'content') else str(response)

class HuggingFaceMockLLMAdapter(AbstractLLMProvider):
    def generate_response(self, prompt: str) -> str:
        return f"[HuggingFace Mock Response for '{prompt}']: I am a specialized model for certain keywords."

class LLMFactory:
    @staticmethod
    def get_llm(provider_name: str, available_llms: dict) -> AbstractLLMProvider:
        if provider_name in available_llms:
            return available_llms[provider_name]
        raise ValueError(f"Unknown LLM provider: {provider_name}")

class LLMRouter:
    def __init__(self, available_llms: dict):
        self.available_llms = available_llms

    def route_query(self, query: str) -> str:
        query_lower = query.lower()

        if "pricing" in query_lower or "billing" in query_lower or len(query) > 100:
            return "openai"
        elif "hello" in query_lower or "greeting" in query_lower or "hi" in query_lower:
            return "gemini"
        elif "special" in query_lower or "unique" in query_lower:
            return "huggingface_mock"
        else:
            return "gemini"

class CustomerSupportChatbot:
    def __init__(self):
        self.available_llms = {
            "openai": OpenAILLMAdapter(),
            "gemini": GoogleGeminiAdapter(),
            "huggingface_mock": HuggingFaceMockLLMAdapter()
        }
        self.router = LLMRouter(self.available_llms)

    def run(self):
        print("\n--- Multi-LLM Customer Support Chatbot ---")
        print("Type 'exit' to quit.")

        while True:
            user_query = input("\nYou: ")
            if user_query.lower() == 'exit':
                break
            if not user_query.strip():
                print("Bot: Please enter a query.")
                continue

            try:
                chosen_provider_name = self.router.route_query(user_query)
                selected_llm = LLMFactory.get_llm(chosen_provider_name, self.available_llms)
                print(f"[Debug: Routing to {chosen_provider_name} model]")
                response = selected_llm.generate_response(user_query)
                print(f"Bot: {response}")
            except Exception as e:
                print(f"Bot Error: An error occurred - {e}")
                print("Bot: I apologize, but I'm having trouble processing your request right now. Please try again later.")

if __name__ == "__main__":
    chatbot = CustomerSupportChatbot()
    chatbot.run()
