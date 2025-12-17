import os
from abc import ABC, abstractmethod
from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage

# --- Configuration (config.py logic) ---
load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "gemini")
    ROUTING_RULES = {
        "billing": "gpt",
        "technical": "gemini",
        "pricing": "gpt",
        "account": "gemini",
    }

# --- LLM Abstraction Layer (llm_abstraction.py logic) ---

class LLMProvider(ABC):
    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        pass

class GPTProvider(LLMProvider):
    def __init__(self):
        if not Config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        self.llm = ChatOpenAI(api_key=Config.OPENAI_API_KEY, model="gpt-3.5-turbo")

    def generate_response(self, prompt: str) -> str:
        messages = [HumanMessage(content=prompt)]
        response = self.llm.invoke(messages)
        return response.content

class GeminiProvider(LLMProvider):
    def __init__(self):
        if not Config.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=Config.GOOGLE_API_KEY)

    def generate_response(self, prompt: str) -> str:
        messages = [HumanMessage(content=prompt)]
        response = self.llm.invoke(messages)
        return response.content

class LlamaProvider(LLMProvider):
    def generate_response(self, prompt: str) -> str:
        return f"[Llama Placeholder Response]: I am a Llama model, and for the prompt: '{prompt}', I would generate a response here."

class LLMFactory:
    def get_provider(self, provider_name: str) -> LLMProvider:
        if provider_name == "gpt":
            return GPTProvider()
        elif provider_name == "gemini":
            return GeminiProvider()
        elif provider_name == "llama":
            return LlamaProvider()
        else:
            raise ValueError(f"Unknown LLM provider: {provider_name}")

# --- LLM Router (router.py logic) ---

class LLMRouter:
    def __init__(self, routing_rules: dict, default_provider: str):
        self.routing_rules = routing_rules
        self.default_provider = default_provider

    def route_llm(self, query: str) -> str:
        query_lower = query.lower()
        for keyword, provider in self.routing_rules.items():
            if keyword in query_lower:
                return provider
        return self.default_provider

# --- Chatbot Core (chatbot.py logic) ---

class CustomerSupportChatbot:
    def __init__(self, llm_router: LLMRouter, llm_factory: LLMFactory):
        self.llm_router = llm_router
        self.llm_factory = llm_factory

    def get_chatbot_response(self, user_query: str) -> str:
        chosen_provider_name = self.llm_router.route_llm(user_query)
        llm_provider = self.llm_factory.get_provider(chosen_provider_name)
        response = llm_provider.generate_response(user_query)
        return f"[Using {chosen_provider_name.upper()}]: {response}"

# --- Main Application Entry Point (main.py logic) ---

def main():
    print("Initializing Customer Support Chatbot...")

    llm_factory = LLMFactory()
    llm_router = LLMRouter(Config.ROUTING_RULES, Config.DEFAULT_LLM_PROVIDER)
    chatbot = CustomerSupportChatbot(llm_router, llm_factory)

    print(f"Chatbot initialized. Default LLM: {Config.DEFAULT_LLM_PROVIDER.upper()}")
    print("Type 'exit' to quit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        try:
            response = chatbot.get_chatbot_response(user_input)
            print(f"Bot: {response}")
        except ValueError as e:
            print(f"Bot Error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()