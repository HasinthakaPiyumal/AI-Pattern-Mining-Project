import os
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import HuggingFaceHub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseChatModel, BaseLLM

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your_openai_api_key_here")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your_gemini_api_key_here")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "your_huggingface_api_key_here")

class LLMProvider:
    def get_llm(self):
        raise NotImplementedError

class OpenAIProvider(LLMProvider):
    def get_llm(self) -> BaseChatModel:
        return ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo", temperature=0.7)

class GeminiProvider(LLMProvider):
    def get_llm(self) -> BaseChatModel:
        return ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GEMINI_API_KEY, temperature=0.7)

class LlamaProvider(LLMProvider):
    def get_llm(self) -> BaseLLM:
        return HuggingFaceHub(
            repo_id="google/flan-t5-large",
            model_kwargs={"temperature": 0.5, "max_length": 64},
            huggingfacehub_api_token=HUGGINGFACE_API_KEY,
        )

class LLMManager:
    def __init__(self):
        self.providers = {
            "openai": OpenAIProvider(),
            "gemini": GeminiProvider(),
            "llama": LlamaProvider(),
        }
        self.current_provider_name = "openai"

    def set_provider(self, provider_name: str):
        if provider_name not in self.providers:
            raise ValueError(f"Unknown LLM provider: {provider_name}. Available: {list(self.providers.keys())}")
        self.current_provider_name = provider_name
        print(f"Switched to LLM provider: {self.current_provider_name}")

    def get_current_llm(self):
        return self.providers[self.current_provider_name].get_llm()

    def get_chain(self):
        llm = self.get_current_llm()
        
        if isinstance(llm, BaseChatModel):
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful customer support assistant. Answer user questions concisely and professionally."),
                ("user", "{question}")
            ])
        else:
            prompt = ChatPromptTemplate.from_template(
                "You are a helpful customer support assistant. Answer user questions concisely and professionally.\nQuestion: {question}\nAnswer:"
            )
        
        output_parser = StrOutputParser()
        return prompt | llm | output_parser

class CustomerSupportChatbot:
    def __init__(self):
        self.llm_manager = LLMManager()

    def process_query(self, query: str) -> str:
        try:
            chain = self.llm_manager.get_chain()
            response = chain.invoke({"question": query})
            return response
        except Exception as e:
            return f"Error processing query with {self.llm_manager.current_provider_name}: {e}"

    def switch_llm_provider(self, provider_name: str):
        try:
            self.llm_manager.set_provider(provider_name);
        except ValueError as e:
            print(f"Failed to switch provider: {e}")

def main():
    print("Initializing Customer Support Chatbot...")
    chatbot = CustomerSupportChatbot()

    print("\n--- Demonstrate switching LLM providers ---")

    print("\n--- Using OpenAI (default) ---")
    response_openai = chatbot.process_query("What is your return policy?")
    print(f"Chatbot (OpenAI): {response_openai}")

    chatbot.switch_llm_provider("gemini")
    response_gemini = chatbot.process_query("How do I reset my password?")
    print(f"Chatbot (Gemini): {response_gemini}")

    chatbot.switch_llm_provider("llama")
    response_llama = chatbot.process_query("What are your business hours?")
    print(f"Chatbot (Llama/HuggingFace): {response_llama}")

    chatbot.switch_llm_provider("unknown_llm")

    print("\n--- Start interactive chat (type 'exit' to quit) ---")
    while True:
        user_input = input(f"You ({chatbot.llm_manager.current_provider_name}): ")
        if user_input.lower() == 'exit':
            break
        if user_input.lower().startswith("/switch "):
            new_provider = user_input.split(" ")[1]
            chatbot.switch_llm_provider(new_provider)
            continue
        
        response = chatbot.process_query(user_input)
        print(f"Chatbot ({chatbot.llm_manager.current_provider_name}): {response}")

if __name__ == "__main__":
    main()