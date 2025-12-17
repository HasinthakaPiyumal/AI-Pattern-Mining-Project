import os
import logging
from abc import ABC, abstractmethod
from dotenv import load_dotenv
import openai
import google.generativeai as genai

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "gemini")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.7))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 500))

class LLMService(ABC):
    @abstractmethod
    def generate_response(self, prompt: str, history: list = None) -> str:
        pass

class GPTService(LLMService):
    def __init__(self):
        if not Config.OPENAI_API_KEY:
            logging.error("OpenAI API key not found in environment variables.")
            raise ValueError("OpenAI API key is not set.")
        openai.api_key = Config.OPENAI_API_KEY

    def generate_response(self, prompt: str, history: list = None) -> str:
        messages = []
        if history:
            for item in history:
                messages.append({"role": "user", "content": item["user"]})
                messages.append({"role": "assistant", "content": item["assistant"]})
        messages.append({"role": "user", "content": prompt})
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=Config.LLM_TEMPERATURE,
                max_tokens=Config.LLM_MAX_TOKENS,
            )
            return response.choices[0].message["content"].strip()
        except openai.error.OpenAIError as e:
            logging.error(f"OpenAI API error: {e}")
            return f"An error occurred with GPT: {e}"
        except Exception as e:
            logging.error(f"Unexpected error with GPT: {e}")
            return f"An unexpected error occurred with GPT: {e}"

class GeminiService(LLMService):
    def __init__(self):
        if not Config.GEMINI_API_KEY:
            logging.error("Gemini API key not found in environment variables.")
            raise ValueError("Gemini API key is not set.")
        genai.configure(api_key=Config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-pro')

    def generate_response(self, prompt: str, history: list = None) -> str:
        gemini_history = []
        if history:
            for item in history:
                gemini_history.append({'role': 'user', 'parts': [item["user"]]})
                gemini_history.append({'role': 'model', 'parts': [item["assistant"]]})
        try:
            chat = self.model.start_chat(history=gemini_history)
            response = chat.send_message(prompt, 
                                       generation_config=genai.types.GenerationConfig(
                                           temperature=Config.LLM_TEMPERATURE,
                                           max_output_tokens=Config.LLM_MAX_TOKENS,
                                       ))
            return response.text.strip()
        except Exception as e:
            logging.error(f"Gemini API error: {e}")
            return f"An error occurred with Gemini: {e}"

class LlamaService(LLMService):
    def __init__(self):
        logging.warning("LlamaService is a placeholder and does not make actual API calls.")

    def generate_response(self, prompt: str, history: list = None) -> str:
        return f"[Llama Placeholder Response]: You asked: '{prompt}'. This is a mock response from Llama."

class LLMFactory:
    @staticmethod
    def get_llm_service(provider: str) -> LLMService:
        if provider.lower() == "gpt":
            return GPTService()
        elif provider.lower() == "gemini":
            return GeminiService()
        elif provider.lower() == "llama":
            return LlamaService()
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

class KnowledgeBase:
    def __init__(self):
        self._data = {
            "shipping": "Standard shipping takes 3-5 business days. Expedited shipping is available.",
            "return policy": "Items can be returned within 30 days of purchase with a receipt.",
            "contact support": "You can reach our support team at support@omnichat.com or call us at 1-800-OMNICHAT.",
            "product catalog": "Please visit our website for the latest product catalog and pricing."
        }

    def get_context(self, query: str) -> str:
        query_lower = query.lower()
        for keyword, info in self._data.items():
            if keyword in query_lower:
                return info
        return ""

class Chatbot:
    def __init__(self, llm_service: LLMService, knowledge_base: KnowledgeBase):
        self.llm_service = llm_service
        self.knowledge_base = knowledge_base
        self.conversation_history = []

    def ask(self, user_query: str) -> str:
        logging.info(f"User query: {user_query}")
        context = self.knowledge_base.get_context(user_query)
        
        full_prompt = user_query
        if context:
            full_prompt = f"Based on the following information: '{context}', answer the user's question: '{user_query}'"

        try:
            response = self.llm_service.generate_response(full_prompt, self.conversation_history)
            self.conversation_history.append({"user": user_query, "assistant": response})
            return response
        except ValueError as e: 
            logging.error(f"LLM service configuration error: {e}")
            return "I'm sorry, there's a problem with my current configuration. Please try again later."
        except Exception as e:
            logging.warning(f"Primary LLM failed: {e}. Attempting fallback (if any).")
            return "I'm experiencing some technical difficulties. Please try again or rephrase your question."

if __name__ == "__main__":
    print("OmniChat AI - Adaptive Customer Support Assistant")
    print("Type 'exit' to end the conversation.")

    try:
        current_llm_provider = Config.DEFAULT_LLM_PROVIDER
        print(f"Initializing with LLM provider: {current_llm_provider}")
        llm_service_instance = LLMFactory.get_llm_service(current_llm_provider)
        knowledge_base_instance = KnowledgeBase()
        chatbot = Chatbot(llm_service_instance, knowledge_base_instance)

        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                break
            
            response = chatbot.ask(user_input)
            print(f"Bot: {response}")

    except ValueError as e:
        print(f"Configuration Error: {e}")
        print("Please ensure your .env file is correctly set up with API keys.")
    except Exception as e:
        print(f"An unrecoverable error occurred during initialization: {e}")
