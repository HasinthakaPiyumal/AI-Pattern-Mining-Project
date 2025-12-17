import os
from abc import ABC, abstractmethod
from typing import Type

# Ensure dotenv is imported if you're using environment variables
from dotenv import load_dotenv

# Try to import openai and google.generativeai, handle if not installed
try:
    import openai
except ImportError:
    openai = None
    print("Warning: 'openai' library not found. GPTLLM will not be available.")

try:
    import google.generativeai as genai
except ImportError:
    genai = None
    print("Warning: 'google-generativeai' library not found. GeminiLLM will not be available.")


# config.py content
load_dotenv() # Load environment variables from .env file

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "gemini") # Default to gemini if not set


# llm_interface.py content
class LLMInterface(ABC):
    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        pass


# llm_providers.py content
class GPTLLM(LLMInterface):
    def __init__(self):
        if openai is None:
            raise ImportError("OpenAI library not installed. Cannot use GPTLLM.")
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set in environment variables.")
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)

    def generate_response(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful customer support assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error from GPTLLM: {e}"


class GeminiLLM(LLMInterface):
    def __init__(self):
        if genai is None:
            raise ImportError("Google Generative AI library not installed. Cannot use GeminiLLM.")
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not set in environment variables.")
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel("gemini-pro")

    def generate_response(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error from GeminiLLM: {e}"


# llm_factory.py content
class LLMProviderFactory:
    @staticmethod
    def get_llm_provider(provider_name: str) -> LLMInterface:
        if provider_name.lower() == "gpt":
            return GPTLLM()
        elif provider_name.lower() == "gemini":
            return GeminiLLM()
        else:
            raise ValueError(f"Unsupported LLM provider: {provider_name}")


# chatbot_core.py content
class ChatbotCore:
    def __init__(self, llm_provider: LLMInterface):
        self.llm_provider = llm_provider

    def process_query(self, user_query: str) -> str:
        # Here, you can add pre-processing of the query if needed
        response = self.llm_provider.generate_response(user_query)
        # Here, you can add post-processing of the response if needed
        return response


# main.py content (Example Usage)
if __name__ == "__main__":
    print(f"Default LLM Provider set to: {DEFAULT_LLM_PROVIDER}")

    try:
        # Get an LLM provider using the factory
        llm_instance = LLMProviderFactory.get_llm_provider(DEFAULT_LLM_PROVIDER)

        # Instantiate ChatbotCore with the chosen LLM provider
        chatbot = ChatbotCore(llm_instance)

        # Interact with the chatbot
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            
            if not user_input.strip():
                print("Chatbot: Please enter a query.")
                continue

            print(f"Chatbot ({DEFAULT_LLM_PROVIDER}): Thinking...")
            response = chatbot.process_query(user_input)
            print(f"Chatbot ({DEFAULT_LLM_PROVIDER}): {response}")

    except ValueError as e:
        print(f"Configuration Error: {e}")
        print("Please ensure API keys are set and provider names are correct.")
    except ImportError as e:
        print(f"Dependency Error: {e}")
        print("Please install the required libraries (e.g., pip install openai google-generativeai python-dotenv).")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")