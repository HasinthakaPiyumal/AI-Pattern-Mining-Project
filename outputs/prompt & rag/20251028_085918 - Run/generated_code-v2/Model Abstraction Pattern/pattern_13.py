import abc
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- 1. LLM Abstraction Layer ---

class LLMAbstract(abc.ABC):

    @abc.abstractmethod
    def generate_response(self, prompt: str) -> str:
        pass

    @abc.abstractmethod
    def get_model_info(self) -> dict:
        pass


class GPTLLM(LLMAbstract):

    def __init__(self, api_key: str):
        self.api_key = api_key
        # In a real application, you would initialize the OpenAI client here
        # from openai import OpenAI
        # self.client = OpenAI(api_key=api_key)

    def generate_response(self, prompt: str) -> str:
        # Placeholder for actual OpenAI API call
        # response = self.client.chat.completions.create(...)
        return f"GPT-powered response to: '{prompt}' (using API key: {self.api_key[:5]}...)"

    def get_model_info(self) -> dict:
        return {"provider": "OpenAI", "model": "GPT-3.5-turbo", "version": "latest"}


class GeminiLLM(LLMAbstract):

    def __init__(self, api_key: str):
        self.api_key = api_key
        # In a real application, you would initialize the Google Generative AI client here
        # import google.generativeai as genai
        # genai.configure(api_key=api_key)
        # self.model = genai.GenerativeModel('gemini-pro')

    def generate_response(self, prompt: str) -> str:
        # Placeholder for actual Google Gemini API call
        # response = self.model.generate_content(prompt)
        return f"Gemini-powered response to: '{prompt}' (using API key: {self.api_key[:5]}...)"

    def get_model_info(self) -> dict:
        return {"provider": "Google", "model": "Gemini-Pro", "version": "latest"}


class LlamaLLM(LLMAbstract):

    def __init__(self, api_key: str = "mock_llama_key"):
        self.api_key = api_key # Llama often runs locally or via specific APIs

    def generate_response(self, prompt: str) -> str:
        # Placeholder for actual Llama interaction (e.g., via Hugging Face Transformers or local server)
        return f"Llama-powered response to: '{prompt}' (using mock key: {self.api_key[:5]}...)"

    def get_model_info(self) -> dict:
        return {"provider": "Meta", "model": "Llama-2-7b", "version": "2.0"}


# --- 2. LLM Provider Manager ---

class LLMManager:

    def __init__(self):
        self._llm_providers = {}
        self._current_provider_name = None

    def register_llm_provider(self, name: str, llm_instance: LLMAbstract):
        self._llm_providers[name] = llm_instance

    def set_current_provider(self, provider_name: str):
        if provider_name not in self._llm_providers:
            raise ValueError(f"Provider '{provider_name}' not registered.")
        self._current_provider_name = provider_name

    def get_current_llm(self) -> LLMAbstract:
        if not self._current_provider_name:
            raise RuntimeError("No LLM provider has been set.")
        return self._llm_providers[self._current_provider_name]

    def list_providers(self) -> list[str]:
        return list(self._llm_providers.keys())


# --- 3. Customer Support Chatbot Application Logic ---

class CustomerSupportChatbot:

    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
        self.conversation_history = []

    def _format_prompt(self, user_query: str) -> str:
        # Simple prompt formatting, can be more complex with context, roles, etc.
        history_str = "\n".join(self.conversation_history)
        if history_str:
            return f"Conversation History:\n{history_str}\nUser: {user_query}\nAssistant:"
        return f"User: {user_query}\nAssistant:"

    def handle_query(self, user_query: str) -> str:
        formatted_prompt = self._format_prompt(user_query)
        try:
            llm_response = self.llm_manager.get_current_llm().generate_response(formatted_prompt)
            self.conversation_history.append(f"User: {user_query}")
            self.conversation_history.append(f"Assistant: {llm_response}")
            return llm_response
        except Exception as e:
            return f"Error handling query with current LLM provider: {e}"

    def switch_llm_provider(self, provider_name: str):
        try:
            self.llm_manager.set_current_provider(provider_name)
            print(f"Switched LLM provider to: {provider_name}")
            # Clear history or adapt as needed for new model context
            self.conversation_history = []
        except ValueError as e:
            print(f"Failed to switch provider: {e}")

    def get_current_llm_info(self) -> dict:
        try:
            return self.llm_manager.get_current_llm().get_model_info()
        except RuntimeError:
            return {"status": "No LLM provider set"}


# --- Main Application Logic / Demonstration ---

if __name__ == "__main__":
    # Initialize LLM Manager
    llm_manager = LLMManager()

    # Get API keys from environment variables
    gpt_api_key = os.getenv("OPENAI_API_KEY", "sk-mock_gpt_key")
    gemini_api_key = os.getenv("GEMINI_API_KEY", "mock_gemini_key")

    # Register LLM providers
    llm_manager.register_llm_provider("GPT", GPTLLM(api_key=gpt_api_key))
    llm_manager.register_llm_provider("Gemini", GeminiLLM(api_key=gemini_api_key))
    llm_manager.register_llm_provider("Llama", LlamaLLM())

    print(f"Available LLM providers: {llm_manager.list_providers()}")

    # Initialize Chatbot with the manager
    chatbot = CustomerSupportChatbot(llm_manager)

    print("\n--- Starting Chatbot Session ---")

    # Set initial provider (e.g., based on config or default)
    chatbot.switch_llm_provider("GPT")
    print(f"Current LLM: {chatbot.get_current_llm_info()}")

    # Simulate customer queries
    response1 = chatbot.handle_query("What is the return policy for electronics?")
    print(f"Chatbot: {response1}")

    response2 = chatbot.handle_query("How long does shipping usually take?")
    print(f"Chatbot: {response2}")

    print("\n--- Switching to Gemini ---")
    chatbot.switch_llm_provider("Gemini")
    print(f"Current LLM: {chatbot.get_current_llm_info()}")

    response3 = chatbot.handle_query("I want to track my order. My order number is #12345.")
    print(f"Chatbot: {response3}")

    response4 = chatbot.handle_query("Can I change my delivery address?")
    print(f"Chatbot: {response4}")

    print("\n--- Switching to Llama ---")
    chatbot.switch_llm_provider("Llama")
    print(f"Current LLM: {chatbot.get_current_llm_info()}")

    response5 = chatbot.handle_query("Tell me about your loyalty program.")
    print(f"Chatbot: {response5}")

    print("\n--- Attempting to switch to an unregistered provider ---")
    chatbot.switch_llm_provider("UnknownLLM")

    print("\n--- Final Chatbot History ---")
    for entry in chatbot.conversation_history:
        print(entry)