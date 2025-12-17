import abc
from typing import Dict, Type

class AbstractLLMProvider(abc.ABC):
    @abc.abstractmethod
    def generate_response(self, prompt: str) -> str:
        pass

class OpenAIProvider(AbstractLLMProvider):
    def __init__(self, api_key: str = "mock_openai_key"):
        self.api_key = api_key
        print(f"OpenAIProvider initialized with key: {"***" * 5 if api_key else "None"}")

    def generate_response(self, prompt: str) -> str:
        print(f"[OpenAI] Processing prompt: \'{prompt[:50]}...\' using key: {"***" * 5 if self.api_key else "None"}")
        return f"OpenAI (GPT) response to \'{prompt}\' - " \
               f"Here\'s a detailed answer regarding your query."


class GeminiProvider(AbstractLLMProvider):
    def __init__(self, api_key: str = "mock_gemini_key"):
        self.api_key = api_key
        print(f"GeminiProvider initialized with key: {"***" * 5 if api_key else "None"}")

    def generate_response(self, prompt: str) -> str:
        print(f"[Gemini] Processing prompt: \'{prompt[:50]}...\' using key: {"***" * 5 if self.api_key else "None"}")
        return f"Gemini response to \'{prompt}\' - " \
               f"I can help you with that! How about this solution?"


class LLMManager:
    def __init__(self):
        self._providers: Dict[str, AbstractLLMProvider] = {}
        self._current_provider: AbstractLLMProvider = None
        self._current_provider_name: str = ""

    def register_provider(self, name: str, provider_instance: AbstractLLMProvider):
        if not isinstance(provider_instance, AbstractLLMProvider):
            raise TypeError("Registered instance must be an AbstractLLMProvider.")
        self._providers[name] = provider_instance
        print(f"Provider \'{name}\' registered successfully.")

    def set_current_provider(self, name: str):
        if name not in self._providers:
            raise ValueError(f"Provider \'{name}\' not found. Available providers: {list(self._providers.keys())}")
        self._current_provider = self._providers[name]
        self._current_provider_name = name
        print(f"Switched to LLM provider: \'{name}\'")

    def get_response(self, prompt: str) -> str:
        if self._current_provider is None:
            raise RuntimeError("No LLM provider is currently set. Please call set_current_provider first.")
        return self._current_provider.generate_response(prompt)

    def get_current_provider_name(self) -> str:
        return self._current_provider_name


class CustomerSupportChatbot:
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
        print("CustomerSupportChatbot initialized.")

    def start_chat(self):
        print("\n--- Welcome to Dynamic Customer Support Chatbot ---")
        print("Type your query, or type \'switch [provider_name]\' to change LLM (e.g., \'switch openai\', \'switch gemini\').")
        print("Type \'exit\' to end the chat.")

        while True:
            user_input = input(f"\nYou ({self.llm_manager.get_current_provider_name()}): ").strip()

            if user_input.lower() == 'exit':
                print("Thank you for chatting! Goodbye.")
                break
            elif user_input.lower().startswith('switch '):
                _, provider_name = user_input.split(' ', 1)
                try:
                    self.llm_manager.set_current_provider(provider_name)
                except (ValueError, TypeError) as e:
                    print(f"Error switching provider: {e}")
                continue
            elif not user_input:
                continue

            try:
                response = self.llm_manager.get_response(user_input)
                print(f"Bot ({self.llm_manager.get_current_provider_name()}): {response}")
            except RuntimeError as e:
                print(f"Bot Error: {e}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    manager = LLMManager()

    openai_key = "" 
    gemini_key = "" 

    manager.register_provider("openai", OpenAIProvider(api_key=openai_key))
    manager.register_provider("gemini", GeminiProvider(api_key=gemini_key))

    try:
        manager.set_current_provider("openai")
    except ValueError as e:
        print(f"Initialization error: {e}. Please ensure providers are registered correctly.")
        exit()

    chatbot = CustomerSupportChatbot(manager)
    chatbot.start_chat()