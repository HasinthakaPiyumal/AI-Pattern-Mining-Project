import os
from abc import ABC, abstractmethod
from dotenv import load_dotenv
import openai
import google.generativeai as genai


class ConfigHandler:
    def __init__(self):
        load_dotenv()

    def get_openai_api_key(self):
        return os.getenv("OPENAI_API_KEY")

    def get_gemini_api_key(self):
        return os.getenv("GEMINI_API_KEY")


class LLMAbstractBase(ABC):
    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        pass


class GPTAdapter(LLMAbstractBase):
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)

    def generate_response(self, prompt: str) -> str:
        try:
            completion = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return completion.choices[0].message.content
        except openai.APIError as e:
            return f"Error from GPT: {e}"


class GeminiAdapter(LLMAbstractBase):
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')

    def generate_response(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error from Gemini: {e}"


class LlamaAdapter(LLMAbstractBase):
    def __init__(self):
        pass

    def generate_response(self, prompt: str) -> str:
        return f"[Mock Llama Response for '{prompt}']: This is a simulated response from Llama."


class LLMManager:
    def __init__(self, config: ConfigHandler):
        self.llm_providers = {
            "gpt": GPTAdapter(config.get_openai_api_key()) if config.get_openai_api_key() else None,
            "gemini": GeminiAdapter(config.get_gemini_api_key()) if config.get_gemini_api_key() else None,
            "llama": LlamaAdapter()
        }
        self._active_llm_name = None
        self._active_llm = None

        # Set default active LLM to the first available one
        for name, adapter in self.llm_providers.items():
            if adapter:
                self.switch_llm(name)
                break
        if not self._active_llm:
            print("Warning: No valid LLM provider configured. Chatbot may not function correctly.")

    def switch_llm(self, provider_name: str):
        provider_name = provider_name.lower()
        if provider_name in self.llm_providers and self.llm_providers[provider_name]:
            self._active_llm_name = provider_name
            self._active_llm = self.llm_providers[provider_name]
            print(f"Switched to {self._active_llm_name.upper()} LLM.")
        else:
            print(f"Error: LLM provider '{provider_name}' not found or not configured.")
            print("Available providers: " + ", ".join([k for k, v in self.llm_providers.items() if v is not None]))

    def generate_response(self, prompt: str) -> str:
        if self._active_llm:
            return self._active_llm.generate_response(prompt)
        else:
            return "No active LLM provider. Please configure one."

    def get_active_llm_name(self):
        return self._active_llm_name


class ChatbotApplication:
    def __init__(self):
        self.config = ConfigHandler()
        self.llm_manager = LLMManager(self.config)

    def run(self):
        print("\n--- Intelligent Customer Support Chatbot ---")
        print(f"Initial LLM: {self.llm_manager.get_active_llm_name().upper() if self.llm_manager.get_active_llm_name() else 'None'}")
        print("Type your query, or '/switch <provider>' to change LLM (e.g., /switch gemini).")
        print("Type 'exit' or press Ctrl+D to quit.")

        while True:
            try:
                user_input = input(f"\nYou ({self.llm_manager.get_active_llm_name().upper() if self.llm_manager.get_active_llm_name() else 'N/A'})> ").strip()
                if not user_input:
                    continue

                if user_input.lower() == 'exit':
                    print("Exiting chatbot. Goodbye!")
                    break

                if user_input.startswith('/switch '):
                    provider_name = user_input.split(' ', 1)[1].strip()
                    self.llm_manager.switch_llm(provider_name)
                else:
                    print(f"Chatbot ({self.llm_manager.get_active_llm_name().upper() if self.llm_manager.get_active_llm_name() else 'N/A'})> Thinking...")
                    response = self.llm_manager.generate_response(user_input)
                    print(f"Chatbot ({self.llm_manager.get_active_llm_name().upper() if self.llm_manager.get_active_llm_name() else 'N/A'})> {response}")
            except EOFError:
                print("\nExiting chatbot. Goodbye!")
                break
            except KeyboardInterrupt:
                print("\nExiting chatbot. Goodbye!")
                break


if __name__ == "__main__":
    app = ChatbotApplication()
    app.run()