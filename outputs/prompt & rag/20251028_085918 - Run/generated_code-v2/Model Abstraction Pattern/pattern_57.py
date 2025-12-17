import os
from abc import ABC, abstractmethod
from dotenv import load_dotenv

# --- llm_abstract_factory.py ---

class AbstractLLMProvider(ABC):
    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        pass

class OpenAILLMProvider(AbstractLLMProvider):
    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model_name = model_name
        # In a real application, you would initialize the OpenAI client here.
        # self.client = openai.OpenAI(api_key=api_key)

    def generate_response(self, prompt: str) -> str:
        # Mocking OpenAI API call for demonstration
        # In a real app:
        # response = self.client.chat.completions.create(
        #     model=self.model_name,
        #     messages=[{"role": "user", "content": prompt}]
        # )
        # return response.choices[0].message.content
        return f"[OpenAI {self.model_name}]: Responding to \"{prompt}\"."

    def get_model_name(self) -> str:
        return self.model_name

class GoogleGeminiLLMProvider(AbstractLLMProvider):
    def __init__(self, api_key: str, model_name: str = "gemini-pro"):
        self.api_key = api_key
        self.model_name = model_name
        # In a real application, you would initialize the Gemini client here.
        # import google.generativeai as genai
        # genai.configure(api_key=api_key)
        # self.model = genai.GenerativeModel(model_name)

    def generate_response(self, prompt: str) -> str:
        # Mocking Gemini API call for demonstration
        # In a real app:
        # response = self.model.generate_content(prompt)
        # return response.text
        return f"[Google Gemini {self.model_name}]: Responding to \"{prompt}\"."

    def get_model_name(self) -> str:
        return self.model_name

class LLMFactory:
    @staticmethod
    def get_provider(provider_type: str, api_key: str, model_name: str = None) -> AbstractLLMProvider:
        if provider_type.lower() == "openai":
            return OpenAILLMProvider(api_key, model_name or "gpt-3.5-turbo")
        elif provider_type.lower() == "gemini":
            return GoogleGeminiLLMProvider(api_key, model_name or "gemini-pro")
        else:
            raise ValueError(f"Unknown LLM provider type: {provider_type}")

# --- llm_provider_config.py ---

class LLMProviderConfig:
    def __init__(self):
        load_dotenv()  # Load environment variables from .env file

        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")

        self.openai_model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")
        self.gemini_model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-pro")

    def get_openai_config(self):
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        return {"api_key": self.openai_api_key, "model_name": self.openai_model_name}

    def get_gemini_config(self):
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        return {"api_key": self.gemini_api_key, "model_name": self.gemini_model_name}

# --- chatbot_core.py ---

class CustomerSupportChatbot:
    def __init__(self, llm_provider: AbstractLLMProvider):
        if not isinstance(llm_provider, AbstractLLMProvider):
            raise TypeError("llm_provider must be an instance of AbstractLLMProvider")
        self.llm_provider = llm_provider

    def process_message(self, user_message: str) -> str:
        chat_prompt = f"Customer inquiry: {user_message}. Provide a concise support response."
        llm_response = self.llm_provider.generate_response(chat_prompt)
        return llm_response

    def get_current_llm_info(self) -> str:
        return f"Currently using LLM: {self.llm_provider.get_model_name()}"

# --- main.py ---

def main():
    print("Initializing LLM Provider Configuration...")
    config = LLMProviderConfig()

    # --- Scenario 1: Using OpenAI Provider ---
    print("\n--- Scenario 1: Using OpenAI LLM Provider ---")
    try:
        openai_config = config.get_openai_config()
        openai_provider = LLMFactory.get_provider("openai", **openai_config)
        chatbot_openai = CustomerSupportChatbot(openai_provider)

        print(chatbot_openai.get_current_llm_info())
        response1 = chatbot_openai.process_message("My order #12345 is delayed. What's the status?")
        print(f"Chatbot: {response1}")

        response2 = chatbot_openai.process_message("I want to return an item. How do I initiate the process?")
        print(f"Chatbot: {response2}")

    except ValueError as e:
        print(f"Error with OpenAI configuration: {e}. Skipping OpenAI demo.")

    # --- Scenario 2: Switching to Google Gemini Provider ---
    print("\n--- Scenario 2: Switching to Google Gemini LLM Provider ---")
    try:
        gemini_config = config.get_gemini_config()
        gemini_provider = LLMFactory.get_provider("gemini", **gemini_config)
        chatbot_gemini = CustomerSupportChatbot(gemini_provider)

        print(chatbot_gemini.get_current_llm_info())
        response3 = chatbot_gemini.process_message("My subscription renewed automatically, but I want to cancel.")
        print(f"Chatbot: {response3}")

        response4 = chatbot_gemini.process_message("How can I update my shipping address?")
        print(f"Chatbot: {response4}")

    except ValueError as e:
        print(f"Error with Gemini configuration: {e}. Skipping Gemini demo.")

    print("\nDemonstration complete.")

if __name__ == "__main__":
    main()

# --- requirements.txt ---
# openai
# google-generativeai
# python-dotenv
