import os
from dotenv import load_dotenv
from abc import ABC, abstractmethod
import openai
import google.generativeai as genai

# config.py content
def load_environment_variables():
    load_dotenv()
    return {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
        "DEFAULT_LLM_PROVIDER": os.getenv("DEFAULT_LLM_PROVIDER", "openai")
    }

# llm_providers.py content
class AbstractLLMProvider(ABC):
    @abstractmethod
    def generate_response(self, prompt: str, history: list) -> str:
        pass

class OpenAIProvider(AbstractLLMProvider):
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)

    def generate_response(self, prompt: str, history: list) -> str:
        messages = []
        for entry in history:
            if entry["role"] == "user":
                messages.append({"role": "user", "content": entry["content"]})
            elif entry["role"] == "ai":
                messages.append({"role": "assistant", "content": entry["content"]})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error from OpenAI: {e}"

class GeminiProvider(AbstractLLMProvider):
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')

    def generate_response(self, prompt: str, history: list) -> str:
        convo_history = []
        for entry in history:
            if entry["role"] == "user":
                convo_history.append({"role": "user", "parts": [entry["content"]]})
            elif entry["role"] == "ai":
                convo_history.append({"role": "model", "parts": [entry["content"]]})
        
        try:
            convo = self.model.start_chat(history=convo_history)
            response = convo.send_message(prompt)
            return response.text
        except Exception as e:
            return f"Error from Gemini: {e}"

class LlamaProvider(AbstractLLMProvider):
    def __init__(self):
        pass

    def generate_response(self, prompt: str, history: list) -> str:
        return f"Llama (mock response): You asked '{prompt}'"

# chatbot_platform.py content
class ChatbotPlatform:
    def __init__(self, providers: dict, default_provider: str):
        self.providers = providers
        if default_provider not in self.providers:
            raise ValueError(f"Default provider '{default_provider}' not found in available providers.")
        self.current_provider = self.providers[default_provider]
        self.conversation_history = []

    def set_provider(self, provider_name: str):
        if provider_name not in self.providers:
            raise ValueError(f"Provider '{provider_name}' is not registered.")
        self.current_provider = self.providers[provider_name]
        print(f"Switched to LLM provider: {provider_name}")

    def send_message(self, user_message: str) -> str:
        self.conversation_history.append({"role": "user", "content": user_message})
        
        response_content = self.current_provider.generate_response(user_message, self.conversation_history)
        
        self.conversation_history.append({"role": "ai", "content": response_content})
        return response_content

    def get_history(self) -> list:
        return self.conversation_history

    def clear_history(self):
        self.conversation_history = []
        print("Conversation history cleared.")

# main.py content
if __name__ == "__main__":
    env_vars = load_environment_variables()

    openai_api_key = env_vars["OPENAI_API_KEY"]
    gemini_api_key = env_vars["GEMINI_API_KEY"]
    default_provider_name = env_vars["DEFAULT_LLM_PROVIDER"]

    # Initialize providers
    openai_provider = OpenAIProvider(api_key=openai_api_key) if openai_api_key else None
    gemini_provider = GeminiProvider(api_key=gemini_api_key) if gemini_api_key else None
    llama_provider = LlamaProvider()

    providers = {
        "openai": openai_provider,
        "gemini": gemini_provider,
        "llama": llama_provider
    }
    
    # Filter out None providers if API keys are missing
    active_providers = {name: prov for name, prov in providers.items() if prov is not None}
    
    if not active_providers:
        print("No LLM providers could be initialized. Please set API keys in .env file.")
    elif default_provider_name not in active_providers:
        print(f"Warning: Default provider '{default_provider_name}' is not active or configured. Using first available provider.")
        default_provider_name = list(active_providers.keys())[0]
        chatbot_platform = ChatbotPlatform(active_providers, default_provider_name)
    else:
        chatbot_platform = ChatbotPlatform(active_providers, default_provider_name)

    print(f"Chatbot initialized with default provider: {default_provider_name}")

    print("\n--- First interaction (using default provider) ---")
    response1 = chatbot_platform.send_message("Hello, tell me about the capital of France.")
    print(f"User: Hello, tell me about the capital of France.")
    print(f"Chatbot: {response1}")

    if "gemini" in active_providers and active_providers["gemini"] is not None:
        print("\n--- Switching to Gemini provider ---")
        chatbot_platform.set_provider("gemini")
        response2 = chatbot_platform.send_message("Now, provide some interesting facts about its history.")
        print(f"User: Now, provide some interesting facts about its history.")
        print(f"Chatbot: {response2}")
    else:
        print("\n--- Gemini provider not available or configured. Skipping switch. ---")

    if "llama" in active_providers and active_providers["llama"] is not None:
        print("\n--- Switching to Llama (mock) provider ---")
        chatbot_platform.set_provider("llama")
        response3 = chatbot_platform.send_message("Summarize the previous conversation.")
        print(f"User: Summarize the previous conversation.")
        print(f"Chatbot: {response3}")
    else:
        print("\n--- Llama provider not available. Skipping switch. ---")

    print("\n--- Conversation History ---")
    for entry in chatbot_platform.get_history():
        print(f"{entry['role'].capitalize()}: {entry['content']}")

    chatbot_platform.clear_history()
    print(chatbot_platform.get_history()) # Should be empty