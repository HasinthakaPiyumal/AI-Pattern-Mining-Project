import os
from abc import ABC, abstractmethod
from dotenv import load_dotenv

# --- config.py ---
load_dotenv()

class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "gemini")

# --- llm_abstract.py ---
class AbstractLLM(ABC):
    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        pass

# --- llm_gemini.py ---
# For actual use, you would install google-generativeai
# pip install google-generativeai
try:
    import google.generativeai as genai
except ImportError:
    genai = None
    print("Warning: google-generativeai not installed. GeminiLLM will be non-functional.")

class GeminiLLM(AbstractLLM):
    def __init__(self, api_key: str):
        if genai is None:
            raise RuntimeError("google-generativeai is not installed. Cannot initialize GeminiLLM.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-pro")

    def generate_response(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error from Gemini: {e}"

# --- llm_openai_gpt.py ---
# For actual use, you would install openai
# pip install openai
try:
    import openai
except ImportError:
    openai = None
    print("Warning: openai library not installed. OpenAIGPTLLM will be non-functional.")

class OpenAIGPTLLM(AbstractLLM):
    def __init__(self, api_key: str):
        if openai is None:
            raise RuntimeError("openai library is not installed. Cannot initialize OpenAIGPTLLM.")
        openai.api_key = api_key

    def generate_response(self, prompt: str) -> str:
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",  # Or "gpt-4"
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error from OpenAI GPT: {e}"

# --- llm_llama.py (Placeholder) ---
class LlamaLLM(AbstractLLM):
    def __init__(self, api_key: str = None):
        self.api_key = api_key # Llama API key might be used here for a real implementation

    def generate_response(self, prompt: str) -> str:
        return f"[Llama Placeholder Response] Responding to: '{prompt}' - How can I assist you today?"

# --- llm_factory.py ---
class LLMProviderFactory:
    @staticmethod
    def get_llm(provider_name: str, api_key: str) -> AbstractLLM:
        if provider_name.lower() == "gemini":
            return GeminiLLM(api_key)
        elif provider_name.lower() == "gpt":
            return OpenAIGPTLLM(api_key)
        elif provider_name.lower() == "llama":
            return LlamaLLM(api_key)
        else:
            raise ValueError(f"Unknown LLM provider: {provider_name}")

# --- customer_support_agent.py ---
class CustomerSupportAgent:
    def __init__(self, llm_instance: AbstractLLM):
        self.llm = llm_instance

    def handle_query(self, query: str) -> str:
        print(f"Agent received query: '{query}'")
        response = self.llm.generate_response(query)
        return response

# --- main.py ---
if __name__ == "__main__":
    print(f"Loading configuration. Default LLM Provider: {Config.DEFAULT_LLM_PROVIDER}")

    selected_provider = Config.DEFAULT_LLM_PROVIDER
    api_key = None

    if selected_provider.lower() == "gemini":
        api_key = Config.GEMINI_API_KEY
        if not api_key:
            print("GEMINI_API_KEY not found in .env. GeminiLLM might not work correctly.")
    elif selected_provider.lower() == "gpt":
        api_key = Config.OPENAI_API_KEY
        if not api_key:
            print("OPENAI_API_KEY not found in .env. OpenAIGPTLLM might not work correctly.")
    elif selected_provider.lower() == "llama":
        # For Llama placeholder, API key is optional
        pass
    else:
        print(f"Invalid DEFAULT_LLM_PROVIDER '{selected_provider}' specified in .env. Falling back to Llama placeholder.")
        selected_provider = "llama"

    try:
        llm_instance = LLMProviderFactory.get_llm(selected_provider, api_key)
        agent = CustomerSupportAgent(llm_instance)

        customer_query_1 = "What is your return policy?"
        response_1 = agent.handle_query(customer_query_1)
        print(f"\nAgent Response (Provider: {selected_provider}): {response_1}")

        customer_query_2 = "I need help tracking my order #12345."
        response_2 = agent.handle_query(customer_query_2)
        print(f"\nAgent Response (Provider: {selected_provider}): {response_2}")

    except Exception as e:
        print(f"An error occurred during agent initialization or query handling: {e}")
        print("Please ensure required LLM libraries are installed and API keys are configured in a .env file.")
