import abc
import os

class LLMProvider(abc.ABC):
    @abc.abstractmethod
    def generate_response(self, prompt: str) -> str:
        pass

class OpenAILLM(LLMProvider):
    def __init__(self, api_key: str):
        self._api_key = api_key

    def generate_response(self, prompt: str) -> str:
        return f"OpenAI's response to: '{prompt}' (using key ending in {self._api_key[-4:] if self._api_key else 'None'})"

class GeminiLLM(LLMProvider):
    def __init__(self, api_key: str):
        self._api_key = api_key

    def generate_response(self, prompt: str) -> str:
        return f"Gemini's response to: '{prompt}' (using key ending in {self._api_key[-4:] if self._api_key else 'None'})"

class LLMFactory:
    @staticmethod
    def get_llm_provider(provider_name: str, api_key: str) -> LLMProvider:
        if provider_name.lower() == "openai":
            return OpenAILLM(api_key)
        elif provider_name.lower() == "gemini":
            return GeminiLLM(api_key)
        else:
            raise ValueError(f"Unknown LLM provider: {provider_name}")

class CustomerSupportAssistant:
    def __init__(self, llm_provider: LLMProvider):
        self._llm_provider = llm_provider

    def handle_query(self, query: str) -> str:
        prompt = f"Customer query: {query}. Provide a concise answer."
        return self._llm_provider.generate_response(prompt)

    def troubleshoot(self, issue: str) -> str:
        prompt = f"Troubleshoot the following issue: {issue}. Suggest steps to resolve it."
        return self._llm_provider.generate_response(prompt)

    def escalate_issue(self, issue: str) -> str:
        prompt = f"Escalate the following customer issue: {issue}. Summarize for a human agent."
        llm_response = self._llm_provider.generate_response(prompt)
        return f"Escalated: {llm_response}"

def get_config():
    return {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "sk-openai-testkey1234"),
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY", "gemini-testkey5678"),
        "DEFAULT_LLM_PROVIDER": os.getenv("DEFAULT_LLM_PROVIDER", "openai")
    }

if __name__ == "__main__":
    config = get_config()

    print("--- Testing with OpenAI LLM ---")
    openai_api_key = config["OPENAI_API_KEY"]
    openai_provider = LLMFactory.get_llm_provider("openai", openai_api_key)
    assistant_openai = CustomerSupportAssistant(openai_provider)

    print(assistant_openai.handle_query("My internet is not working."))
    print(assistant_openai.troubleshoot("Email sending failed."))
    print(assistant_openai.escalate_issue("Customer is very angry about a recurring billing error."))
    print("-" * 30)

    print("--- Testing with Gemini LLM ---")
    gemini_api_key = config["GEMINI_API_KEY"]
    gemini_provider = LLMFactory.get_llm_provider("gemini", gemini_api_key)
    assistant_gemini = CustomerSupportAssistant(gemini_provider)

    print(assistant_gemini.handle_query("How do I reset my password?"))
    print(assistant_gemini.troubleshoot("Application crashes on startup."))
    print(assistant_gemini.escalate_issue("Critical data loss incident reported."))
    print("-" * 30)

    print("--- Demonstrating dynamic switching (e.g., based on default config) ---")
    default_provider_name = config["DEFAULT_LLM_PROVIDER"]
    default_api_key = config[f"{default_provider_name.upper()}_API_KEY"]
    default_llm_provider = LLMFactory.get_llm_provider(default_provider_name, default_api_key)
    assistant_default = CustomerSupportAssistant(default_llm_provider)
    print(f"Default assistant using: {default_llm_provider.__class__.__name__}")
    print(assistant_default.handle_query("What are your business hours?"))
    print("-" * 30)

    try:
        print("--- Testing unknown provider ---")
        LLMFactory.get_llm_provider("unknown", "some-key")
    except ValueError as e:
        print(f"Error: {e}")
