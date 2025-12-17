import abc
import os

# 1. LLM Provider Interface (Abstract Base Class)
class AbstractLLMProvider(abc.ABC):
    @abc.abstractmethod
    def generate_response(self, prompt: str) -> str:
        pass

# 2. Concrete LLM Provider Implementations
class GPTProvider(AbstractLLMProvider):
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY", "MOCK_OPENAI_KEY")

    def generate_response(self, prompt: str) -> str:
        # In a real application, you would use the openai library here.
        # from openai import OpenAI
        # client = OpenAI(api_key=self.api_key)
        # response = client.chat.completions.create(
        #     model="gpt-3.5-turbo",
        #     messages=[
        #         {"role": "user", "content": prompt}
        #     ]
        # )
        # return response.choices[0].message.content
        return f"GPT-3.5-turbo response to: '{prompt}' (using key: {self.api_key[:5]}...)"

class GeminiProvider(AbstractLLMProvider):
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY", "MOCK_GEMINI_KEY")

    def generate_response(self, prompt: str) -> str:
        # In a real application, you would use the google-generativeai library here.
        # import google.generativeai as genai
        # genai.configure(api_key=self.api_key)
        # model = genai.GenerativeModel("gemini-pro")
        # response = model.generate_content(prompt)
        # return response.text
        return f"Gemini-Pro response to: '{prompt}' (using key: {self.api_key[:5]}...)"

class LlamaProvider(AbstractLLMProvider):
    def __init__(self):
        self.api_key = os.getenv("LLAMA_API_KEY", "MOCK_LLAMA_KEY")

    def generate_response(self, prompt: str) -> str:
        # In a real application, you would integrate with a Llama model,
        # e.g., via Hugging Face transformers or a custom API endpoint.
        return f"Llama-2 response to: '{prompt}' (using key: {self.api_key[:5]}...)"

# 3. LLM Manager/Factory
class LLMManager:
    def __init__(self):
        self._providers = {}
        self._current_provider_name = None

    def register_provider(self, name: str, provider: AbstractLLMProvider):
        self._providers[name] = provider
        if not self._current_provider_name:
            self._current_provider_name = name

    def set_current_provider(self, name: str):
        if name not in self._providers:
            raise ValueError(f"Provider '{name}' not registered.")
        self._current_provider_name = name
        print(f"Active LLM provider switched to: {name}")

    def get_current_provider(self) -> AbstractLLMProvider:
        if not self._current_provider_name:
            raise RuntimeError("No LLM provider is set as current.")
        return self._providers[self._current_provider_name]

    def generate_response(self, prompt: str) -> str:
        provider = self.get_current_provider()
        return provider.generate_response(prompt)

# 4. Chatbot Application Logic
if __name__ == "__main__":
    print("Starting Multi-Provider AI Customer Support Chatbot...")

    llm_manager = LLMManager()

    # Register providers
    llm_manager.register_provider("GPT", GPTProvider())
    llm_manager.register_provider("Gemini", GeminiProvider())
    llm_manager.register_provider("Llama", LlamaProvider())

    print("Available LLM providers:")
    for name in llm_manager._providers.keys():
        print(f"- {name}")

    print("Type 'exit' to quit. Type 'switch [provider_name]' to change LLM.")

    while True:
        user_input = input(f"\n[{llm_manager._current_provider_name}] You: ").strip()

        if user_input.lower() == "exit":
            print("Exiting chatbot. Goodbye!")
            break
        elif user_input.lower().startswith("switch "):
            parts = user_input.split(maxsplit=1)
            if len(parts) == 2:
                new_provider_name = parts[1]
                try:
                    llm_manager.set_current_provider(new_provider_name)
                except ValueError as e:
                    print(f"Error: {e}. Please choose from: {', '.join(llm_manager._providers.keys())}")
            else:
                print("Usage: switch [provider_name]")
            continue

        if not user_input:
            continue

        try:
            response = llm_manager.generate_response(user_input)
            print(f"[{llm_manager._current_provider_name}] Bot: {response}")
        except Exception as e:
            print(f"An error occurred: {e}")
