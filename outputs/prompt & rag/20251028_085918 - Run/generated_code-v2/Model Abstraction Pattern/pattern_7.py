from abc import ABC, abstractmethod

class AbstractLLM(ABC):
    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        pass

class GPTAdapter(AbstractLLM):
    def __init__(self, api_key: str = "dummy_gpt_key"):
        self.api_key = api_key

    def generate_response(self, prompt: str) -> str:
        return f"GPT's take on '{prompt}'. This is a detailed response from GPT, focusing on problem resolution and common FAQs."

class GeminiAdapter(AbstractLLM):
    def __init__(self, api_key: str = "dummy_gemini_key"):
        self.api_key = api_key

    def generate_response(self, prompt: str) -> str:
        return f"Gemini's perspective on '{prompt}'. This response from Gemini is concise and direct, often suitable for quick answers."

class LLMManager:
    def __init__(self, llm_providers: dict[str, AbstractLLM]):
        self.llm_providers = llm_providers

    def get_llm(self, provider_name: str) -> AbstractLLM:
        llm = self.llm_providers.get(provider_name)
        if not llm:
            raise ValueError(f"LLM provider '{provider_name}' not found.")
        return llm

    def route_query(self, query: str) -> AbstractLLM:
        query_lower = query.lower()

        if "technical" in query_lower or "troubleshoot" in query_lower or "problem" in query_lower:
            return self.get_llm("gpt")
        elif "quick question" in query_lower or "faq" in query_lower or "simple" in query_lower:
            return self.get_llm("gemini")
        else:
            return self.get_llm("gpt")

def main():
    gpt_model = GPTAdapter(api_key="sk-gpt-xxxx")
    gemini_model = GeminiAdapter(api_key="sk-gemini-yyyy")

    llm_manager = LLMManager({
        "gpt": gpt_model,
        "gemini": gemini_model
    })

    while True:
        user_query = input("You: ")
        if user_query.lower() == 'exit':
            break

        try:
            selected_llm = llm_manager.route_query(user_query)
            response = selected_llm.generate_response(user_query)
            print(f"Bot: {response}")
        except ValueError as e:
            print(f"Bot Error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()