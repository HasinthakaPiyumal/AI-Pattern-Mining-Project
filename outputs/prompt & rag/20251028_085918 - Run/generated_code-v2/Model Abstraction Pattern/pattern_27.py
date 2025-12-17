import abc

# config.py
CONFIG = {
    "openai_api_key": "your_openai_api_key",
    "gemini_api_key": "your_gemini_api_key",
    "llama_api_key": "your_llama_api_key",
    "default_llm_provider": "openai",
}

# llm_abstraction/abstract_llm.py
class AbstractLLM(abc.ABC):
    @abc.abstractmethod
    def generate_response(self, prompt: str) -> str:
        pass

# llm_abstraction/openai_llm.py
class OpenAILLM(AbstractLLM):
    def __init__(self, api_key: str):
        self.api_key = api_key
        print(f"OpenAILLM initialized with API key: {api_key[:5]}...")

    def generate_response(self, prompt: str) -> str:
        print(f"Simulating OpenAI GPT response for prompt: '{prompt}'")
        # In a real scenario, this would call the OpenAI API
        return f"OpenAI GPT's response to '{prompt}'."

# llm_abstraction/gemini_llm.py
class GeminiLLM(AbstractLLM):
    def __init__(self, api_key: str):
        self.api_key = api_key
        print(f"GeminiLLM initialized with API key: {api_key[:5]}...")

    def generate_response(self, prompt: str) -> str:
        print(f"Simulating Google Gemini response for prompt: '{prompt}'")
        # In a real scenario, this would call the Google Gemini API
        return f"Google Gemini's response to '{prompt}'."

# llm_abstraction/llama_llm.py
class LlamaLLM(AbstractLLM):
    def __init__(self, api_key: str):
        self.api_key = api_key
        print(f"LlamaLLM initialized with API key: {api_key[:5]}...")

    def generate_response(self, prompt: str) -> str:
        print(f"Simulating Meta Llama response for prompt: '{prompt}'")
        # In a real scenario, this would call the Llama API
        return f"Meta Llama's response to '{prompt}'."

# llm_abstraction/llm_factory.py
class LLMFactory:
    @staticmethod
    def get_llm(provider_name: str, config: dict) -> AbstractLLM:
        if provider_name.lower() == "openai":
            return OpenAILLM(config["openai_api_key"])
        elif provider_name.lower() == "gemini":
            return GeminiLLM(config["gemini_api_key"])
        elif provider_name.lower() == "llama":
            return LlamaLLM(config["llama_api_key"])
        else:
            raise ValueError(f"Unknown LLM provider: {provider_name}")

# llm_abstraction/chatbot_core.py
class CustomerSupportChatbot:
    def __init__(self, llm: AbstractLLM):
        self.llm = llm
        print("CustomerSupportChatbot initialized with an LLM instance.")

    def handle_query(self, query: str) -> str:
        print(f"Chatbot handling query: '{query}'")
        response = self.llm.generate_response(query)
        print(f"Chatbot received response.")
        return response

# main.py
if __name__ == "__main__":
    print("\n--- Demonstrating Model Abstraction Pattern ---")

    # Scenario 1: Using OpenAI LLM
    print("\n--- Using OpenAI LLM ---")
    openai_llm_instance = LLMFactory.get_llm("openai", CONFIG)
    chatbot_openai = CustomerSupportChatbot(openai_llm_instance)
    response_openai = chatbot_openai.handle_query("What is the return policy?")
    print(f"Final Chatbot Response (OpenAI): {response_openai}")

    # Scenario 2: Using Gemini LLM
    print("\n--- Using Gemini LLM ---")
    gemini_llm_instance = LLMFactory.get_llm("gemini", CONFIG)
    chatbot_gemini = CustomerSupportChatbot(gemini_llm_instance)
    response_gemini = chatbot_gemini.handle_query("How can I track my order?")
    print(f"Final Chatbot Response (Gemini): {response_gemini}")

    # Scenario 3: Dynamically switching to Llama LLM
    print("\n--- Dynamically switching to Llama LLM ---")
    llama_llm_instance = LLMFactory.get_llm("llama", CONFIG)
    chatbot_llama = CustomerSupportChatbot(llama_llm_instance)
    response_llama = chatbot_llama.handle_query("Can you help me with a product recommendation?")
    print(f"Final Chatbot Response (Llama): {response_llama}")

    # Demonstrate changing LLM in an existing chatbot (if needed, though usually new instance is better)
    print("\n--- Changing LLM in an existing chatbot ---")
    print("Re-configuring chatbot_openai to use Gemini")
    chatbot_openai.llm = LLMFactory.get_llm("gemini", CONFIG) # This shows the flexibility
    response_switched = chatbot_openai.handle_query("Where is my package?")
    print(f"Final Chatbot Response (Switched to Gemini): {response_switched}")

    print("\n--- Model Abstraction Demonstration Complete ---")