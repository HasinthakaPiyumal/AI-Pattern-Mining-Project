import abc

# 1. LLM Abstraction Layer
class AbstractLLM(abc.ABC):
    @abc.abstractmethod
    def generate_response(self, prompt: str) -> str:
        pass

class GPTAdapter(AbstractLLM):
    def generate_response(self, prompt: str) -> str:
        return f"GPT Mock Response for: '{prompt}'"

class GeminiAdapter(AbstractLLM):
    def generate_response(self, prompt: str) -> str:
        return f"Gemini Mock Response for: '{prompt}'"

class LlamaAdapter(AbstractLLM):
    def generate_response(self, prompt: str) -> str:
        return f"Llama Mock Response for: '{prompt}'"

# 2. LLM Manager/Factory
class LLMFactory:
    def __init__(self, config: dict):
        self.config = config

    def get_llm(self, provider: str) -> AbstractLLM:
        if provider == "GPT":
            return GPTAdapter()
        elif provider == "Gemini":
            return GeminiAdapter()
        elif provider == "Llama":
            return LlamaAdapter()
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

# 3. Chatbot Application Logic
class Chatbot:
    def __init__(self, llm: AbstractLLM):
        self.llm = llm

    def get_answer(self, user_query: str) -> str:
        return self.llm.generate_response(user_query)

# 4. Configuration and Main Application Flow
if __name__ == "__main__":
    # --- Configuration ---
    # Change this to switch LLM providers
    CURRENT_LLM_PROVIDER = "Gemini" # Options: "GPT", "Gemini", "Llama"

    print(f"\n--- Initializing Chatbot with {CURRENT_LLM_PROVIDER} provider ---")
    llm_factory = LLMFactory(config={})
    
    try:
        chosen_llm = llm_factory.get_llm(CURRENT_LLM_PROVIDER)
        chatbot_app = Chatbot(chosen_llm)

        print("Type 'exit' or 'quit' to end the chat.")
        while True:
            user_input = input("You: ")
            if user_input.lower() in ("exit", "quit"):
                print("Chatbot: Goodbye!")
                break
            
            response = chatbot_app.get_answer(user_input)
            print(f"Chatbot ({CURRENT_LLM_PROVIDER}): {response}")
            
    except ValueError as e:
        print(f"Error: {e}")
        print("Please configure a valid LLM provider (GPT, Gemini, Llama).")

    # --- Demonstration of switching providers dynamically ---
    print("\n--- Demonstrating dynamic LLM switching to Llama ---")
    try:
        switched_llm = llm_factory.get_llm("Llama")
        chatbot_app.llm = switched_llm # Update the chatbot's LLM instance
        CURRENT_LLM_PROVIDER = "Llama"

        test_query = "What are the main features of the latest update?"
        print(f"You (after switch): {test_query}")
        response_after_switch = chatbot_app.get_answer(test_query)
        print(f"Chatbot ({CURRENT_LLM_PROVIDER}): {response_after_switch}")

    except ValueError as e:
        print(f"Error during switch: {e}")
