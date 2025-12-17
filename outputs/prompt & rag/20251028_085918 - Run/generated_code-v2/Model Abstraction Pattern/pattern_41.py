from abc import ABC, abstractmethod

class LLMProvider(ABC):
    @abstractmethod
    def _invoke_model(self, formatted_prompt: str) -> str:
        pass

    @abstractmethod
    def _format_prompt(self, prompt: str) -> str:
        pass

class GPTProvider(LLMProvider):
    def _format_prompt(self, prompt: str) -> str:
        return f"User: {prompt}\nAssistant:"

    def _invoke_model(self, formatted_prompt: str) -> str:
        # Simulate GPT-like response
        if "complex" in formatted_prompt.lower() or "technical" in formatted_prompt.lower():
            return "(GPT - I can provide a detailed and comprehensive answer to your complex query.)"
        elif "unhappy" in formatted_prompt.lower() or "frustrated" in formatted_prompt.lower():
            return "(GPT - I understand your frustration. Let me help you with this.)"
        return "(GPT - Hello! How can I assist you today?)"

class GeminiProvider(LLMProvider):
    def _format_prompt(self, prompt: str) -> str:
        return f"human: {prompt}\nbot:"

    def _invoke_model(self, formatted_prompt: str) -> str:
        # Simulate Gemini-like response
        if "high complexity" in formatted_prompt.lower() or "detailed analysis" in formatted_prompt.lower():
            return "(Gemini - I'm equipped for high-complexity tasks. Let's delve into the details.)"
        return "(Gemini - How can I help you today?)"

class LlamaProvider(LLMProvider):
    def _format_prompt(self, prompt: str) -> str:
        return f"[INST] {prompt} [/INST]"

    def _invoke_model(self, formatted_prompt: str) -> str:
        # Simulate Llama-like response
        if "general question" in formatted_prompt.lower() or "simple query" in formatted_prompt.lower():
            return "(Llama - I can provide a quick answer to your general query.)"
        return "(Llama - What can I do for you?)"

class LLMAbstractor:
    def __init__(self):
        self.providers = {
            "gpt": GPTProvider(),
            "gemini": GeminiProvider(),
            "llama": LlamaProvider(),
        }

    def generate_response(self, prompt: str, model_name: str) -> str:
        provider = self.providers.get(model_name.lower())
        if not provider:
            raise ValueError(f"Unknown LLM provider: {model_name}")

        formatted_prompt = provider._format_prompt(prompt)
        response = provider._invoke_model(formatted_prompt)
        return response

class LLMRouter:
    def route_llm(self, query: str, user_sentiment: str = "neutral", query_complexity: str = "medium") -> str:
        if query_complexity == "high":
            return "gemini"
        elif user_sentiment == "negative":
            return "gpt"
        else:
            return "llama"

class ChatbotApplication:
    def __init__(self):
        self.llm_abstractor = LLMAbstractor()
        self.llm_router = LLMRouter()

    def process_user_query(self, query: str, user_sentiment: str = "neutral", query_complexity: str = "medium") -> str:
        selected_model = self.llm_router.route_llm(query, user_sentiment, query_complexity)
        response = self.llm_abstractor.generate_response(query, selected_model)
        return f"[Using {selected_model.upper()}] {response}"

    def run_chat(self):
        print("Welcome to the Multi-LLM Customer Support Chatbot!")
        print("Type 'exit' to end the chat.")

        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                print("Chat ended. Goodbye!")
                break

            # For demonstration, let's hardcode sentiment and complexity based on keywords
            # In a real app, these would come from NLP analysis
            sentiment = "neutral"
            complexity = "medium"

            if "unhappy" in user_input.lower() or "frustrated" in user_input.lower() or "issue" in user_input.lower():
                sentiment = "negative"
            if "complex" in user_input.lower() or "technical" in user_input.lower() or "detailed" in user_input.lower():
                complexity = "high"
            elif "simple" in user_input.lower() or "general" in user_input.lower():
                complexity = "low"

            response = self.process_user_query(user_input, user_sentiment=sentiment, query_complexity=complexity)
            print(f"Bot: {response}")

if __name__ == "__main__":
    app = ChatbotApplication()
    app.run_chat()