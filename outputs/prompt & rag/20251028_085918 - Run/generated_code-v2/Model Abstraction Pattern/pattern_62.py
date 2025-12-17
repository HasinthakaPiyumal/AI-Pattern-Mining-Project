import os
from abc import ABC, abstractmethod
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- 1. Unified LLM Interface/Adapter Layer ---
class AbstractLLM(ABC):
    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        pass

class GPTAdapter(AbstractLLM):
    def __init__(self):
        self.api_key = os.getenv("GPT_API_KEY", "sk-mock-gpt-key")
        self.model_name = os.getenv("GPT_MODEL_NAME", "gpt-3.5-turbo")

    def generate_response(self, prompt: str) -> str:
        print(f"\n[GPT Adapter] Using model: {self.model_name}, API Key: {self.api_key[:5]}...")
        print(f"[GPT Adapter] Sending prompt: '{prompt}'")
        # Simulate API call to GPT
        if "price" in prompt.lower() or "cost" in prompt.lower():
            return "The price for that item is $99.99. Is there anything else I can help you with रिजनिंग क्विज?"
        elif "complex" in prompt.lower():
            return "[GPT Advanced Response] I understand this is a complex issue. Let me provide a detailed explanation of potential solutions."
        return f"[GPT Basic Response] Thank you for your query: '{prompt}'. I'm processing your request with GPT."

class GeminiAdapter(AbstractLLM):
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY", "mock-gemini-key")
        self.model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-pro")

    def generate_response(self, prompt: str) -> str:
        print(f"\n[Gemini Adapter] Using model: {self.model_name}, API Key: {self.api_key[:5]}...")
        print(f"[Gemini Adapter] Sending prompt: '{prompt}'")
        # Simulate API call to Gemini
        if "hello" in prompt.lower() or "hi" in prompt.lower():
            return "Hello! How can I assist you today?"
        elif "sentiment" in prompt.lower():
            return "[Gemini Empathetic Response] I detect a strong emotion in your query. Could you please elaborate?"
        return f"[Gemini Basic Response] Processing '{prompt}' using Gemini."

class LlamaAdapter(AbstractLLM):
    def __init__(self):
        self.api_key = os.getenv("LLAMA_API_KEY", "mock-llama-key") # Often not API key for local/hosted Llama, but placeholder
        self.model_name = os.getenv("LLAMA_MODEL_NAME", "llama2-7b-chat")

    def generate_response(self, prompt: str) -> str:
        print(f"\n[Llama Adapter] Using model: {self.model_name}, API Key: {self.api_key[:5]}...")
        print(f"[Llama Adapter] Sending prompt: '{prompt}'")
        # Simulate API call to Llama
        if "simple" in prompt.lower() or "faq" in prompt.lower():
            return "[Llama FAQ Response] Here's a quick answer to your simple question. For more details, please visit our FAQ page."
        return f"[Llama Basic Response] Llama model is generating a response for: '{prompt}'."

# --- 2. LLM Router/Manager ---
class LLMRouter:
    def __init__(self):
        self.adapters = {
            "gpt": GPTAdapter(),
            "gemini": GeminiAdapter(),
            "llama": LlamaAdapter()
        }

    def route_query(self, query: str) -> AbstractLLM:
        query_lower = query.lower()

        if "price" in query_lower or "cost" in query_lower or "billing" in query_lower:
            print("[LLM Router] Routing to GPTAdapter (cost/billing query).")
            return self.adapters["gpt"]
        elif "hello" in query_lower or "hi" in query_lower or "greeting" in query_lower:
            print("[LLM Router] Routing to GeminiAdapter (greeting/simple social).")
            return self.adapters["gemini"]
        elif "complex" in query_lower or "troubleshoot" in query_lower or "technical" in query_lower:
            print("[LLM Router] Routing to GPTAdapter (complex/technical query).")
            return self.adapters["gpt"]
        elif "simple" in query_lower or "faq" in query_lower or "easy" in query_lower:
            print("[LLM Router] Routing to LlamaAdapter (simple/FAQ query).")
            return self.adapters["llama"]
        elif len(query_lower.split()) < 5: # Short queries for Gemini
            print("[LLM Router] Routing to GeminiAdapter (short query).")
            return self.adapters["gemini"]
        else: # Default to GPT for general queries
            print("[LLM Router] Routing to GPTAdapter (default).")
            return self.adapters["gpt"]

# --- 3. Chatbot Core Logic ---
class SmartChatbot:
    def __init__(self):
        self.llm_router = LLMRouter()
        self.conversation_history = []

    def get_response(self, user_query: str) -> str:
        # Add user query to history
        self.conversation_history.append(f"User: {user_query}")

        # Route query and get response from selected LLM
        selected_llm = self.llm_router.route_query(user_query)
        llm_response = selected_llm.generate_response(user_query)

        # Add LLM response to history
        self.conversation_history.append(f"Bot: {llm_response}")

        return llm_response
    
    def print_history(self):
        print("\n--- Conversation History ---")
        for message in self.conversation_history:
            print(message)
        print("----------------------------")

# --- 4. Configuration Management (handled by python-dotenv and os.getenv) ---
# Example .env file content:
# GPT_API_KEY="your_gpt_api_key_here"
# GEMINI_API_KEY="your_gemini_api_key_here"
# LLAMA_API_KEY="your_llama_api_key_here"

# --- 5. Demo/Testing Interface (CLI) ---
if __name__ == "__main__":
    print("\nWelcome to the Smart Customer Support Chatbot!")
    print("Type 'exit' or 'quit' to end the conversation.")

    chatbot = SmartChatbot()

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        
        response = chatbot.get_response(user_input)
        print(f"Bot: {response}")
        
    chatbot.print_history()
