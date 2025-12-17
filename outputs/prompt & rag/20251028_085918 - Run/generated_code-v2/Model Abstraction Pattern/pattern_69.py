import os
from abc import ABC, abstractmethod
from dotenv import load_dotenv
import openai
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# --- LLM Abstraction Layer ---
class AbstractLLMAdapter(ABC):
    @abstractmethod
    def generate_response(self, prompt: str, model_config: dict) -> str:
        pass

# --- LLM Provider Adapters ---
class GPTAdapter(AbstractLLMAdapter):
    def __init__(self, api_key: str):
        openai.api_key = api_key

    def generate_response(self, prompt: str, model_config: dict) -> str:
        try:
            model = model_config.get("model", "gpt-3.5-turbo")
            response = openai.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=model_config.get("temperature", 0.7)
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error with GPT API: {e}"

class GeminiAdapter(AbstractLLMAdapter):
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)

    def generate_response(self, prompt: str, model_config: dict) -> str:
        try:
            model = model_config.get("model", "gemini-pro")
            client = genai.GenerativeModel(model)
            response = client.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error with Gemini API: {e}"

class LlamaAdapter(AbstractLLMAdapter):
    def generate_response(self, prompt: str, model_config: dict) -> str:
        # This is a placeholder. In a real scenario, you'd integrate with
        # a local Llama model (e.g., via Ollama), Hugging Face Transformers,
        # or a commercial Llama API.
        return f"Llama (Placeholder) would process: '{prompt}' (Model: {model_config.get('model', 'default-llama')})"

# --- LLM Configuration & Manager ---
class LLMManager:
    def __init__(self):
        self.api_keys = {
            "openai": os.getenv("OPENAI_API_KEY"),
            "gemini": os.getenv("GEMINI_API_KEY"),
            # "llama": os.getenv("LLAMA_API_KEY"), # If using a commercial Llama API
        }
        self._adapters = {
            "openai": GPTAdapter,
            "gemini": GeminiAdapter,
            "llama": LlamaAdapter, # Placeholder for now
        }
        self.active_adapter: AbstractLLMAdapter = None
        self.active_provider_name: str = None

    def set_active_provider(self, provider_name: str, model_config: dict = None):
        if provider_name not in self._adapters:
            raise ValueError(f"Unsupported LLM provider: {provider_name}")

        api_key = self.api_keys.get(provider_name)
        if not api_key and provider_name not in ["llama"]:
            print(f"Warning: API key for {provider_name} not found. Some providers might not work without it.")

        # Instantiate the adapter. LlamaAdapter doesn't strictly need an API key for its placeholder impl.
        if provider_name == "llama":
            self.active_adapter = self._adapters[provider_name]()
        else:
            self.active_adapter = self._adapters[provider_name](api_key=api_key)
        self.active_provider_name = provider_name
        print(f"Active LLM provider set to: {self.active_provider_name}")

    def get_active_llm_adapter(self) -> AbstractLLMAdapter:
        if not self.active_adapter:
            raise RuntimeError("No active LLM provider set. Please call set_active_provider first.")
        return self.active_adapter

# --- Knowledge Base ---
class KnowledgeBase:
    def __init__(self):
        self.faqs = {
            "shipping status": "Please provide your order number to check the shipping status.",
            "return policy": "Our return policy allows returns within 30 days of purchase with a valid receipt. Items must be unused and in original packaging.",
            "contact support": "You can reach our support team by emailing support@example.com or calling 1-800-123-4567.",
            "product features": "Could you please specify which product you are interested in?"
        }

    def get_answer(self, query: str) -> str | None:
        query_lower = query.lower()
        for keyword, answer in self.faqs.items():
            if keyword in query_lower:
                return answer
        return None

# --- Human Escalation Module ---
class HumanEscalationModule:
    def escalate_to_human(self, query: str):
        print(f"\n--- Escalating '{query}' to a human agent. Please wait.---")
        # In a real system, this would trigger an alert, create a ticket, etc.
        print("A support agent will contact you shortly.")

# --- Chatbot Core ---
class ChatbotCore:
    def __init__(self, llm_manager: LLMManager, kb: KnowledgeBase, human_escalation: HumanEscalationModule):
        self.llm_manager = llm_manager
        self.kb = kb
        self.human_escalation = human_escalation
        self.conversation_history = []

    def process_query(self, query: str) -> str:
        self.conversation_history.append(f"User: {query}")

        # 1. Check Knowledge Base first
        kb_answer = self.kb.get_answer(query)
        if kb_answer:
            response = f"KB: {kb_answer}"
            self.conversation_history.append(f"Chatbot: {response}")
            return response

        # 2. Basic intent recognition for human escalation
        if any(keyword in query.lower() for keyword in ["speak to human", "talk to agent", "complex issue"]):
            self.human_escalation.escalate_to_human(query)
            response = "I understand this is a complex issue. I've escalated your query to a human agent. They will be with you shortly."
            self.conversation_history.append(f"Chatbot: {response}")
            return response

        # 3. Use LLM if KB doesn't have an answer and no escalation needed
        try:
            llm_adapter = self.llm_manager.get_active_llm_adapter()
            # Simple prompt construction. For complex apps, use RAG or prompt engineering frameworks.
            llm_prompt = f"Based on the following conversation, provide a concise answer to the user's latest query:\n{'\n'.join(self.conversation_history)}\nChatbot: "
            # You could pass specific model configurations here if needed, e.g., for GPT/Gemini
            model_config = {"temperature": 0.7, "model": self.llm_manager.active_provider_name if self.llm_manager.active_provider_name == "gemini" else "gpt-3.5-turbo"}
            response = llm_adapter.generate_response(llm_prompt, model_config)
            self.conversation_history.append(f"Chatbot: {response}")
            return response
        except RuntimeError as e:
            return f"Chatbot Error: {e}. Please ensure an LLM provider is set."
        except Exception as e:
            return f"An unexpected error occurred with the LLM: {e}"

# --- Main Application / UI ---
def main():
    print("Welcome to the Intelligent Customer Support Chatbot!")
    print("You can switch LLM providers using 'switch to [openai/gemini/llama]'.")
    print("Type 'exit' to quit or 'help' for FAQs.")

    llm_manager = LLMManager()
    knowledge_base = KnowledgeBase()
    human_escalation = HumanEscalationModule()
    chatbot = ChatbotCore(llm_manager, knowledge_base, human_escalation)

    # Set a default LLM provider
    try:
        llm_manager.set_active_provider("openai") # Default to OpenAI
    except Exception as e:
        print(f"Could not set default LLM provider (openai): {e}. Trying Gemini...")
        try:
            llm_manager.set_active_provider("gemini") # Fallback to Gemini
        except Exception as e_gemini:
            print(f"Could not set default LLM provider (gemini): {e_gemini}. Using Llama placeholder.")
            llm_manager.set_active_provider("llama")


    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == "exit":
            print("Thank you for using the chatbot. Goodbye!")
            break
        elif user_input.lower() == "help":
            print("\n--- Available FAQ Topics ---")
            for topic in knowledge_base.faqs.keys():
                print(f"- {topic}")
            print("\n--- Chatbot Commands ---")
            print("- 'switch to [openai/gemini/llama]' to change LLM provider")
            print("- 'speak to human' to escalate the issue")
            continue
        elif user_input.lower().startswith("switch to "):
            parts = user_input.lower().split(" ")
            if len(parts) == 3 and parts[0] == "switch" and parts[1] == "to":
                new_provider = parts[2]
                try:
                    llm_manager.set_active_provider(new_provider)
                except ValueError as e:
                    print(f"Chatbot: {e}")
                except Exception as e:
                    print(f"Chatbot: Error switching provider: {e}")
            else:
                print("Chatbot: Invalid 'switch to' command. Use 'switch to [openai/gemini/llama]'.")
            continue

        response = chatbot.process_query(user_input)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    main()
