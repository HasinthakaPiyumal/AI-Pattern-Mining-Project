import os
from abc import ABC, abstractmethod
from dotenv import load_dotenv
import google.generativeai as genai
import openai
import requests
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AbstractLLM(ABC):
    @abstractmethod
    def generate_response(self, prompt: str, max_tokens: int = 150) -> str:
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        pass

class GeminiLLM(AbstractLLM):
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        logging.info("GeminiLLM initialized")

    def generate_response(self, prompt: str, max_tokens: int = 150) -> str:
        try:
            response = self.model.generate_content(prompt, generation_config=genai.types.GenerationConfig(max_output_tokens=max_tokens))
            return response.text
        except Exception as e:
            logging.error(f"Error with Gemini API: {e}")
            return "Sorry, I'm having trouble connecting to Gemini at the moment."

    def get_model_name(self) -> str:
        return "Google Gemini Pro"

class GPTLLM(AbstractLLM):
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.model_name = "gpt-3.5-turbo"
        logging.info("GPTLLM initialized")

    def generate_response(self, prompt: str, max_tokens: int = 150) -> str:
        try:
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Error with OpenAI API: {e}")
            return "Sorry, I'm having trouble connecting to OpenAI at the moment."

    def get_model_name(self) -> str:
        return self.model_name

class LlamaLLM(AbstractLLM):
    def __init__(self):
        self.api_endpoint = os.getenv("LLAMA_API_ENDPOINT", "https://api.example.com/llama") # Placeholder
        self.api_key = os.getenv("LLAMA_API_KEY", "dummy_key") # Placeholder
        logging.info("LlamaLLM (placeholder) initialized")

    def generate_response(self, prompt: str, max_tokens: int = 150) -> str:
        logging.warning("Using placeholder LlamaLLM. This is not a real Llama API integration.")
        try:
            # Simulate an API call
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            payload = {"prompt": prompt, "max_tokens": max_tokens}
            # In a real scenario, this would call a Llama inference API (e.g., Hugging Face Inference API, local server)
            # response = requests.post(self.api_endpoint, headers=headers, json=payload)
            # response.raise_for_status()
            # return response.json().get("text", "")
            return f"[Llama Placeholder Response for '{prompt[:30]}...'] This is a simulated response from Llama, max_tokens={max_tokens}."
        except Exception as e:
            logging.error(f"Error with Llama placeholder API: {e}")
            return "Sorry, I'm having trouble simulating Llama's response at the moment."

    def get_model_name(self) -> str:
        return "Llama Placeholder Model"

class LLMFactory:
    @staticmethod
    def get_llm(provider_name: str) -> AbstractLLM:
        provider_name = provider_name.lower()
        if provider_name == "gemini":
            return GeminiLLM()
        elif provider_name == "gpt":
            return GPTLLM()
        elif provider_name == "llama":
            return LlamaLLM()
        else:
            raise ValueError(f"Unknown LLM provider: {provider_name}")

class Chatbot:
    def __init__(self, default_provider: str = "gemini"):
        self._current_llm = None
        self.switch_provider(default_provider)
        logging.info(f"Chatbot initialized with default provider: {self._current_llm.get_model_name()}")

    def switch_provider(self, provider_name: str):
        try:
            self._current_llm = LLMFactory.get_llm(provider_name)
            logging.info(f"Switched LLM provider to: {self._current_llm.get_model_name()}")
        except ValueError as e:
            logging.error(f"Failed to switch provider: {e}")
            raise

    def respond(self, query: str, max_tokens: int = 150) -> str:
        logging.info(f"Chatbot received query: '{query[:50]}...' using {self._current_llm.get_model_name()}")
        # Simple query analysis for dynamic switching (example)
        if "price" in query.lower() or "cost" in query.lower() or "faq" in query.lower():
            # For simple queries, might prefer a cheaper model if available, or just stick to current
            # For demonstration, we'll just use the current model
            pass
        elif "troubleshoot" in query.lower() or "complex issue" in query.lower():
            # For complex queries, might prefer a more advanced model
            # Example: Ensure we are using GPT if query is complex, assuming GPT is more advanced
            if self._current_llm.get_model_name() != "gpt-3.5-turbo": # Assuming GPT is for complex
                logging.info("Complex query detected, attempting to switch to GPT.")
                try:
                    self.switch_provider("gpt")
                    logging.info("Switched to GPT for complex query.")
                except ValueError:
                    logging.warning("Could not switch to GPT, using current LLM.")
        
        response = self._current_llm.generate_response(query, max_tokens)
        logging.info(f"Chatbot response generated (from {self._current_llm.get_model_name()}): '{response[:50]}...'")
        return response

    def get_current_provider(self) -> str:
        return self._current_llm.get_model_name()

if __name__ == "__main__":
    # Example .env file content:
    # GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
    # OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
    # LLAMA_API_ENDPOINT="http://your-llama-inference-server.com/generate"
    # LLAMA_API_KEY="YOUR_LLAMA_API_KEY"

    print("Starting Unified Customer Support Chatbot demo...")

    try:
        # Initialize chatbot with a default provider (e.g., Gemini)
        chatbot = Chatbot(default_provider="gemini")
        print(f"Chatbot initialized. Current LLM provider: {chatbot.get_current_provider()}")

        # Scenario 1: Simple FAQ query
        print("\n--- Scenario 1: Simple FAQ (using current provider) ---")
        query1 = "What is your return policy?"
        response1 = chatbot.respond(query1)
        print(f"User: {query1}")
        print(f"Bot ({chatbot.get_current_provider()}): {response1}")

        # Scenario 2: Complex troubleshooting query (should trigger GPT if available)
        print("\n--- Scenario 2: Complex Troubleshooting (attempting GPT) ---")
        query2 = "I am having trouble with my device's connectivity, and I've tried restarting it multiple times. What are the advanced troubleshooting steps?"
        response2 = chatbot.respond(query2)
        print(f"User: {query2}")
        print(f"Bot ({chatbot.get_current_provider()}): {response2}")

        # Scenario 3: Explicitly switch to Llama (placeholder)
        print("\n--- Scenario 3: Explicitly switching to Llama (placeholder) ---")
        try:
            chatbot.switch_provider("llama")
            print(f"Switched LLM provider to: {chatbot.get_current_provider()}")
            query3 = "Can you tell me a short story about a brave knight?"
            response3 = chatbot.respond(query3)
            print(f"User: {query3}")
            print(f"Bot ({chatbot.get_current_provider()}): {response3}")
        except ValueError as e:
            print(f"Could not switch to Llama: {e}")

        # Scenario 4: Switch back to Gemini
        print("\n--- Scenario 4: Switching back to Gemini ---")
        try:
            chatbot.switch_provider("gemini")
            print(f"Switched LLM provider to: {chatbot.get_current_provider()}")
            query4 = "What are your operating hours today?"
            response4 = chatbot.respond(query4)
            print(f"User: {query4}")
            print(f"Bot ({chatbot.get_current_provider()}): {response4}")
        except ValueError as e:
            print(f"Could not switch to Gemini: {e}")

    except Exception as e:
        logging.critical(f"An unhandled error occurred in the main application: {e}")
        print(f"An error occurred: {e}")

    print("\nDemo complete.")
