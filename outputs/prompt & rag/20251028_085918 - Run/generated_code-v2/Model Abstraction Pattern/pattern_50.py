import os
import abc
import logging
from dotenv import load_dotenv

# Optional: Try to import specific LLM client libraries, handle if not installed
try:
    import openai
except ImportError:
    openai = None
    print("OpenAI library not found. OpenAI LLM functionality will be disabled.")

try:
    import google.generativeai as genai
except ImportError:
    genai = None
    print("Google Generative AI library not found. Gemini LLM functionality will be disabled.")


load_dotenv() # Load environment variables from .env file

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

class AbstractLLM(abc.ABC):
    @abc.abstractmethod
    def generate_text(self, prompt: str) -> str:
        pass

    @abc.abstractmethod
    def analyze_sentiment(self, text: str) -> str:
        pass

    @abc.abstractmethod
    def summarize_text(self, text: str) -> str:
        pass

class OpenAILLM(AbstractLLM):
    def __init__(self, api_key: str):
        if not openai:
            raise RuntimeError("OpenAI library is not installed or available.")
        if not api_key:
            raise ValueError("OpenAI API key not provided.")
        self.client = openai.OpenAI(api_key=api_key)
        self.model = "gpt-3.5-turbo" # Can be configured

    def generate_text(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except openai.OpenAIError as e:
            logging.error(f"OpenAI API error: {e}")
            return f"Error generating text with OpenAI: {e}"
        except Exception as e:
            logging.error(f"Unexpected error with OpenAI API: {e}")
            return f"Unexpected error: {e}"

    def analyze_sentiment(self, text: str) -> str:
        prompt = f"Analyze the sentiment of the following text and return 'positive', 'negative', or 'neutral': '{text}'"
        return self.generate_text(prompt)

    def summarize_text(self, text: str) -> str:
        prompt = f"Summarize the following text concisely: '{text}'"
        return self.generate_text(prompt)

class GeminiLLM(AbstractLLM):
    def __init__(self, api_key: str):
        if not genai:
            raise RuntimeError("Google Generative AI library is not installed or available.")
        if not api_key:
            raise ValueError("Gemini API key not provided.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-pro") # Can be configured

    def generate_text(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e: # genai.core.exceptions.GoogleAPIError or other specific exceptions
            logging.error(f"Gemini API error: {e}")
            return f"Error generating text with Gemini: {e}"

    def analyze_sentiment(self, text: str) -> str:
        prompt = f"Analyze the sentiment of the following text and return 'positive', 'negative', or 'neutral': '{text}'"
        return self.generate_text(prompt)

    def summarize_text(self, text: str) -> str:
        prompt = f"Summarize the following text concisely: '{text}'"
        return self.generate_text(prompt)

class LLMProviderFactory:
    @staticmethod
    def get_llm(provider: str, openai_api_key: str = None, gemini_api_key: str = None) -> AbstractLLM:
        if provider == "openai":
            if openai_api_key:
                logging.info("Initializing OpenAI LLM.")
                return OpenAILLM(openai_api_key)
            else:
                logging.error("OpenAI API key not found. Cannot initialize OpenAI LLM.")
                raise ValueError("OpenAI API key required for OpenAI provider.")
        elif provider == "gemini":
            if gemini_api_key:
                logging.info("Initializing Gemini LLM.")
                return GeminiLLM(gemini_api_key)
            else:
                logging.error("Gemini API key not found. Cannot initialize Gemini LLM.")
                raise ValueError("Gemini API key required for Gemini provider.")
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

class ChatbotService:
    def __init__(self, llm_instance: AbstractLLM):
        self.llm = llm_instance

    def handle_faq_query(self, query: str) -> str:
        prompt = f"Answer the following customer support FAQ: '{query}'"
        response = self.llm.generate_text(prompt)
        logging.info(f"FAQ Query: '{query}' -> Response: '{response}'")
        return response

    def perform_sentiment_analysis(self, text: str) -> str:
        sentiment = self.llm.analyze_sentiment(text)
        logging.info(f"Sentiment Analysis: '{text}' -> Sentiment: '{sentiment}'")
        return sentiment

    def resolve_complex_query(self, query: str) -> str:
        prompt = f"Provide a detailed solution or explanation for the following complex customer support query: '{query}'"
        response = self.llm.generate_text(prompt)
        logging.info(f"Complex Query: '{query}' -> Response: '{response}'")
        return response

    def summarize_conversation(self, conversation_history: str) -> str:
        summary = self.llm.summarize_text(conversation_history)
        logging.info(f"Conversation Summary: '{conversation_history[:50]}...' -> Summary: '{summary}'")
        return summary

if __name__ == "__main__":
    logging.info(f"Attempting to load LLM provider: {LLM_PROVIDER}")
    try:
        llm_instance = LLMProviderFactory.get_llm(
            provider=LLM_PROVIDER,
            openai_api_key=OPENAI_API_KEY,
            gemini_api_key=GEMINI_API_KEY
        )
        chatbot = ChatbotService(llm_instance)

        print(f"\n--- Using LLM Provider: {LLM_PROVIDER.upper()} ---")

        # Example 1: FAQ Query
        faq_response = chatbot.handle_faq_query("How do I reset my password?")
        print(f"FAQ Response: {faq_response}")

        # Example 2: Sentiment Analysis
        sentiment_result = chatbot.perform_sentiment_analysis("The product arrived broken and I am very unhappy.")
        print(f"Sentiment: {sentiment_result}")

        # Example 3: Complex Query
        complex_response = chatbot.resolve_complex_query("Explain the warranty policy for electronics purchased last year.")
        print(f"Complex Query Response: {complex_response}")

        # Example 4: Conversation Summary
        conversation = "Customer: My internet is not working. Agent: Have you tried restarting your router? Customer: Yes, I did that already. Agent: Let me check your connection status."
        summary_result = chatbot.summarize_conversation(conversation)
        print(f"Conversation Summary: {summary_result}")

    except (ValueError, RuntimeError) as e:
        logging.critical(f"Failed to initialize chatbot: {e}")
        print(f"\nError: {e}")
        print("Please ensure you have set the 'LLM_PROVIDER' environment variable to 'openai' or 'gemini' and provided the corresponding API key (OPENAI_API_KEY or GEMINI_API_KEY) in your .env file or environment.")
    except Exception as e:
        logging.critical(f"An unexpected error occurred: {e}")
        print(f"\nAn unexpected error occurred: {e}")