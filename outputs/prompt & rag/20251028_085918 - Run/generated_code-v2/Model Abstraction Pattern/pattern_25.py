from abc import ABC, abstractmethod
import os
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import ChatPromptTemplate

# --- Configuration (simulating config.py) ---
load_dotenv()

class AbstractLLM(ABC):
    @abstractmethod
    def get_response(self, prompt: str) -> str:
        pass

    @abstractmethod
    def analyze_sentiment(self, text: str) -> str:
        pass

    @abstractmethod
    def summarize_text(self, text: str) -> str:
        pass

class OpenAILLM(AbstractLLM):
    def __init__(self):
        self.model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, api_key=os.getenv("OPENAI_API_KEY"))
        self.sentiment_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a sentiment analysis expert. Determine the sentiment (positive, negative, neutral) of the following text."),
            ("user", "{text}")
        ])
        self.summarization_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a text summarization expert. Summarize the following text concisely."),
            ("user", "{text}")
        ])

    def get_response(self, prompt: str) -> str:
        messages = [HumanMessage(content=prompt)]
        response = self.model.invoke(messages)
        return response.content

    def analyze_sentiment(self, text: str) -> str:
        chain = self.sentiment_prompt | self.model
        response = chain.invoke({"text": text})
        return response.content

    def summarize_text(self, text: str) -> str:
        chain = self.summarization_prompt | self.model
        response = chain.invoke({"text": text})
        return response.content

class GeminiLLM(AbstractLLM):
    def __init__(self):
        self.model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7, google_api_key=os.getenv("GOOGLE_API_KEY"))
        self.sentiment_prompt = ChatPromptTemplate.from_messages([
            ("system", "Analyze the sentiment (positive, negative, neutral) of the given text."),
            ("user", "{text}")
        ])
        self.summarization_prompt = ChatPromptTemplate.from_messages([
            ("system", "Provide a concise summary of the following text."),
            ("user", "{text}")
        ])

    def get_response(self, prompt: str) -> str:
        messages = [HumanMessage(content=prompt)]
        response = self.model.invoke(messages)
        return response.content

    def analyze_sentiment(self, text: str) -> str:
        chain = self.sentiment_prompt | self.model
        response = chain.invoke({"text": text})
        return response.content

    def summarize_text(self, text: str) -> str:
        chain = self.summarization_prompt | self.model
        response = chain.invoke({"text": text})
        return response.content

class LlamaLLM(AbstractLLM):
    def __init__(self, model_path: str = "./llama-2-7b-chat.gguf"):
        # This is a placeholder for demonstration. Requires LlamaCpp installation
        # and a local GGUF model file.
        try:
            self.model = LlamaCpp(model_path=model_path, temperature=0.7, max_tokens=2000)
        except Exception as e:
            print(f"Warning: LlamaCpp initialization failed. Ensure llama-cpp-python is installed and model_path is correct. Error: {e}")
            self.model = None

        self.sentiment_prompt = "Please determine the sentiment (positive, negative, neutral) of the following text: {text}"
        self.summarization_prompt = "Please summarize the following text: {text}"

    def get_response(self, prompt: str) -> str:
        if not self.model: return "LlamaLLM not available."
        return self.model.invoke(prompt)

    def analyze_sentiment(self, text: str) -> str:
        if not self.model: return "LlamaLLM not available."
        return self.model.invoke(self.sentiment_prompt.format(text=text))

    def summarize_text(self, text: str) -> str:
        if not self.model: return "LlamaLLM not available."
        return self.model.invoke(self.summarization_prompt.format(text=text))

class LLMFactory:
    def __init__(self):
        self.providers = {
            "openai": OpenAILLM,
            "gemini": GeminiLLM,
            "llama": LlamaLLM
        }

    def get_llm(self, provider_name: str) -> AbstractLLM:
        provider_name = provider_name.lower()
        if provider_name not in self.providers:
            raise ValueError(f"Unsupported LLM provider: {provider_name}")
        return self.providers[provider_name]()

class CustomerSupportChatbot:
    def __init__(self, llm: AbstractLLM):
        self._llm = llm

    def set_llm(self, llm: AbstractLLM):
        self._llm = llm
        print(f"Chatbot now using LLM: {self._llm.__class__.__name__}")

    def handle_query(self, query: str) -> str:
        return self._llm.get_response(query)

    def process_sentiment_request(self, text: str) -> str:
        return self._llm.analyze_sentiment(text)

    def process_summarization_request(self, text: str) -> str:
        return self._llm.summarize_text(text)

if __name__ == "__main__":
    print("Initializing Dynamic Customer Support Chatbot...")
    llm_factory = LLMFactory()

    # Default LLM provider from environment or fallback
    default_provider = os.getenv("DEFAULT_LLM_PROVIDER", "openai").lower()
    try:
        initial_llm = llm_factory.get_llm(default_provider)
        chatbot = CustomerSupportChatbot(initial_llm)
        print(f"Chatbot initialized with default LLM: {initial_llm.__class__.__name__}")
    except ValueError as e:
        print(f"Error initializing default LLM: {e}. Falling back to OpenAI if available.")
        try:
            initial_llm = llm_factory.get_llm("openai")
            chatbot = CustomerSupportChatbot(initial_llm)
            print(f"Chatbot initialized with fallback LLM: {initial_llm.__class__.__name__}")
        except ValueError as e_fallback:
            print(f"Fallback to OpenAI also failed: {e_fallback}. Exiting.")
            exit(1)

    print("\n--- Chatbot Ready ---\n")
    print("Type 'switch <provider>' to change LLM (e.g., 'switch gemini').")
    print("Type 'sentiment <text>' to analyze sentiment.")
    print("Type 'summarize <text>' to summarize text.")
    print("Type 'exit' to quit.")
    print("\n--- Start Chatting ---\n")

    while True:
        user_input = input(f"You ({chatbot._llm.__class__.__name__}): ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        elif user_input.lower().startswith('switch '):
            try:
                new_provider = user_input.split(' ', 1)[1].strip()
                new_llm = llm_factory.get_llm(new_provider)
                chatbot.set_llm(new_llm)
            except ValueError as e:
                print(f"Error: {e}. Available providers: {', '.join(llm_factory.providers.keys())}")
            except Exception as e:
                print(f"An unexpected error occurred during switch: {e}")
        elif user_input.lower().startswith('sentiment '):
            text_to_analyze = user_input.split(' ', 1)[1].strip()
            if text_to_analyze:
                sentiment = chatbot.process_sentiment_request(text_to_analyze)
                print(f"Sentiment ({chatbot._llm.__class__.__name__}): {sentiment}")
            else:
                print("Please provide text for sentiment analysis.")
        elif user_input.lower().startswith('summarize '):
            text_to_summarize = user_input.split(' ', 1)[1].strip()
            if text_to_summarize:
                summary = chatbot.process_summarization_request(text_to_summarize)
                print(f"Summary ({chatbot._llm.__class__.__name__}): {summary}")
            else:
                print("Please provide text to summarize.")
        else:
            response = chatbot.handle_query(user_input)
            print(f"Bot ({chatbot._llm.__class__.__name__}): {response}")
