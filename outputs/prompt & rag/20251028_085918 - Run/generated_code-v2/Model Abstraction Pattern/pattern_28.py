from abc import ABC, abstractmethod
from typing import Dict
import os

# Mocking external libraries for demonstration
class MockOpenAI:
    def Completion(self, model, prompt, max_tokens):
        class MockCompletion:
            def create(self, model, prompt, max_tokens):
                return type('obj', (object,), {'choices': [type('obj', (object,), {'text': f"[GPT-generated for {model}]: {prompt}."})]})()
        return MockCompletion()

class MockGoogleGenerativeAI:
    def generate_text(self, model, prompt):
        class MockTextGeneration:
            def generate_content(self, prompt):
                return type('obj', (object,), {'text': f"[Gemini-generated for {model}]: {prompt}."})()
        return MockTextGeneration().generate_content(prompt)

class MockTransformersPipeline:
    def __call__(self, prompt, max_new_tokens):
        return [{type('obj', (object,), {'generated_text': f"[Llama-generated]: {prompt}."})}]

class MockLLAMAClient:
    def __init__(self, model_name):
        self.model_name = model_name

    def generate_text(self, prompt):
        # Simulate a network call or model inference
        return f"[Llama-generated via {self.model_name}]: {prompt}."

# config.py
class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "sk-mock_openai_key")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "mock_gemini_key")
    LLAMA_API_KEY: str = os.getenv("LLAMA_API_KEY", "mock_llama_key")
    DEFAULT_LLM_PROVIDER: str = os.getenv("DEFAULT_LLM_PROVIDER", "GPT").upper()

    def __init__(self):
        pass


# llm_abstraction.py
class AbstractLLMService(ABC):
    @abstractmethod
    def generate_text(self, prompt: str) -> str:
        pass

class GPTService(AbstractLLMService):
    def __init__(self, api_key: str):
        self.client = MockOpenAI() # Replace with actual openai.OpenAI()
        self.model = "gpt-3.5-turbo-instruct"

    def generate_text(self, prompt: str) -> str:
        response = self.client.Completion(model=self.model, prompt=prompt, max_tokens=100).create(model=self.model, prompt=prompt, max_tokens=100)
        return response.choices[0].text.strip()

class GeminiService(AbstractLLMService):
    def __init__(self, api_key: str):
        self.client = MockGoogleGenerativeAI() # Replace with actual google.generativeai
        self.model = "gemini-pro"
        # Mocking configure function
        # self.client.configure(api_key=api_key)

    def generate_text(self, prompt: str) -> str:
        response = self.client.generate_text(model=self.model, prompt=prompt)
        return response.text.strip()

class LlamaService(AbstractLLMService):
    def __init__(self, api_key: str = None):
        # For demonstration, we'll use a simple mock or a string representation
        # In a real scenario, this would involve loading a local model with transformers
        # or using an API like Replicate or Hugging Face Inference API.
        self.client = MockLLAMAClient(model_name="llama-2-7b-chat") # Or transformers.pipeline("text-generation", model="meta-llama/Llama-2-7b-chat-hf")

    def generate_text(self, prompt: str) -> str:
        # Mocking the generation process
        return self.client.generate_text(prompt)


# llm_manager.py
class LLMProviderFactory:
    @staticmethod
    def get_llm_service(provider_name: str, settings: Settings) -> AbstractLLMService:
        provider_name = provider_name.upper()
        if provider_name == "GPT":
            return GPTService(settings.OPENAI_API_KEY)
        elif provider_name == "GEMINI":
            return GeminiService(settings.GEMINI_API_KEY)
        elif provider_name == "LLAMA":
            return LlamaService(settings.LLAMA_API_KEY)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider_name}")


# product_description_generator.py
class ProductDescriptionGenerator:
    def __init__(self, llm_service: AbstractLLMService):
        self.llm_service = llm_service

    def generate_product_description(self, product_details: Dict[str, str]) -> str:
        prompt = (
            f"Generate a compelling e-commerce product description for the following product:\n"
            f"Product Name: {product_details.get("name")}\n"
            f"Category: {product_details.get("category")}\n"
            f"Key Features: {product_details.get("features")}\n"
            f"Target Audience: {product_details.get("audience")}\n"
            f"Benefits: {product_details.get("benefits")}\n"
            f"Include a call to action. Keep it under 200 words."
        )
        return self.llm_service.generate_text(prompt)

class ChatbotResponseGenerator:
    def __init__(self, llm_service: AbstractLLMService):
        self.llm_service = llm_service

    def generate_chatbot_response(self, user_query: str, context: str = "") -> str:
        prompt = (
            f"You are a helpful customer support assistant for an e-commerce store.\n"
            f"User Query: {user_query}\n"
            f"Context: {context}\n"
            f"Provide a concise and helpful response."
        )
        return self.llm_service.generate_text(prompt)


# main.py
if __name__ == "__main__":
    # Load settings (mocking .env loading for this single file)
    # For a real application, you'd use from dotenv import load_dotenv; load_dotenv()
    # and from pydantic_settings import BaseSettings, Settings as PydanticSettings

    settings = Settings()

    print(f"\n--- Using Default LLM Provider: {settings.DEFAULT_LLM_PROVIDER} ---")
    try:
        default_llm_service = LLMProviderFactory.get_llm_service(settings.DEFAULT_LLM_PROVIDER, settings)
        product_generator = ProductDescriptionGenerator(default_llm_service)
        chatbot_generator = ChatbotResponseGenerator(default_llm_service)

        product_details = {
            "name": "Smartwatch Pro X",
            "category": "Electronics",
            "features": "Heart rate monitor, GPS, 7-day battery, waterproof, customizable faces",
            "audience": "Fitness enthusiasts, tech-savvy individuals",
            "benefits": "Track your health, navigate easily, stay connected on the go"
        }

        description = product_generator.generate_product_description(product_details)
        print("\nGenerated Product Description:")
        print(description)

        customer_query = "What is the warranty for the Smartwatch Pro X?"
        chatbot_response = chatbot_generator.generate_chatbot_response(customer_query, context="Standard 1-year manufacturer warranty.")
        print("\nGenerated Chatbot Response:")
        print(chatbot_response)

    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Switching to Gemini Provider ---")
    try:
        gemini_llm_service = LLMProviderFactory.get_llm_service("GEMINI", settings)
        product_generator_gemini = ProductDescriptionGenerator(gemini_llm_service)
        chatbot_generator_gemini = ChatbotResponseGenerator(gemini_llm_service)

        description_gemini = product_generator_gemini.generate_product_description(product_details)
        print("\nGenerated Product Description (Gemini):")
        print(description_gemini)

        chatbot_response_gemini = chatbot_generator_gemini.generate_chatbot_response(customer_query, context="Standard 1-year manufacturer warranty.")
        print("\nGenerated Chatbot Response (Gemini):")
        print(chatbot_response_gemini)

    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Switching to Llama Provider ---")
    try:
        llama_llm_service = LLMProviderFactory.get_llm_service("LLAMA", settings)
        product_generator_llama = ProductDescriptionGenerator(llama_llm_service)
        chatbot_generator_llama = ChatbotResponseGenerator(llama_llm_service)

        description_llama = product_generator_llama.generate_product_description(product_details)
        print("\nGenerated Product Description (Llama):")
        print(description_llama)

        chatbot_response_llama = chatbot_generator_llama.generate_chatbot_response(customer_query, context="Standard 1-year manufacturer warranty.")
        print("\nGenerated Chatbot Response (Llama):")
        print(chatbot_response_llama)

    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Attempting Unsupported Provider ---")
    try:
        unsupported_llm_service = LLMProviderFactory.get_llm_service("UNSUPPORTED", settings)
    except ValueError as e:
        print(f"Expected Error: {e}")
