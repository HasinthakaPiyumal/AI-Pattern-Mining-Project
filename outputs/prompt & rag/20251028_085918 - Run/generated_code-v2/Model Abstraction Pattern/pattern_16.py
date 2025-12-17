from abc import ABC, abstractmethod
import os

# --- llm_abstraction_layer.py ---
class LLMProvider(ABC):
    @abstractmethod
    def generate_description(self, product_details: str) -> str:
        pass

class LLMFactory:
    def get_provider(self, provider_name: str):
        if provider_name == "gpt":
            return GPTProvider()
        elif provider_name == "gemini":
            return GeminiProvider()
        else:
            raise ValueError(f"Unknown LLM provider: {provider_name}")

# --- gpt_provider.py ---
# These imports need to be at the top level for a single file
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

class GPTProvider(LLMProvider):
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY"))

    def generate_description(self, product_details: str) -> str:
        prompt = f"Generate a compelling product description based on the following details: {product_details}\n\nDescription:"
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content.strip()
        except Exception as e:
            return f"Error generating description with GPT: {e}"

# --- gemini_provider.py ---
# These imports need to be at the top level for a single file
from langchain_google_genai import ChatGoogleGenerativeAI

class GeminiProvider(LLMProvider):
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=os.environ.get("GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY"))

    def generate_description(self, product_details: str) -> str:
        prompt = f"Generate a compelling product description based on the following details: {product_details}\n\nDescription:"
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content.strip()
        except Exception as e:
            return f"Error generating description with Gemini: {e}"

# --- product_description_generator.py ---
class ProductDescriptionGenerator:
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider

    def generate_product_description(self, product_details: str) -> str:
        print(f"Using LLM provider: {type(self.llm_provider).__name__}")
        return self.llm_provider.generate_description(product_details)

# --- main.py ---
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_HERE")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY_HERE")

if __name__ == "__main__":
    product_details_1 = "Product: Wireless Bluetooth Earbuds, Features: Noise cancellation, 20-hour battery life, ergonomic design, waterproof. Target Audience: Commuters, fitness enthusiasts."
    product_details_2 = "Product: Organic Green Tea, Features: Hand-picked leaves, rich in antioxidants, calming effect, ethically sourced. Target Audience: Health-conscious individuals, tea connoisseurs."

    llm_factory = LLMFactory()

    print("--- Generating description using GPT provider ---")
    try:
        gpt_provider = llm_factory.get_provider("gpt")
        gpt_generator = ProductDescriptionGenerator(gpt_provider)
        description_gpt = gpt_generator.generate_product_description(product_details_1)
        print(f"GPT Description: {description_gpt}\n")
    except ValueError as e:
        print(f"Could not initialize GPT provider: {e}. Skipping GPT demo.\n")
    except Exception as e:
        print(f"An error occurred with GPT provider: {e}. Skipping GPT demo.\n")

    print("--- Generating description using Gemini provider ---")
    try:
        gemini_provider = llm_factory.get_provider("gemini")
        gemini_generator = ProductDescriptionGenerator(gemini_provider)
        description_gemini = gemini_generator.generate_product_description(product_details_2)
        print(f"Gemini Description: {description_gemini}\n")
    except ValueError as e:
        print(f"Could not initialize Gemini provider: {e}. Skipping Gemini demo.\n")
    except Exception as e:
        print(f"An error occurred with Gemini provider: {e}. Skipping Gemini demo.\n")

    print("\n--- Demonstrating dynamic switching (e.g., for different product categories) ---")
    print("Using Gemini for Product 1 (hypothetically for a category best suited for Gemini)")
    try:
        gemini_provider_for_p1 = llm_factory.get_provider("gemini")
        dynamic_generator_1 = ProductDescriptionGenerator(gemini_provider_for_p1)
        description_dynamic_1 = dynamic_generator_1.generate_product_description(product_details_1)
        print(f"Dynamic (Gemini) Description for Product 1: {description_dynamic_1}\n")
    except Exception as e:
        print(f"An error occurred during dynamic switching with Gemini: {e}.\n")

    print("Using GPT for Product 2 (hypothetically for a category best suited for GPT)")
    try:
        gpt_provider_for_p2 = llm_factory.get_provider("gpt")
        dynamic_generator_2 = ProductDescriptionGenerator(gpt_provider_for_p2)
        description_dynamic_2 = dynamic_generator_2.generate_product_description(product_details_2)
        print(f"Dynamic (GPT) Description for Product 2: {description_dynamic_2}\n")
    except Exception as e:
        print(f"An error occurred during dynamic switching with GPT: {e}.\n")