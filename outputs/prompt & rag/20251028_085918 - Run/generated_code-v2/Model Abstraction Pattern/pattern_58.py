import abc
import os

# Configuration Management (Mocking .env loading)
class Config:
    GPT4_API_KEY = os.getenv("GPT4_API_KEY", "mock_gpt4_key")
    GEMINI_PRO_API_KEY = os.getenv("GEMINI_PRO_API_KEY", "mock_gemini_pro_key")
    LLAMA2_API_KEY = os.getenv("LLAMA2_API_KEY", "mock_llama2_key")

# 1. Model Abstraction Layer
class LLMProvider(abc.ABC):
    @abc.abstractmethod
    def generate_product_description(self, product_info: dict) -> str:
        pass

    @abc.abstractmethod
    def generate_seo_keywords(self, description: str) -> list:
        pass

    @abc.abstractmethod
    def generate_social_media_caption(self, description: str) -> str:
        pass

class GPT4Provider(LLMProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def generate_product_description(self, product_info: dict) -> str:
        return f"GPT-4 Description: A high-quality description for {product_info.get('name', 'product')} with features like {', '.join(product_info.get('features', []))}."

    def generate_seo_keywords(self, description: str) -> list:
        return [f"gpt4-{keyword}" for keyword in description.split()[:3]]

    def generate_social_media_caption(self, description: str) -> str:
        return f"#GPT4_Exclusive! Get your {description.split(' ')[-1]} now!"

class GeminiProProvider(LLMProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def generate_product_description(self, product_info: dict) -> str:
        return f"Gemini Pro Description: Concise summary for {product_info.get('name', 'item')}. It's {product_info.get('category', 'unknown category')} and has {len(product_info.get('features', []))} key features."

    def generate_seo_keywords(self, description: str) -> list:
        return [f"gemini-{keyword}" for keyword in description.split()[:2]]

    def generate_social_media_caption(self, description: str) -> str:
        return f"Fast facts: {description.split(' ')[0]} {description.split(' ')[1]}! #GeminiPro"

class Llama2Provider(LLMProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def generate_product_description(self, product_info: dict) -> str:
        return f"Llama 2 Description: Cost-effective text about {product_info.get('name', 'product')}. Suitable for bulk generation. Category: {product_info.get('category', 'N/A')}."

    def generate_seo_keywords(self, description: str) -> list:
        return [f"llama2-{keyword}" for keyword in description.split()[:4]]

    def generate_social_media_caption(self, description: str) -> str:
        return f"New product alert from Llama 2! Check it out! {description.split(' ')[2]} {description.split(' ')[3]}."

class LLMService:
    def __init__(self, api_keys: dict):
        self._providers = {
            "GPT4": GPT4Provider(api_keys.get("GPT4_API_KEY", "")), # Mock key
            "GeminiPro": GeminiProProvider(api_keys.get("GEMINI_PRO_API_KEY", "")), # Mock key
            "Llama2": Llama2Provider(api_keys.get("LLAMA2_API_KEY", "")) # Mock key
        }
        self.current_provider: LLMProvider = None
        self.set_provider("GPT4") # Set a default provider

    def set_provider(self, provider_name: str):
        if provider_name in self._providers:
            self.current_provider = self._providers[provider_name]
            print(f"Switched LLM provider to: {provider_name}")
        else:
            raise ValueError(f"Unknown LLM provider: {provider_name}")

    def generate_product_description(self, product_info: dict) -> str:
        return self.current_provider.generate_product_description(product_info)

    def generate_seo_keywords(self, description: str) -> list:
        return self.current_provider.generate_seo_keywords(description)

    def generate_social_media_caption(self, description: str) -> str:
        return self.current_provider.generate_social_media_caption(description)

# 2. Product Description Generator
class ProductDescriptionGenerator:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def create_full_description(self, product_data: dict) -> str:
        return self.llm_service.generate_product_description(product_data)

    def generate_keywords(self, description: str) -> list:
        return self.llm_service.generate_seo_keywords(description)

    def generate_captions(self, description: str) -> str:
        return self.llm_service.generate_social_media_caption(description)

# 4. Main Application Logic / Entry Point
if __name__ == "__main__":
    # Load API keys (mocked for this example, in a real app use python-dotenv)
    api_keys = {
        "GPT4_API_KEY": Config.GPT4_API_KEY,
        "GEMINI_PRO_API_KEY": Config.GEMINI_PRO_API_KEY,
        "LLAMA2_API_KEY": Config.LLAMA2_API_KEY,
    }

    # Initialize LLMService with API keys
    llm_service = LLMService(api_keys)

    # Initialize ProductDescriptionGenerator
    product_generator = ProductDescriptionGenerator(llm_service)

    # Sample product data
    sample_product_data = {
        "name": "Smartwatch Pro",
        "features": ["GPS", "Heart Rate Monitor", "Waterproof"],
        "category": "Wearable Tech",
        "price": "$299.99"
    }

    print("\n--- Using GPT-4 Provider ---")
    full_description = product_generator.create_full_description(sample_product_data)
    print(f"Full Description: {full_description}")
    seo_keywords = product_generator.generate_keywords(full_description)
    print(f"SEO Keywords: {seo_keywords}")
    social_caption = product_generator.generate_captions(full_description)
    print(f"Social Media Caption: {social_caption}")

    print("\n--- Switching to Gemini Pro Provider ---")
    llm_service.set_provider("GeminiPro")
    full_description = product_generator.create_full_description(sample_product_data)
    print(f"Full Description: {full_description}")
    seo_keywords = product_generator.generate_keywords(full_description)
    print(f"SEO Keywords: {seo_keywords}")
    social_caption = product_generator.generate_captions(full_description)
    print(f"Social Media Caption: {social_caption}")

    print("\n--- Switching to Llama 2 Provider ---")
    llm_service.set_provider("Llama2")
    full_description = product_generator.create_full_description(sample_product_data)
    print(f"Full Description: {full_description}")
    seo_keywords = product_generator.generate_keywords(full_description)
    print(f"SEO Keywords: {seo_keywords}")
    social_caption = product_generator.generate_captions(full_description)
    print(f"Social Media Caption: {social_caption}")

    print("\n--- Attempting to switch to an unknown provider ---")
    try:
        llm_service.set_provider("UnknownLLM")
    except ValueError as e:
        print(f"Error: {e}")