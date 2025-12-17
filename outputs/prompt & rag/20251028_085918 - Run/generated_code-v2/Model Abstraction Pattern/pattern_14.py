from abc import ABC, abstractmethod
from typing import Optional


class AbstractLLMProvider(ABC):
    @abstractmethod
    def generate_description(self, product_details: dict) -> str:
        pass


class GPTProvider(AbstractLLMProvider):
    def generate_description(self, product_details: dict) -> str:
        product_name = product_details.get("name", "")
        product_category = product_details.get("category", "General")
        product_features = product_details.get("features", [])
        
        prompt = f"Generate a compelling product description for a {product_category} item named '{product_name}'. Key features: {', '.join(product_features)}. Focus on benefits and unique selling points."
        
        # Simulate GPT API call
        print(f"[GPTProvider] Sending prompt to GPT: '{prompt[:70]}...'\n")
        mock_response = f"Discover the amazing {product_name}, a top-tier {product_category} designed with {', '.join(product_features)}. Experience unparalleled quality and innovation."
        return mock_response


class GeminiProvider(AbstractLLMProvider):
    def generate_description(self, product_details: dict) -> str:
        product_name = product_details.get("name", "")
        product_category = product_details.get("category", "General")
        product_description = product_details.get("description", "")
        
        prompt = f"Craft a concise and engaging description for '{product_name}' ({product_category}). Highlight its essence: {product_description}."
        
        # Simulate Gemini API call
        print(f"[GeminiProvider] Sending prompt to Gemini: '{prompt[:70]}...'\n")
        mock_response = f"Introducing the {product_name}, a versatile {product_category}. {product_description} It's perfect for enhancing your daily life."
        return mock_response


class LlamaProvider(AbstractLLMProvider):
    def generate_description(self, product_details: dict) -> str:
        product_name = product_details.get("name", "")
        product_usage = product_details.get("usage", "everyday use")
        
        prompt = f"Write a short, impactful description for '{product_name}', ideal for {product_usage}."
        
        # Simulate Llama API call
        print(f"[LlamaProvider] Sending prompt to Llama: '{prompt[:70]}...'\n")
        mock_response = f"The {product_name} is an indispensable item for {product_usage}. Its robust design ensures lasting performance."
        return mock_response


class ProductDescriptionGenerator:
    def __init__(self):
        self._providers = {}
        self._default_provider_name = None
        self._routing_rules = {
            "electronics": "GPT",
            "fashion": "Gemini",
            "books": "Llama"
        }

    def add_provider(self, name: str, provider: AbstractLLMProvider):
        self._providers[name] = provider

    def set_default_provider(self, provider_name: str):
        if provider_name in self._providers:
            self._default_provider_name = provider_name
        else:
            raise ValueError(f"Provider '{provider_name}' not registered.")

    def _dynamic_route(self, product_details: dict) -> Optional[str]:
        category = product_details.get("category", "").lower()
        return self._routing_rules.get(category)

    def generate_description(
        self, product_details: dict, preferred_provider: Optional[str] = None
    ) -> str:
        selected_provider_name = None

        if preferred_provider and preferred_provider in self._providers:
            selected_provider_name = preferred_provider
            print(f"Using preferred provider: {selected_provider_name}")
        else:
            routed_provider = self._dynamic_route(product_details)
            if routed_provider and routed_provider in self._providers:
                selected_provider_name = routed_provider
                print(f"Using dynamically routed provider: {selected_provider_name}")
            elif self._default_provider_name:
                selected_provider_name = self._default_provider_name
                print(f"Using default provider: {selected_provider_name}")
            else:
                raise RuntimeError("No LLM provider available or routable.")

        provider = self._providers[selected_provider_name]
        return provider.generate_description(product_details)


if __name__ == "__main__":
    # Initialize providers
    gpt_provider = GPTProvider()
    gemini_provider = GeminiProvider()
    llama_provider = LlamaProvider()

    # Initialize generator
    generator = ProductDescriptionGenerator()
    generator.add_provider("GPT", gpt_provider)
    generator.add_provider("Gemini", gemini_provider)
    generator.add_provider("Llama", llama_provider)

    # Set a default provider
    generator.set_default_provider("GPT")

    print("--- Generating descriptions with various strategies ---\n")

    # Scenario 1: Dynamic routing for an electronics product
    product_1 = {
        "name": "Smartwatch X",
        "category": "Electronics",
        "features": ["GPS", "Heart Rate Monitor", "Waterproof"],
    }
    print("Product 1 (Electronics - dynamic routing):")
    desc_1 = generator.generate_description(product_1)
    print(f"Generated Description: {desc_1}\n\n")

    # Scenario 2: Dynamic routing for a fashion product
    product_2 = {
        "name": "Summer Dress",
        "category": "Fashion",
        "description": "A light and airy dress perfect for warm weather and casual outings.",
    }
    print("Product 2 (Fashion - dynamic routing):")
    desc_2 = generator.generate_description(product_2)
    print(f"Generated Description: {desc_2}\n\n")

    # Scenario 3: Explicitly specify Llama for a book product
    product_3 = {
        "name": "The Quantum Realm",
        "category": "Books",
        "usage": "advanced physics studies",
    }
    print("Product 3 (Books - preferred provider Llama):")
    desc_3 = generator.generate_description(product_3, preferred_provider="Llama")
    print(f"Generated Description: {desc_3}\n\n")

    # Scenario 4: Product with no specific routing rule, falls back to default
    product_4 = {
        "name": "Organic Coffee Beans",
        "category": "Food & Beverage",
        "features": ["Fair Trade", "Rich Aroma"],
    }
    print("Product 4 (Food & Beverage - default provider GPT):")
    desc_4 = generator.generate_description(product_4)
    print(f"Generated Description: {desc_4}\n\n")

    # Scenario 5: Product with preferred provider explicitly set to GPT
    product_5 = {
        "name": "Wireless Earbuds",
        "category": "Audio",
        "features": ["Noise Cancellation", "Long Battery Life"],
    }
    print("Product 5 (Audio - preferred provider GPT):")
    desc_5 = generator.generate_description(product_5, preferred_provider="GPT")
    print(f"Generated Description: {desc_5}\n\n")