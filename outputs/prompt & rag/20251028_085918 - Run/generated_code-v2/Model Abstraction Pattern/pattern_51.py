from abc import ABC, abstractmethod

class LLMAbstractor(ABC):
    @abstractmethod
    def generate_description(self, product_name: str, features: list[str], keywords: list[str]) -> str:
        pass

class GPTAbstractor(LLMAbstractor):
    def generate_description(self, product_name: str, features: list[str], keywords: list[str]) -> str:
        prompt = f"Generate a product description for {product_name} with features: {', '.join(features)} and keywords: {', '.join(keywords)} (GPT style)."
        return f"[GPT Generated Description] {prompt} - This is a fantastic {product_name} with advanced features."

class GeminiAbstractor(LLMAbstractor):
    def generate_description(self, product_name: str, features: list[str], keywords: list[str]) -> str:
        prompt = f"Craft a compelling description for {product_name}, highlighting: {', '.join(features)}. Integrate these terms: {', '.join(keywords)} (Gemini tone)."
        return f"[Gemini Generated Description] {prompt} - Elevate your experience with this {product_name}, designed for excellence."

class LlamaAbstractor(LLMAbstractor):
    def generate_description(self, product_name: str, features: list[str], keywords: list[str]) -> str:
        prompt = f"Write a concise description for {product_name}. Key attributes: {', '.join(features)}. Focus on: {', '.join(keywords)} (Llama format)."
        return f"[Llama Generated Description] {prompt} - Discover the robust {product_name}, built for performance and durability."

class LLMFactory:
    @staticmethod
    def get_abstractor(provider_name: str) -> LLMAbstractor:
        if provider_name == "GPT":
            return GPTAbstractor()
        elif provider_name == "Gemini":
            return GeminiAbstractor()
        elif provider_name == "Llama":
            return LlamaAbstractor()
        else:
            raise ValueError(f"Unknown LLM provider: {provider_name}")

if __name__ == "__main__":
    product_data = {
        "product_name": "Smartwatch Pro",
        "features": ["GPS", "Heart Rate Monitor", "Waterproof", "Long Battery Life"],
        "keywords": ["fitness", "tech", "wearable", "health"]
    }

    print("\n--- Generating with Gemini ---\n")
    gemini_generator = LLMFactory.get_abstractor("Gemini")
    gemini_description = gemini_generator.generate_description(
        product_data["product_name"],
        product_data["features"],
        product_data["keywords"]
    )
    print(gemini_description)

    print("\n--- Generating with GPT ---\n")
    gpt_generator = LLMFactory.get_abstractor("GPT")
    gpt_description = gpt_generator.generate_description(
        product_data["product_name"],
        product_data["features"],
        product_data["keywords"]
    )
    print(gpt_description)

    print("\n--- Generating with Llama ---\n")
    llama_generator = LLMFactory.get_abstractor("Llama")
    llama_description = llama_generator.generate_description(
        product_data["product_name"],
        product_data["features"],
        product_data["keywords"]
    )
    print(llama_description)

    print("\n--- Attempting unknown provider ---\n")
    try:
        unknown_generator = LLMFactory.get_abstractor("Bard")
        print(unknown_generator.generate_description(**product_data))
    except ValueError as e:
        print(f"Error: {e}")