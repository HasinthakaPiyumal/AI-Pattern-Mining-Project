from abc import ABC, abstractmethod

# abstract_llm.py
class AbstractLLM(ABC):
    @abstractmethod
    def generate_description(self, product_name: str, features: list) -> str:
        pass

# gpt_llm.py
class GPTLLM(AbstractLLM):
    def generate_description(self, product_name: str, features: list) -> str:
        feature_list = ", ".join(features)
        return f"**GPT-powered Description:** Discover the amazing {product_name} with features like {feature_list}. Perfect for modern lifestyles."

# gemini_llm.py
class GeminiLLM(AbstractLLM):
    def generate_description(self, product_name: str, features: list) -> str:
        feature_list = ", ".join(features)
        return f"**Gemini-AI crafted:** Elevate your experience with the {product_name}. Boasting {feature_list}, it's designed for excellence."

# llama_llm.py
class LlamaLLM(AbstractLLM):
    def generate_description(self, product_name: str, features: list) -> str:
        feature_list = ", ".join(features)
        return f"**Llama-generated Product Info:** Introducing the {product_name}. Key highlights include {feature_list}. A smart choice for you."

# product_description_generator.py
class ProductDescriptionGenerator:
    def __init__(self, llm_provider: AbstractLLM):
        self.llm_provider = llm_provider

    def generate(self, product_name: str, features: list) -> str:
        return self.llm_provider.generate_description(product_name, features)

# main.py (demonstration)
if __name__ == "__main__":
    # Product data
    product_name = "Smartwatch Pro"
    product_features = ["heart rate monitor", "GPS tracking", "water resistance", "long battery life"]

    print("--- Demonstrating with GPTLLM ---")
    gpt_llm = GPTLLM()
    gpt_generator = ProductDescriptionGenerator(gpt_llm)
    gpt_description = gpt_generator.generate(product_name, product_features)
    print(gpt_description)
    print("\n")

    print("--- Demonstrating with GeminiLLM ---")
    gemini_llm = GeminiLLM()
    gemini_generator = ProductDescriptionGenerator(gemini_llm)
    gemini_description = gemini_generator.generate(product_name, product_features)
    print(gemini_description)
    print("\n")

    print("--- Demonstrating with LlamaLLM ---")
    llama_llm = LlamaLLM()
    llama_generator = ProductDescriptionGenerator(llama_llm)
    llama_description = llama_generator.generate(product_name, product_features)
    print(llama_description)
    print("\n")

    print("--- Switching LLM dynamically (example: using Gemini then GPT for different products) ---")
    product_name_2 = "Wireless Earbuds X"
    product_features_2 = ["noise cancellation", "bluetooth 5.3", "10-hour playback"]

    # Start with Gemini
    dynamic_generator = ProductDescriptionGenerator(gemini_llm)
    print(f"Product: {product_name_2}")
    print(dynamic_generator.generate(product_name_2, product_features_2))

    # Switch to GPT for another product or scenario
    product_name_3 = "Ergonomic Office Chair"
    product_features_3 = ["lumbar support", "adjustable armrests", "breathable mesh"]
    dynamic_generator.llm_provider = gpt_llm  # Dynamic switching
    print(f"\nProduct: {product_name_3}")
    print(dynamic_generator.generate(product_name_3, product_features_3))
