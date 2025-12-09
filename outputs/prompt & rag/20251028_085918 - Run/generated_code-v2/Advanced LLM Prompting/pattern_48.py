import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class ECommerceDescriptionGenerator:
    def __init__(self, model_name="gpt-3.5-turbo", temperature=0.7):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it.")
        self.llm = ChatOpenAI(model=model_name, temperature=temperature, openai_api_key=api_key)
        self.parser = StrOutputParser()

        # --- Step 1: Product Feature Extraction ---
        self.feature_extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert in extracting key features and selling points from product data."),
            ("human", "Extract key features and selling points from the following product data, presenting them as a concise list or bullet points:\n\n{product_data}")
        ])
        self.feature_extraction_chain = self.feature_extraction_prompt | self.llm | self.parser

        # --- Step 2: Draft Product Description Generation ---
        self.draft_description_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a creative copywriter specializing in engaging product descriptions."),
            ("human", "Using the following product features, generate an engaging and creative initial draft for a product description. Focus on highlighting benefits and appealing to potential customers. Keep it concise, around 100-150 words.\n\n{extracted_features}")
        ])
        self.draft_description_chain = self.draft_description_prompt | self.llm | self.parser

        # --- Step 3: SEO Keyword Integration ---
        self.seo_integration_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an SEO specialist, skilled in naturally integrating keywords into product descriptions without sacrificing readability."),
            ("human", "Refine the following product description by naturally integrating the provided SEO keywords to improve searchability. Ensure the flow and readability remain excellent.\n\nProduct Description: {draft_description}\n\nSEO Keywords: {seo_keywords}")
        ])
        self.seo_integration_chain = self.seo_integration_prompt | self.llm | self.parser

        # --- Step 4: Tone and Style Adjustment/Refinement ---
        self.tone_adjustment_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a professional editor, able to adjust the tone and style of text to meet specific brand guidelines or target audiences."),
            ("human", "Adjust the tone and style of the following product description to be {tone}. Make any final refinements to enhance its appeal to the target audience. The final description should be around 150-200 words.\n\nProduct Description: {seo_description}")
        ])
        self.tone_adjustment_chain = self.tone_adjustment_prompt | self.llm | self.parser

    def generate_optimized_description(self, product_data: str, seo_keywords: list, tone: str) -> str:
        """
        Generates an SEO-optimized and tone-adjusted product description using a prompt chain.

        Args:
            product_data: Raw product information (e.g., name, specs, materials).
            seo_keywords: A list of target SEO keywords.
            tone: The desired tone for the description (e.g., "luxurious", "casual", "technical").

        Returns:
            The final, optimized product description.
        """
        print("\n--- Step 1: Extracting Product Features ---")
        extracted_features = self.feature_extraction_chain.invoke({"product_data": product_data})
        print(f"Extracted Features: {extracted_features}")

        print("\n--- Step 2: Generating Draft Description ---")
        draft_description = self.draft_description_chain.invoke({"extracted_features": extracted_features})
        print(f"Draft Description: {draft_description}")

        print("\n--- Step 3: Integrating SEO Keywords ---")
        seo_description = self.seo_integration_chain.invoke({
            "draft_description": draft_description,
            "seo_keywords": ", ".join(seo_keywords) # Join keywords for the prompt
        })
        print(f"SEO Optimized Description: {seo_description}")

        print("\n--- Step 4: Adjusting Tone and Style ---")
        final_description = self.tone_adjustment_chain.invoke({
            "seo_description": seo_description,
            "tone": tone
        })
        print(f"Final Description (Tone: {tone}): {final_description}")

        return final_description

if __name__ == "__main__":
    # Example Usage:
    generator = ECommerceDescriptionGenerator()

    # Example 1: Wireless Earbuds
    product_data_1 = "Product Name: AuraPods Pro, Specifications: Bluetooth 5.3, 40-hour battery life with case, Active Noise Cancellation, IPX7 waterproof, Touch Controls, Ergonomic Fit. Material: Premium Matte Plastic, Soft Silicone Eartips. Features: Crystal-clear audio, quick charging, seamless device switching."
    seo_keywords_1 = ["wireless earbuds", "noise cancelling headphones", "long battery life earbuds", "waterproof headphones", "bluetooth 5.3 audio"]
    tone_1 = "luxurious and high-tech"

    print("\n===============================================")
    print("Generating description for AuraPods Pro...")
    final_desc_1 = generator.generate_optimized_description(product_data_1, seo_keywords_1, tone_1)
    print("\n--- GENERATION COMPLETE (AuraPods Pro) ---")
    print(final_desc_1)
    print("===============================================")

    # Example 2: Organic Coffee Beans
    product_data_2 = "Product Name: Morning Bliss Organic Coffee, Type: Whole Bean, Roast: Medium-Dark, Origin: Ethiopian Yirgacheffe, Flavor Notes: Berry, Floral, Citrus. Certifications: USDA Organic, Fair Trade. Bag Size: 12 oz. Features: Single-origin, ethically sourced, vibrant aroma."
    seo_keywords_2 = ["organic coffee beans", "ethiopian yirgacheffe", "fair trade coffee", "medium dark roast", "gourmet whole bean"]
    tone_2 = "friendly and artisanal"

    print("\n===============================================")
    print("Generating description for Morning Bliss Organic Coffee...")
    final_desc_2 = generator.generate_optimized_description(product_data_2, seo_keywords_2, tone_2)
    print("\n--- GENERATION COMPLETE (Morning Bliss Organic Coffee) ---")
    print(final_desc_2)
    print("===============================================")
