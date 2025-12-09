import os
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain, SequentialChain
from dotenv import load_dotenv

load_dotenv()

class ProductInput(BaseModel):
    product_name: str
    category: str
    material: str
    color: str
    features: str
    target_audience: str

class EcommerceProductOptimizer:
    def __init__(self, openai_api_key: str, model_name: str = "gpt-3.5-turbo"):
        self.llm = ChatOpenAI(openai_api_key=openai_api_key, model_name=model_name, temperature=0.7)

        self.product_description_template = ChatPromptTemplate.from_template(
            "Generate an engaging product description for a {product_name} ({category}) made of {material} in {color}. Key features include {features}. It is targeted at {target_audience}."
        )

        self.keyword_extraction_template = ChatPromptTemplate.from_template(
            "From the following product description, extract 5 main SEO keywords and suggest 3 long-tail keywords. Provide them as a comma-separated list. Description: {initial_description}"
        )

        self.seo_optimization_template = ChatPromptTemplate.from_template(
            "Rewrite the following product description to naturally incorporate these keywords: {keywords}. Ensure the description is compelling and optimized for search engines. Original Description: {initial_description}"
        )

        self.social_media_template = ChatPromptTemplate.from_template(
            "Create a short, engaging social media post (e.g., for Instagram/Facebook) for the following product, using these keywords and relevant hashtags. Product Description: {seo_optimized_description} Keywords: {keywords}"
        )

        self.product_description_chain = LLMChain(llm=self.llm, prompt=self.product_description_template, output_key="initial_description")
        self.keyword_extraction_chain = LLMChain(llm=self.llm, prompt=self.keyword_extraction_template, output_key="keywords")
        self.seo_optimization_chain = LLMChain(llm=self.llm, prompt=self.seo_optimization_template, output_key="seo_optimized_description")
        self.social_media_chain = LLMChain(llm=self.llm, prompt=self.social_media_template, output_key="social_media_post")

        self.overall_chain = SequentialChain(
            chains=[self.product_description_chain, self.keyword_extraction_chain, self.seo_optimization_chain, self.social_media_chain],
            input_variables=["product_name", "category", "material", "color", "features", "target_audience"],
            output_variables=["initial_description", "keywords", "seo_optimized_description", "social_media_post"],
            verbose=True
        )

    def optimize_product(self, product_input: ProductInput):
        return self.overall_chain.invoke(product_input.dict())

if __name__ == "__main__":
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it.")

    optimizer = EcommerceProductOptimizer(openai_api_key=openai_api_key)

    example_product = ProductInput(
        product_name="Smart Watch Pro",
        category="Electronics",
        material="Aluminum alloy and silicone",
        color="Space Gray",
        features="Heart rate monitor, GPS, 7-day battery life, waterproof, customizable watch faces",
        target_audience="Fitness enthusiasts and tech-savvy individuals"
    )

    try:
        optimized_output = optimizer.optimize_product(example_product)
        print("\n--- Optimized Product Output ---")
        print(f"Initial Description: {optimized_output.get("initial_description")}")
        print(f"Keywords: {optimized_output.get("keywords")}")
        print(f"SEO Optimized Description: {optimized_output.get("seo_optimized_description")}")
        print(f"Social Media Post: {optimized_output.get("social_media_post")}")
    except Exception as e:
        print(f"An error occurred: {e}")
