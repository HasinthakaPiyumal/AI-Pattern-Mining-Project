from fastapi import FastAPI
from pydantic import BaseModel
from abc import ABC, abstractmethod
import os
from dotenv import load_dotenv
import openai
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# 1. Pydantic Models for Request and Response
class ProductDetails(BaseModel):
    product_name: str
    category: str
    key_features: list[str]
    target_audience: str
    length: str = "medium" # e.g., "short", "medium", "long"

class DescriptionResponse(BaseModel):
    description: str
    generated_by_llm: str

# 2. LLM Abstract Provider Interface
class LLMProvider(ABC):
    @abstractmethod
    def generate_description(self, prompt: str) -> str:
        pass

# 3. Concrete LLM Providers
class OpenAIGPTProvider(LLMProvider):
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in environment variables.")
        openai.api_key = api_key

    def generate_description(self, prompt: str) -> str:
        response = openai.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates engaging product descriptions."}, 
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

class GoogleGeminiProvider(LLMProvider):
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in environment variables.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-pro"))

    def generate_description(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return response.text

class LlamaPlaceholderProvider(LLMProvider):
    def generate_description(self, prompt: str) -> str:
        # In a real application, this would integrate with a Llama model via HuggingFace Transformers
        # or a local API (e.g., using Ollama or vLLM).
        # For demonstration, we return a simple placeholder.
        print("Using Llama Placeholder. In a real scenario, this would involve a Llama model integration.")
        return f"[Llama Placeholder Description] Based on your request: {prompt[:100]}..."

# 4. LLM Provider Factory/Manager
class LLMProviderFactory:
    def get_provider(self, provider_name: str) -> LLMProvider:
        if provider_name.lower() == "openai":
            return OpenAIGPTProvider()
        elif provider_name.lower() == "gemini":
            return GoogleGeminiProvider()
        elif provider_name.lower() == "llama": # Example for a local or hosted Llama model
            return LlamaPlaceholderProvider()
        else:
            raise ValueError(f"Unknown LLM provider: {provider_name}")

# 5. Product Description Service
class ProductDescriptionService:
    def __init__(self, preferred_llm: str = None):
        self.factory = LLMProviderFactory()
        # Prioritize preferred_llm from parameter, then environment variable, default to openai
        self.active_llm_provider_name = preferred_llm or os.getenv("DEFAULT_LLM_PROVIDER", "openai")

    def generate_product_description(self, product_details: ProductDetails) -> DescriptionResponse:
        llm_provider = self.factory.get_provider(self.active_llm_provider_name)

        prompt = f"Generate a {product_details.length} product description for the following product:\n\n"
        prompt += f"Product Name: {product_details.product_name}\n"
        prompt += f"Category: {product_details.category}\n"
        prompt += f"Key Features: {', '.join(product_details.key_features)}\n"
        prompt += f"Target Audience: {product_details.target_audience}\n\n"
        prompt += f"Make sure the description is engaging, highlights benefits, and is optimized for search engines."

        description = llm_provider.generate_description(prompt)
        return DescriptionResponse(description=description, generated_by_llm=self.active_llm_provider_name)

# 6. FastAPI Application
app = FastAPI(title="Dynamic Product Description Generator")

@app.post("/generate-description", response_model=DescriptionResponse, summary="Generate a product description using a flexible LLM backend")
async def generate_description_endpoint(
    product_details: ProductDetails,
    preferred_llm: str = None # Optional: override default LLM for this request
):
    """
    Generates an engaging and SEO-friendly product description.

    - **product_name**: The name of the product.
    - **category**: The category the product belongs to (e.g., 'Electronics', 'Apparel').
    - **key_features**: A list of essential features of the product.
    - **target_audience**: Who the product is designed for.
    - **length**: Desired length of the description ('short', 'medium', 'long').
    - **preferred_llm**: (Optional) Specify 'openai', 'gemini', or 'llama' to override the default LLM provider for this request.
    """
    service = ProductDescriptionService(preferred_llm=preferred_llm)
    response = service.generate_product_description(product_details)
    return response
