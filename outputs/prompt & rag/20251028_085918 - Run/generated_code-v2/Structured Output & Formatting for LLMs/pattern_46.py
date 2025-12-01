from pydantic import BaseModel
from typing import List
import json
from fastapi import FastAPI, Depends

class Product(BaseModel):
    name: str
    category: str
    price: float
    features: List[str]
    description: str
    sku: str

class ProductExtractor:
    def _simulate_llm_response(self, prompt: str) -> str:
        simulated_output = {
            "name": "Luxury Smartwatch X1",
            "category": "Electronics",
            "price": 299.99,
            "features": ["Heart Rate Monitor", "GPS", "Waterproof", "Long Battery Life"],
            "description": "The Luxury Smartwatch X1 combines elegant design with advanced technology. Track your fitness, receive notifications, and stay connected on the go.",
            "sku": "SWX1-2023-ABC"
        }
        return json.dumps(simulated_output)

    def extract_info(self, product_description: str) -> Product:
        prompt = (
            "Extract the following information from the product description and return it as a JSON object:\n"
            "- Product Name (name: string)\n"
            "- Category (category: string)\n"
            "- Price (price: float)\n"
            "- Features (features: list of strings)\n"
            "- Description (description: string)\n"
            "- SKU (sku: string)\n\n"
            "Product Description:\n"
            f"'\n{product_description}\n'\n\n"
            "Ensure the output is a valid JSON object adhering to the specified types. Example: "
            "{'name': 'Example Product', 'category': 'Example Category', 'price': 99.99, 'features': ['Feature 1', 'Feature 2'], 'description': 'A short description.', 'sku': 'SKU123'}"
        )

        llm_json_string = self._simulate_llm_response(prompt)
        
        parsed_data = json.loads(llm_json_string)
        
        product_data = Product(**parsed_data)
        
        return product_data

app = FastAPI()

def get_product_extractor():
    return ProductExtractor()

@app.post("/extract-product-info", response_model=Product)
async def extract_product_info_endpoint(
    product_description: str,
    extractor: ProductExtractor = Depends(get_product_extractor)
):
    structured_data = extractor.extract_info(product_description)
    return structured_data