import json
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional

class Product(BaseModel):
    product_name: str = Field(..., description="The name of the product.")
    category: str = Field(..., description="The category the product belongs to.")
    brand: Optional[str] = Field(None, description="The brand of the product.")
    features: List[str] = Field(..., description="A list of key features.")
    price: float = Field(..., description="The price of the product.")
    sku: str = Field(..., description="The Stock Keeping Unit.")

def simulate_llm_response(prompt: str) -> str:
    # This function simulates an LLM's response based on the prompt.
    # In a real application, this would involve calling an actual LLM API (e.g., OpenAI).
    # For demonstration, we'll return a hardcoded JSON string that matches our schema.
    
    # A very simplified check to simulate different outputs based on input, 
    # though a real LLM would parse the description.
    if "Smartwatch" in prompt:
        return json.dumps({
            "product_name": "XYZ Smartwatch Pro",
            "category": "Wearable Technology",
            "brand": "TechGear",
            "features": ["Heart rate monitor", "GPS", "Waterproof", "Long battery life"],
            "price": 199.99,
            "sku": "TG-SWP-001"
        })
    elif "Laptop" in prompt:
        return json.dumps({
            "product_name": "UltraBook X1",
            "category": "Laptops",
            "brand": "CompCorp",
            "features": ["16GB RAM", "512GB SSD", "13-inch Retina Display", "Intel i7"],
            "price": 1200.50,
            "sku": "CC-UBX-005"
        })
    else:
        # Default or error case for other inputs
        return json.dumps({
            "product_name": "Generic Product",
            "category": "Miscellaneous",
            "brand": "Unknown",
            "features": ["Feature 1", "Feature 2"],
            "price": 9.99,
            "sku": "GEN-PROD-000"
        })

def extract_product_info(description: str) -> Optional[Product]:
    prompt_template = """
    Extract the following information from the product description below and return it as a JSON object with the specified keys and data types:

    - product_name (string): The name of the product.
    - category (string): The category the product belongs to.
    - brand (string, optional): The brand of the product.
    - features (list of strings): A list of key features.
    - price (float): The price of the product.
    - sku (string): The Stock Keeping Unit.

    Product Description: 
    {} 

    JSON Output:
    """
    
    llm_prompt = prompt_template.format(description)
    raw_json_output = simulate_llm_response(llm_prompt)
    
    try:
        parsed_data = json.loads(raw_json_output)
        product = Product(**parsed_data)
        return product
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from LLM: {e}")
        return None
    except ValidationError as e:
        print(f"Validation error for product data: {e.errors()}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

if __name__ == "__main__":
    product_description_1 = "Introducing the new XYZ Smartwatch Pro with advanced heart rate monitoring, built-in GPS for accurate tracking, and a waterproof design. Enjoy up to 7 days of battery life on a single charge. Priced at $199.99. SKU: TG-SWP-001."
    product_description_2 = "Experience seamless performance with the UltraBook X1. Featuring a stunning 13-inch Retina Display, 16GB RAM, 512GB SSD, and powered by an Intel i7 processor. Ideal for professionals. Available for $1200.50. SKU: CC-UBX-005."
    product_description_3 = "A simple pen with blue ink."

    print("\n--- Processing Product Description 1 ---")
    product_data_1 = extract_product_info(product_description_1)
    if product_data_1:
        print(f"Extracted Data: {product_data_1.model_dump_json(indent=2)}")

    print("\n--- Processing Product Description 2 ---")
    product_data_2 = extract_product_info(product_description_2)
    if product_data_2:
        print(f"Extracted Data: {product_data_2.model_dump_json(indent=2)}")

    print("\n--- Processing Product Description 3 (Invalid/Generic) ---")
    product_data_3 = extract_product_info(product_description_3)
    if product_data_3:
        print(f"Extracted Data: {product_data_3.model_dump_json(indent=2)}")