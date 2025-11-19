import json
from typing import List, Dict
from pydantic import BaseModel, ValidationError

class ProductSchema(BaseModel):
    product_name: str
    category: str
    brand: str
    key_features: List[str]
    technical_specifications: Dict[str, str]
    brief_description: str

def _call_llm_api(prompt: str) -> str:
    # This is a placeholder for actual LLM API call.
    # In a real application, this would interact with an LLM service (e.g., OpenAI, Hugging Face).
    # For demonstration, it returns a hardcoded JSON string.
    print(f"LLM Prompt:\n{prompt}\n---")
    
    # Example of a valid JSON output from an LLM
    if "smartphone" in prompt.lower():
        return json.dumps({
            "product_name": "SuperPhone X",
            "category": "Electronics",
            "brand": "TechCorp",
            "key_features": ["6.1-inch OLED Display", "A15 Bionic Chip", "Dual Camera System", "5G Connectivity"],
            "technical_specifications": {
                "display_size": "6.1 inches",
                "processor": "A15 Bionic",
                "storage_options": "128GB, 256GB, 512GB",
                "camera": "12MP Wide, 12MP Ultra-Wide",
                "battery_life": "Up to 20 hours video playback"
            },
            "brief_description": "The SuperPhone X offers cutting-edge performance and an immersive display, perfect for modern users."
        })
    elif "laptop" in prompt.lower():
        return json.dumps({
            "product_name": "UltraBook Pro",
            "category": "Computers",
            "brand": "NexGen",
            "key_features": ["14-inch Retina Display", "Intel Core i7", "16GB RAM", "512GB SSD"],
            "technical_specifications": {
                "display_size": "14 inches",
                "processor": "Intel Core i7-1185G7",
                "storage": "512GB NVMe SSD",
                "ram": "16GB DDR4",
                "graphics": "Intel Iris Xe Graphics"
            },
            "brief_description": "The UltraBook Pro is a powerful and lightweight laptop designed for professionals on the go."
        })
    else:
        # Example of a potentially invalid or incomplete JSON output for error handling test
        return "{\"product_name\": \"Generic Item\", \"category\": \"Misc\", \"brand\": \"Unknown\", \"key_features\": [\"basic\"], \"brief_description\": \"A generic item.\"}"

def generate_structured_product_data(raw_text: str) -> ProductSchema:
    prompt = f"""Generate a JSON object for the following product information. The JSON should conform to the following schema:
{{
    "product_name": "string",
    "category": "string",
    "brand": "string",
    "key_features": ["string"],
    "technical_specifications": {{
        "spec_name_1": "spec_value_1",
        "spec_name_2": "spec_value_2"
    }},
    "brief_description": "string"
}}

Raw Product Data: {raw_text}

JSON Output:"""

    try:
        llm_output_json_str = _call_llm_api(prompt)
        parsed_data = json.loads(llm_output_json_str)
        product_data = ProductSchema(**parsed_data)
        return product_data
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from LLM: {e}")
        raise ValueError("LLM output was not valid JSON") from e
    except ValidationError as e:
        print(f"Error validating LLM output against ProductSchema: {e}")
        raise ValueError("LLM output did not conform to the expected schema") from e
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise

if __name__ == "__main__":
    print("--- Testing with valid raw text (Smartphone) ---")
    raw_product_info_1 = "A new smartphone from TechCorp with a great screen and camera, supports 5G."
    try:
        structured_data_1 = generate_structured_product_data(raw_product_info_1)
        print("Successfully generated and validated structured data:")
        print(structured_data_1.model_dump_json(indent=2))
    except ValueError as e:
        print(f"Failed to process product data: {e}")

    print("\n--- Testing with valid raw text (Laptop) ---")
    raw_product_info_2 = "NexGen's latest laptop, very powerful and light, ideal for work."
    try:
        structured_data_2 = generate_structured_product_data(raw_product_info_2)
        print("Successfully generated and validated structured data:")
        print(structured_data_2.model_dump_json(indent=2))
    except ValueError as e:
        print(f"Failed to process product data: {e}")

    print("\n--- Testing with raw text leading to potentially incomplete LLM output ---")
    raw_product_info_3 = "Just a random item."
    try:
        structured_data_3 = generate_structured_product_data(raw_product_info_3)
        print("Successfully generated and validated structured data:")
        print(structured_data_3.model_dump_json(indent=2))
    except ValueError as e:
        print(f"Failed to process product data: {e}")

    # To test JSONDecodeError, you could modify _call_llm_api to return invalid JSON
    # or pass a string that cannot be parsed as JSON to json.loads directly.
    # To test ValidationError, you could modify _call_llm_api to return valid JSON 
    # but missing a required field or with an incorrect type. 
    # For this example, the 'Generic Item' path in _call_llm_api returns a valid JSON
    # but it could be easily adjusted to cause a ValidationError by removing a required field.
