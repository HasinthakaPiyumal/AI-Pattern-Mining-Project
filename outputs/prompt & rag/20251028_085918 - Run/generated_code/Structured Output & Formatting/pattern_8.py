import json
import random
from typing import List, Dict, Optional
from pydantic import BaseModel, ValidationError

# 1. ProductInfo Pydantic Model
class ProductInfo(BaseModel):
    product_name: str
    price: float
    category: str
    features: List[str]
    specifications: Dict[str, str]

# 2. Simulate LLM Response (Mock LLM Integration Layer)
def simulate_llm_response(prompt: str, description: str) -> str:
    """
    Simulates an LLM response to demonstrate different scenarios:
    - Valid JSON output
    - Malformed JSON output
    - Completely invalid text output
    """
    if "laptop" in description.lower():
        valid_json = json.dumps({
            "product_name": "SuperFast Laptop X",
            "price": 1299.99,
            "category": "Electronics",
            "features": ["16GB RAM", "512GB SSD", "Intel i7 Processor", "14-inch Full HD Display"],
            "specifications": {
                "Brand": "TechCorp",
                "OS": "Windows 11",
                "Weight": "1.5 kg",
                "Battery Life": "10 hours"
            }
        })
        malformed_json_missing_comma = '{"product_name": "UltraBook Pro", "price": 1499.00 "category": "Electronics"}'
        malformed_json_unclosed_quote = '{"product_name": "Gaming PC", "price": 2000.00, "category": "Computers, "features": []}'
        invalid_text = "I don't understand the request. Please ask a clearer question."
        
        responses = [
            valid_json,
            malformed_json_missing_comma,
            malformed_json_unclosed_quote,
            invalid_text
        ]
        return random.choice(responses)
    elif "coffee maker" in description.lower():
        valid_json = json.dumps({
            "product_name": "AromaBrew Coffee Maker",
            "price": 79.50,
            "category": "Home Appliances",
            "features": ["12-cup capacity", "Programmable timer", "Automatic shut-off"],
            "specifications": {
                "Color": "Black",
                "Material": "Stainless Steel",
                "Power": "900W"
            }
        })
        return valid_json # Always return valid for coffee maker for testing purposes
    else:
        # Default for other descriptions if not specifically handled
        return json.dumps({
            "product_name": "Generic Product",
            "price": 9.99,
            "category": "Miscellaneous",
            "features": [],
            "specifications": {}
        })

# 3. Output Processing Layer - JSON Parsing and Schema Validation
def validate_json_output(json_string: str) -> Optional[ProductInfo]:
    """
    Attempts to parse a JSON string and validate it against the ProductInfo Pydantic model.
    Returns a ProductInfo instance if successful, otherwise None.
    """
    try:
        data = json.loads(json_string)
        product_info = ProductInfo(**data)
        print("Successfully parsed and validated JSON.")
        return product_info
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        print(f"Malformed JSON string: {json_string}")
        return None
    except ValidationError as e:
        print(f"Pydantic Validation Error: {e}")
        print(f"JSON content did not match schema: {json_string}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during validation: {e}")
        return None

# Main function to orchestrate the extraction process
def extract_product_info(description: str) -> Optional[ProductInfo]:
    """
    Extracts structured product information from an unstructured description.
    Orchestrates prompt engineering, LLM call simulation, and post-processing.
    """
    print(f"\nProcessing description: '{description}'")
    # Prompt Engineering Module
    prompt = (
        "Please extract the following product information from the text below "
        "and return it as a JSON object with the following keys: "
        "product_name (string), price (float), category (string), "
        "features (list of strings), and specifications (dictionary of string to string)."
        "Strictly ensure the output is valid JSON and adheres to the specified types."
        f"\n\nProduct Description: {description}"
    )

    # LLM Wrapper/Client (Simulated)
    llm_raw_output = simulate_llm_response(prompt, description)
    print(f"LLM Raw Output: {llm_raw_output}")

    # Output Processing Layer
    structured_info = validate_json_output(llm_raw_output)

    return structured_info


if __name__ == "__main__":
    # Example Usage demonstrating different scenarios

    # Scenario 1: Expected Valid JSON Output (Simulated)
    desc1 = "Discover the new TechCorp SuperFast Laptop X with 16GB RAM, 512GB SSD, and a brilliant 14-inch Full HD display. Powered by an Intel i7, this Windows 11 machine weighs only 1.5 kg and offers 10 hours of battery life. Priced at $1299.99."
    product1 = extract_product_info(desc1)
    if product1:
        print("\n--- Extracted Product 1 (Valid) ---")
        print(product1.model_dump_json(indent=2))
    else:
        print("\n--- Failed to extract Product 1 ---")

    print("\n" + "="*50 + "\n")

    # Scenario 2: Unstructured description that might lead to valid JSON (Coffee Maker always valid in simulation)
    desc2 = "Brew the perfect cup every time with our AromaBrew Coffee Maker. Features a 12-cup capacity, programmable timer, and automatic shut-off. Made of stainless steel, black color, 900W power. Only $79.50."
    product2 = extract_product_info(desc2)
    if product2:
        print("\n--- Extracted Product 2 (Valid - Coffee Maker) ---")
        print(product2.model_dump_json(indent=2))
    else:
        print("\n--- Failed to extract Product 2 ---")

    print("\n" + "="*50 + "\n")

    # Scenario 3: Description for which LLM might return malformed/invalid JSON (Laptop is random)
    desc3 = "Looking for a powerful laptop? This model boasts a 15.6-inch display, 8GB RAM, and a fast processor for all your needs. It's a great deal!"
    product3 = extract_product_info(desc3)
    if product3:
        print("\n--- Extracted Product 3 (Potentially Malformed/Invalid) ---")
        print(product3.model_dump_json(indent=2))
    else:
        print("\n--- Failed to extract Product 3 (Likely Malformed/Invalid) ---")