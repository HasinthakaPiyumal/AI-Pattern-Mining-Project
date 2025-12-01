from pydantic import BaseModel, ValidationError
from typing import List, Dict, Any
import json

class ProductDescription(BaseModel):
    product_name: str
    short_description: str
    long_description: str
    features: List[str]
    seo_keywords: List[str]

def _construct_prompt(product_name: str, category: str, key_features: List[str]) -> str:
    features_str = "\n- " + "\n- ".join(key_features) if key_features else ""
    prompt = f"""
Generate a comprehensive e-commerce product description in JSON format for the following product:

Product Name: {product_name}
Category: {category}
Key Features:{features_str}

The output must be a valid JSON object with the following structure:
{{
  "product_name": "[Product Name]",
  "short_description": "[A concise, engaging short description]",
  "long_description": "[A detailed, compelling long description highlighting benefits and use cases]",
  "features": [
    "[Feature 1 description]",
    "[Feature 2 description]"
  ],
  "seo_keywords": [
    "keyword1",
    "keyword2"
  ]
}}
"""
    return prompt

def _mock_llm_client(prompt: str) -> str:
    # This is a mock LLM client. In a real application, you would integrate with an actual LLM service (e.g., OpenAI API).
    # For demonstration, it generates a plausible JSON response based on the input.

    # Simple parsing to extract product name for a more dynamic mock
    product_name_start = prompt.find("Product Name: ") + len("Product Name: ")
    product_name_end = prompt.find("\n", product_name_start)
    product_name = prompt[product_name_start:product_name_end].strip()

    category_start = prompt.find("Category: ") + len("Category: ")
    category_end = prompt.find("\n", category_start)
    category = prompt[category_start:category_end].strip()

    features_start = prompt.find("Key Features:\n")
    mock_features = []
    if features_start != -1:
        features_text = prompt[features_start + len("Key Features:\n"):].split("\n")
        for line in features_text:
            if line.startswith("-"): # Assuming key features are bulleted
                mock_features.append(line[1:].strip())

    mock_response = {
        "product_name": product_name,
        "short_description": f"Elevate your experience with our premium {product_name}. Designed for the modern {category} enthusiast.",
        "long_description": f"Discover the unparalleled quality and innovation of the {product_name}. Crafted with precision and an eye for detail, this {category} product offers exceptional performance and durability. Enjoy a seamless experience thanks to its advanced features including: {' and '.join(mock_features) if mock_features else 'cutting-edge technology'}. Ideal for professionals and hobbyists alike.",
        "features": [f"High-performance {feat}" for feat in mock_features] if mock_features else ["Durable construction", "User-friendly interface"],
        "seo_keywords": [product_name.lower().replace(" ", "-"), category.lower(), "buy-online", "best-price"]
    }
    return json.dumps(mock_response, indent=2)

def _parse_and_validate_output(llm_output: str) -> ProductDescription:
    try:
        data = json.loads(llm_output)
        product_description = ProductDescription.model_validate(data)
        return product_description
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM output is not a valid JSON: {e}\nRaw output: {llm_output}")
    except ValidationError as e:
        raise ValueError(f"LLM output does not match the expected schema: {e}\nRaw output: {llm_output}")

def generate_product_description(
    product_name: str,
    category: str,
    key_features: List[str]
) -> ProductDescription:
    prompt = _construct_prompt(product_name, category, key_features)
    llm_raw_output = _mock_llm_client(prompt)
    validated_description = _parse_and_validate_output(llm_raw_output)
    return validated_description

if __name__ == "__main__":
    # Example Usage:
    product_input = {
        "product_name": "Smartwatch Pro X",
        "category": "Wearable Electronics",
        "key_features": [
            "GPS tracking",
            "Heart rate monitoring",
            "Water resistant (50m)",
            "Long-lasting battery"
        ]
    }

    try:
        generated_desc = generate_product_description(
            product_input["product_name"],
            product_input["category"],
            product_input["key_features"]
        )
        print("Successfully generated and validated product description:")
        print(json.dumps(generated_desc.model_dump(), indent=2))

        # Example of invalid output (to test error handling)
        print("\n--- Testing invalid output (simulated LLM error) ---")
        class MalformedMockLLMClient:
            def __call__(self, prompt: str) -> str:
                return "{'product_name': 'Invalid Watch', 'short_description': 123}" # Invalid JSON and schema

        # Temporarily override the mock client for this test
        original_mock_llm_client = _mock_llm_client
        _mock_llm_client = MalformedMockLLMClient()
        try:
            generate_product_description(
                product_input["product_name"],
                product_input["category"],
                product_input["key_features"]
            )
        except ValueError as e:
            print(f"Caught expected error: {e}")
        finally:
            _mock_llm_client = original_mock_llm_client # Restore original mock

    except Exception as e:
        print(f"An unexpected error occurred: {e}")