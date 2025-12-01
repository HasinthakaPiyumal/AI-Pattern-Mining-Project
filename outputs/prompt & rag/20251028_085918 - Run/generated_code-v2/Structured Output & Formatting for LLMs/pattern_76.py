import json
from pydantic import BaseModel, Field
from typing import List

class ProductDescription(BaseModel):
    product_name: str = Field(..., description="The name of the product.")
    short_description: str = Field(..., description="A concise summary of the product.")
    key_features: List[str] = Field(..., description="A list of key features of the product.")
    benefits: str = Field(..., description="The main benefits a user will get from the product.")
    seo_keywords: str = Field(..., description="Comma-separated SEO keywords for the product.")

def generate_llm_response(prompt: str) -> str:
    # This is a placeholder for a real LLM API call.
    # In a real application, this would interact with an actual LLM.
    # For demonstration, it returns a mocked JSON string.
    print(f"--- LLM Prompt ---\n{prompt}\n---\n")
    mock_response = {
        "product_name": "Super Widget 5000",
        "short_description": "An advanced widget designed for ultimate efficiency and performance.",
        "key_features": [
            "High-speed processing",
            "Durable construction",
            "Energy-efficient design",
            "User-friendly interface"
        ],
        "benefits": "Experience unparalleled productivity and reliability with the Super Widget 5000, making your tasks simpler and faster.",
        "seo_keywords": "widget, super widget, efficiency, performance, productivity, durable"
    }
    return json.dumps(mock_response)

def create_prompt(product_specs: dict) -> str:
    prompt = f"""Generate a detailed e-commerce product description in strict JSON format based on the following specifications. Adhere to the JSON schema provided below. Do not include any additional text or formatting outside of the JSON object.

Product Specifications:
{json.dumps(product_specs, indent=2)}

JSON Schema for Product Description:
{{
  "product_name": "<string>",
  "short_description": "<string>",
  "key_features": [
    "<string>",
    "<string>"
  ],
  "benefits": "<string>",
  "seo_keywords": "<string, string, ...>"
}}

Example Output:
{{
  "product_name": "Example Product Name",
  "short_description": "A brief summary of the example product.",
  "key_features": [
    "Feature one",
    "Feature two"
  ],
  "benefits": "The main benefits of using this example product.",
  "seo_keywords": "keyword1, keyword2, keyword3"
}}

Your output must be ONLY the JSON object.
"""
    return prompt

def generate_product_description(product_specs: dict) -> ProductDescription:
    prompt = create_prompt(product_specs)
    llm_output_str = generate_llm_response(prompt)
    
    try:
        llm_output_data = json.loads(llm_output_str)
        product_description = ProductDescription(**llm_output_data)
        return product_description
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from LLM response: {e}")
        print(f"Raw LLM output: {llm_output_str}")
        raise
    except Exception as e:
        print(f"Error validating product description with Pydantic: {e}")
        print(f"Parsed LLM data: {llm_output_data}")
        raise

if __name__ == "__main__":
    sample_product_specs = {
        "name": "Smart Home Hub",
        "category": "Electronics",
        "brand": "Tech Innovations",
        "features": [
            "Voice Control",
            "Multi-device Compatibility",
            "Energy Monitoring",
            "Secure Data Encryption"
        ],
        "target_audience": "Smart home enthusiasts, tech-savvy users",
        "unique_selling_points": "Seamless integration, intuitive interface, robust security."
    }

    print("Generating product description...")
    try:
        generated_description = generate_product_description(sample_product_specs)
        print("\n--- Generated Product Description (Structured) ---")
        print(generated_description.json(indent=2))

        print("\n--- Accessing specific fields ---")
        print(f"Product Name: {generated_description.product_name}")
        print(f"Short Description: {generated_description.short_description}")
        print(f"Key Features: {', '.join(generated_description.key_features)}")

    except Exception as e:
        print(f"An error occurred during description generation: {e}")