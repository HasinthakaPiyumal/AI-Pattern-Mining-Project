import os
from typing import List
import openai
from pydantic import BaseModel, Field
import instructor

# Ensure the OpenAI API key is set as an environment variable
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# Patch the OpenAI client with instructor
client = instructor.patch(openai.OpenAI())

class ProductDescription(BaseModel):
    product_name: str = Field(..., description="The name of the product.")
    short_description: str = Field(..., description="A concise, engaging summary of the product.")
    features_list: List[str] = Field(..., description="A list of key features of the product.")
    benefits_summary: str = Field(..., description="A summary of the main benefits the product offers to the user.")
    technical_specifications: str = Field(..., description="Any relevant technical details or specifications of the product.")

def generate_product_description(
    product_title: str,
    key_features: List[str]
) -> ProductDescription:
    prompt = f"""You are an expert e-commerce copywriter. Your task is to generate a comprehensive and engaging product description based on the provided product information. Ensure the description is compelling and highlights the key selling points.

Product Title: {product_title}
Key Features: {', '.join(key_features)}

Please generate the product description in JSON format, strictly adhering to the ProductDescription schema provided. Make sure all fields are populated accurately and creatively. Ensure the output is a valid JSON object.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o", # Or gpt-3.5-turbo if preferred
            response_model=ProductDescription,
            messages=[
                {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                {"role": "user", "content": prompt}
            ]
        )
        return response
    except openai.APIError as e:
        print(f"OpenAI API Error: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise

if __name__ == "__main__":
    # Example Usage
    product_title_example = "Smart Home Security Camera"
    key_features_example = [
        "1080p HD video",
        "Two-way audio",
        "Night vision",
        "Motion detection alerts",
        "Cloud storage and local SD card support",
        "Easy DIY installation"
    ]

    print("Generating product description...")
    try:
        description = generate_product_description(product_title_example, key_features_example)
        print("\nGenerated Product Description (JSON):\n")
        print(description.model_dump_json(indent=2))

        print("\nAccessing individual fields:")
        print(f"Product Name: {description.product_name}")
        print(f"Short Description: {description.short_description}")
        print(f"First Feature: {description.features_list[0]}")

    except Exception as e:
        print(f"Failed to generate description: {e}")
    print("\n--- Make sure to set your OPENAI_API_KEY environment variable ---")
