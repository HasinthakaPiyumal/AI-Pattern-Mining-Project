import json
from enum import Enum
from typing import List, Union, Dict
from pydantic import BaseModel, ValidationError, Field


class OutputFormat(Enum):
    JSON = "json"
    MARKDOWN = "markdown"
    XML = "xml"


class ProductInput(BaseModel):
    name: str = Field(..., description="Name of the product")
    category: str = Field(..., description="Category of the product")
    features: List[str] = Field(..., description="List of key features")


class ProductDescriptionJSON(BaseModel):
    title: str = Field(..., description="Catchy title for the product")
    short_description: str = Field(..., description="A concise summary of the product")
    long_description: str = Field(..., description="A detailed description highlighting benefits and use cases")
    key_features: List[str] = Field(..., description="Bulleted list of main features")
    call_to_action: str = Field(..., description="A clear call to action for the customer")


def _construct_prompt(product_info: ProductInput, output_format: OutputFormat, json_schema: str = None) -> str:
    prompt_parts = [
        f"Generate a product description for an e-commerce platform.",
        f"Product Name: {product_info.name}",
        f"Category: {product_info.category}",
        f"Key Features: {', '.join(product_info.features)}",
        f"Please provide the output in {output_format.value} format."
    ]

    if output_format == OutputFormat.JSON and json_schema:
        prompt_parts.append(f"The JSON output must adhere to the following schema:\n{json_schema}")

    return "\n".join(prompt_parts)


def _simulate_llm_response(prompt: str) -> str:
    # This function simulates an LLM call. In a real application, you'd use
    # an actual LLM client (e.g., openai.Completion.create or gemini.GenerativeModel)
    if "json" in prompt.lower() and "product description for" in prompt.lower():
        product_name = prompt.split("Product Name: ")[1].split("\n")[0]
        features = prompt.split("Key Features: ")[1].split("\n")[0].split(", ")
        return json.dumps({
            "title": f"Unleash the Power of {product_name}!",
            "short_description": f"Experience the next generation with our amazing {product_name}.",
            "long_description": f"Dive deep into the world of {product_name}. Designed for ultimate performance and user satisfaction, it boasts incredible features like {', '.join(features)}. Perfect for {product_name.lower()} enthusiasts.",
            "key_features": features,
            "call_to_action": "Buy now and transform your experience!"
        })
    elif "markdown" in prompt.lower() and "product description for" in prompt.lower():
        product_name = prompt.split("Product Name: ")[1].split("\n")[0]
        features = prompt.split("Key Features: ")[1].split("\n")[0].split(", ")
        features_md = "\n".join([f"- {f}" for f in features])
        return f"# {product_name}\n\n## Overview\nExperience the next generation with our amazing {product_name}. Designed for ultimate performance and user satisfaction.\n\n## Key Features\n{features_md}\n\n## Call to Action\n**Buy now and transform your experience!**"
    elif "xml" in prompt.lower() and "product description for" in prompt.lower():
        product_name = prompt.split("Product Name: ")[1].split("\n")[0]
        features = prompt.split("Key Features: ")[1].split("\n")[0].split(", ")
        features_xml = "".join([f"  <feature>{f}</feature>\n" for f in features])
        return f"<productDescription>\n  <title>Unleash the Power of {product_name}!</title>\n  <shortDescription>Experience the next generation with our amazing {product_name}.</shortDescription>\n  <longDescription>Dive deep into the world of {product_name}. Designed for ultimate performance and user satisfaction, it boasts incredible features like {', '.join(features)}. Perfect for {product_name.lower()} enthusiasts.</longDescription>\n  <keyFeatures>\n{features_xml}  </keyFeatures>\n  <callToAction>Buy now and transform your experience!</callToAction>\n</productDescription>"
    return ""


def generate_product_description(
    product_input: ProductInput,
    output_format: OutputFormat
) -> Union[str, Dict]:
    json_schema = None
    if output_format == OutputFormat.JSON:
        json_schema = json.dumps(ProductDescriptionJSON.model_json_schema(), indent=2)

    prompt = _construct_prompt(product_input, output_format, json_schema)
    llm_raw_response = _simulate_llm_response(prompt)

    if output_format == OutputFormat.JSON:
        try:
            parsed_json = json.loads(llm_raw_response)
            validated_description = ProductDescriptionJSON(**parsed_json)
            return validated_description.model_dump()
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to decode JSON from LLM response: {e}")
        except ValidationError as e:
            raise ValueError(f"LLM response did not match JSON schema: {e}")
    elif output_format == OutputFormat.MARKDOWN or output_format == OutputFormat.XML:
        return llm_raw_response
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


if __name__ == "__main__":
    # Example Usage
    product_data = ProductInput(
        name="Quantum Leap Laptop",
        category="Electronics",
        features=["16-core CPU", "32GB RAM", "1TB NVMe SSD", "15-inch OLED Display"]
    )

    print("\n--- Generating JSON Output ---")
    try:
        json_description = generate_product_description(product_data, OutputFormat.JSON)
        print(json.dumps(json_description, indent=2))
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Generating MARKDOWN Output ---")
    try:
        markdown_description = generate_product_description(product_data, OutputFormat.MARKDOWN)
        print(markdown_description)
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Generating XML Output ---")
    try:
        xml_description = generate_product_description(product_data, OutputFormat.XML)
        print(xml_description)
    except ValueError as e:
        print(f"Error: {e}")