import json

def generate_product_description(product_name: str, category: str, attributes: dict, output_format: str = "json") -> str:
    """
    Generates a structured product description using a simulated GenAI.

    Args:
        product_name: The name of the product.
        category: The category of the product.
        attributes: A dictionary of key-value attributes for the product.
        output_format: The desired output format ("json" or "markdown").

    Returns:
        A string containing the formatted product description.
    """

    # Simulate the prompt sent to a GenAI model
    prompt = f"""Generate a comprehensive product description for a {category} named '{product_name}'.\n"
             f"Key features include: {', '.join([f'{k}: {v}' for k, v in attributes.items()])}.\n"
             f"The description should be detailed and engaging. Output the description in {output_format} format."""

    # Simulate a GenAI model's response. In a real application, this would be an API call
    # or an inference from a local model (e.g., using Hugging Face Transformers).
    simulated_genai_raw_output = {
        "title": product_name,
        "category": category,
        "description": f"Introducing the stunning {product_name}, a revolutionary {category} designed to enhance your daily life. "
                       f"With {attributes.get('material', 'premium materials')} and {attributes.get('special_feature', 'innovative design')}, "
                       f"it offers unparalleled {attributes.get('benefit_1', 'performance')} and {attributes.get('benefit_2', 'durability')}. "
                       f"Perfect for {attributes.get('target_audience', 'everyone')}, it combines elegance with functionality."
    }

    if output_format.lower() == "json":
        return json.dumps(simulated_genai_raw_output, indent=2)
    elif output_format.lower() == "markdown":
        md_output = []
        md_output.append(f"# {simulated_genai_raw_output['title']}")
        md_output.append(f"**Category:** {simulated_genai_raw_output['category']}\n")
        md_output.append(f"## Product Description\n")
        md_output.append(simulated_genai_raw_output['description'])
        md_output.append(f"\n## Key Features")
        for k, v in attributes.items():
            md_output.append(f"- **{k.replace('_', ' ').title()}:** {v}")
        return "\n".join(md_output)
    else:
        return f"Unsupported output format: {output_format}. Please choose 'json' or 'markdown'."

# Example Usage:
if __name__ == "__main__":
    product_info_1 = {
        "product_name": "SmartFit Pro Watch",
        "category": "Wearable Technology",
        "attributes": {
            "color": "Midnight Black",
            "material": "Aerospace-grade Aluminum",
            "water_resistance": "5 ATM",
            "battery_life": "7 days",
            "special_feature": "Heart Rate & Sleep Tracking",
            "benefit_1": "health monitoring",
            "benefit_2": "convenience",
            "target_audience": "fitness enthusiasts and tech-savvy individuals"
        }
    }

    # Generate JSON output
    json_description = generate_product_description(**product_info_1, output_format="json")
    print("--- JSON Output ---")
    print(json_description)
    print("\n" * 2)

    # Generate Markdown output
    markdown_description = generate_product_description(**product_info_1, output_format="markdown")
    print("--- Markdown Output ---")
    print(markdown_description)
    print("\n" * 2)

    product_info_2 = {
        "product_name": "Eco-Friendly Bamboo Toothbrush Set",
        "category": "Personal Care",
        "attributes": {
            "material": "Sustainable Bamboo",
            "bristle_type": "Soft Nylon",
            "pack_size": "4-pack",
            "special_feature": "Biodegradable Handle",
            "benefit_1": "environmental friendliness",
            "benefit_2": "gentle cleaning",
            "target_audience": "eco-conscious consumers"
        }
    }

    json_description_2 = generate_product_description(**product_info_2, output_format="json")
    print("--- JSON Output for Toothbrush ---")
    print(json_description_2)
    print("\n" * 2)

    markdown_description_2 = generate_product_description(**product_info_2, output_format="markdown")
    print("--- Markdown Output for Toothbrush ---")
    print(markdown_description_2)