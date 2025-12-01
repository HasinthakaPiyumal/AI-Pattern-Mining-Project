import json
import gradio as gr
from pydantic import BaseModel, ValidationError
from typing import Dict, Any

# 1. Input Module & Pydantic Model for structured data
class ProductInput(BaseModel):
    product_name: str
    features: str
    benefits: str
    target_audience: str

class ProductOutput(BaseModel):
    description: str
    attributes: Dict[str, str]

# 2. Prompt Engineering Module
def generate_prompt(product_details: ProductInput) -> str:
    prompt = f"""
    Generate a concise and engaging product description and its key attributes in JSON format.

    Product Name: {product_details.product_name}
    Key Features: {product_details.features}
    Benefits: {product_details.benefits}
    Target Audience: {product_details.target_audience}

    The output must be a valid JSON object with two top-level keys: "description" and "attributes".
    The "description" key should contain the full product description as a string.
    The "attributes" key should contain a JSON object where keys are attribute names (e.g., "material", "color", "battery_life") and values are their respective values as strings.
    Include attributes relevant to the product based on its features and benefits.

    Example desired JSON output:
    {{
        "description": "A sleek, high-performance gadget designed for modern users...",
        "attributes": {{
            "color": "Space Gray",
            "material": "Aluminum",
            "battery_life": "10 hours"
        }}
    }}
    """
    return prompt

# 3. LLM Integration Module (Simulated)
def call_llm_api(prompt: str) -> str:
    # In a real application, this would call an actual LLM API (e.g., OpenAI, Hugging Face)
    # For this example, we'll return a hardcoded structured response.

    # Simulate different responses based on prompt content for a more dynamic demo
    if "Smartwatch" in prompt:
        return json.dumps({
            "description": "Introducing the revolutionary SmartConnect Watch, your ultimate companion for a healthier and more connected life. Track your fitness, receive notifications, and make calls directly from your wrist. Its sleek design and long-lasting battery make it perfect for any lifestyle.",
            "attributes": {
                "display_type": "AMOLED",
                "water_resistance": "5 ATM",
                "battery_life": "7 days",
                "connectivity": "Bluetooth 5.2"
            }
        })
    elif "Coffee Maker" in prompt:
        return json.dumps({
            "description": "Start your day right with the AromaBrew Deluxe Coffee Maker. Featuring a programmable timer and a reusable filter, it brews perfect coffee every time. Its compact design fits perfectly in any kitchen, delivering rich, flavorful coffee with minimal effort.",
            "attributes": {
                "capacity": "12 cups",
                "filter_type": "Reusable Mesh",
                "programmable_timer": "Yes",
                "material": "Stainless Steel"
            }
        })
    else:
        return json.dumps({
            "description": "This is a fantastic product that solves all your problems! It's designed with the user in mind, providing unparalleled performance and reliability. Get yours today and experience the difference!",
            "attributes": {
                "key_attribute_1": "Value 1",
                "key_attribute_2": "Value 2"
            }
        })

# 4. Output Parsing Module
def parse_llm_output(llm_response: str) -> ProductOutput:
    try:
        parsed_data = json.loads(llm_response)
        return ProductOutput(**parsed_data)
    except json.JSONDecodeError:
        raise ValueError("LLM response is not valid JSON.")
    except ValidationError as e:
        raise ValueError(f"LLM response does not match expected schema: {e}")

# Main application logic
def generate_product_description(product_name: str, features: str, benefits: str, target_audience: str) -> str:
    try:
        product_input = ProductInput(
            product_name=product_name,
            features=features,
            benefits=benefits,
            target_audience=target_audience
        )

        prompt = generate_prompt(product_input)
        llm_response = call_llm_api(prompt)
        product_output = parse_llm_output(llm_response)

        formatted_output = f"""
**Product Description:**
{product_output.description}

**Product Attributes:**
"""
        for key, value in product_output.attributes.items():
            formatted_output += f"- **{key.replace('_', ' ').title()}:** {value}\n"

        return formatted_output

    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

# Gradio Interface
if __name__ == "__main__":
    iface = gr.Interface(
        fn=generate_product_description,
        inputs=[
            gr.Textbox(label="Product Name", placeholder="e.g., SmartConnect Watch"),
            gr.Textbox(label="Key Features", placeholder="e.g., Heart rate monitoring, GPS, notifications, long battery life"),
            gr.Textbox(label="Benefits", placeholder="e.g., Stay healthy, connected, and organized"),
            gr.Textbox(label="Target Audience", placeholder="e.g., Tech-savvy individuals, fitness enthusiasts")
        ],
        outputs=gr.Markdown("Generated Product Description & Attributes"),
        title="E-commerce Product Description Generator",
        description="Enter product details to generate a structured description and attributes."
    )
    iface.launch()