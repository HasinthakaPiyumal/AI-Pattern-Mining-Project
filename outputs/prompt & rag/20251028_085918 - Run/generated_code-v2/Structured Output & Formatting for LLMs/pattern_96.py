from pydantic import BaseModel, HttpUrl
import json
from typing import List, Optional

# 1. Product Data Model (Pydantic)
class ProductInfo(BaseModel):
    name: str
    price: float
    currency: str
    description: str
    features: List[str]
    categories: List[str]
    image_urls: List[HttpUrl]
    availability: str

# 2. Web Scraper (Simulated)
def simulate_web_scraper(product_url: str) -> str:
    return """<html><body>
    <h1>Super Widget Pro</h1>
    <p>Price: $99.99</p>
    <p>Description: This is an amazing widget with many features. It's available now!</p>
    <ul>
        <li>Feature 1: High quality material</li>
        <li>Feature 2: Easy to use</li>
        <li>Feature 3: Durable design</li>
    </ul>
    <span>Category: Electronics, Gadgets</span>
    <img src="https://example.com/widget1.jpg">
    <img src="https://example.com/widget2.jpg">
    </body></html>"""

# 3. Prompt Generator
def generate_prompt(raw_html_content: str, schema: str) -> str:
    return f"""Extract the following product details from the provided HTML content and format the output as a JSON object adhering to the following Pydantic schema:

HTML Content:
{raw_html_content}

Pydantic Schema for JSON output:
{schema}

Ensure the output is a valid JSON object only, with no additional text or formatting outside the JSON.
"""

# 4. AI Model Integration (Simulated)
def simulate_ai_response(prompt: str, raw_data: str) -> str:
    # In a real application, this would call an actual LLM API
    # For this simulation, we return a predefined valid JSON string
    return json.dumps({
        "name": "Super Widget Pro",
        "price": 99.99,
        "currency": "USD",
        "description": "This is an amazing widget with many features. It's available now!",
        "features": ["High quality material", "Easy to use", "Durable design"],
        "categories": ["Electronics", "Gadgets"],
        "image_urls": ["https://example.com/widget1.jpg", "https://example.com/widget2.jpg"],
        "availability": "In Stock"
    })

# 6. Main Application Logic
def main():
    product_url = "https://example.com/product/super-widget-pro"

    # Simulate web scraping
    raw_html = simulate_web_scraper(product_url)
    print("\n--- Simulated Raw HTML Content ---")
    print(raw_html[:100] + "...") # Print a snippet

    # Get the Pydantic schema as a string
    product_schema_json = json.dumps(ProductInfo.model_json_schema(), indent=2)
    print("\n--- Expected JSON Schema (from Pydantic) ---")
    print(product_schema_json)

    # Generate the prompt for the AI
    ai_prompt = generate_prompt(raw_html, product_schema_json)
    print("\n--- Generated AI Prompt (Snippet) ---")
    print(ai_prompt[:300] + "...") # Print a snippet

    # Simulate AI response
    ai_raw_json_output = simulate_ai_response(ai_prompt, raw_html)
    print("\n--- Simulated AI Raw JSON Output ---")
    print(ai_raw_json_output)

    # 5. JSON Parser and Validator (using Pydantic)
    try:
        parsed_product_info = ProductInfo.model_validate_json(ai_raw_json_output)
        print("\n--- Successfully Parsed and Validated Product Info ---")
        print(parsed_product_info.model_dump_json(indent=2))
        print(f"Product Name: {parsed_product_info.name}")
        print(f"Product Price: {parsed_product_info.price} {parsed_product_info.currency}")
        print(f"First Image URL: {parsed_product_info.image_urls[0] if parsed_product_info.image_urls else 'N/A'}")
    except Exception as e:
        print(f"\nError parsing or validating AI output: {e}")

if __name__ == "__main__":
    main()