import json
import re
from pydantic import BaseModel, Field
from typing import Optional

class Product(BaseModel):
    name: str = Field(..., description="Name of the product")
    description: Optional[str] = Field(None, description="Detailed description of the product")
    price: float = Field(..., description="Price of the product")
    category: str = Field(..., description="Category the product belongs to")
    availability: bool = Field(..., description="Is the product currently available?")
    sku: str = Field(..., description="Stock Keeping Unit (SKU) for the product")

def simulate_genai_extraction(prompt: str) -> str:
    listing_match = re.search(r"Product Listing: (.*)\nJSON Output:", prompt, re.DOTALL)
    if not listing_match:
        return json.dumps({"error": "Product Listing not found in prompt"})

    product_listing_content = listing_match.group(1).strip()

    name = "Unknown Product"
    description = None
    price = 0.0
    category = "Uncategorized"
    availability = False
    sku = "N/A"

    name_match = re.search(r"Name: ([^;]+)", product_listing_content)
    if name_match:
        name = name_match.group(1).strip()

    desc_match = re.search(r"Description: ([^;]+)", product_listing_content)
    if desc_match:
        description = desc_match.group(1).strip()

    price_match = re.search(r"Price: \$?([\d.]+)", product_listing_content)
    if price_match:
        try:
            price = float(price_match.group(1))
        except ValueError:
            pass

    category_match = re.search(r"Category: ([^;]+)", product_listing_content)
    if category_match:
        category = category_match.group(1).strip()

    availability_match = re.search(r"Availability: ([^;]+)", product_listing_content)
    if availability_match:
        avail_str = availability_match.group(1).strip().lower()
        availability = "in stock" in avail_str or "available" in avail_str or "true" == avail_str

    sku_match = re.search(r"SKU: ([^;]+)", product_listing_content)
    if sku_match:
        sku = sku_match.group(1).strip()

    extracted_data = {
        "name": name,
        "description": description,
        "price": price,
        "category": category,
        "availability": availability,
        "sku": sku
    }
    return json.dumps(extracted_data)

def craft_extraction_prompt(product_listing_text: str, schema_description: str) -> str:
    prompt = (
        f"Extract the product information from the following listing and return it as a JSON object.\n"
        f"The JSON object must adhere to the following schema:\n"
        f"{schema_description}\n\n"
        f"Product Listing: {product_listing_text}\n"
        f"JSON Output:"
    )
    return prompt

if __name__ == "__main__":
    unstructured_product_listings = [
        "Name: Super-Fast SSD; Description: 1TB NVMe M.2 SSD for ultimate performance; Price: $129.99; Category: Computer Components; Availability: In stock; SKU: SSD-1TM2",
        "Name: Ergonomic Office Chair; Description: Adjustable lumbar support, breathable mesh. Perfect for long hours.; Price: $249.00; Category: Office Furniture; Availability: Out of stock, arriving next week; SKU: OC-ERG-001",
        "Name: Wireless Bluetooth Headset; Description: Noise-cancelling, 20-hour battery life. Great sound quality.; Price: $79.50; Category: Audio Devices; Availability: Limited stock; SKU: BH-WIR-007",
        "Name: Organic Coffee Beans; Description: Medium roast, ethically sourced from Brazil, 1lb bag.; Price: $15.75; Category: Groceries; Availability: Available; SKU: CF-ORG-BRAZIL",
        "Name: Vintage Vinyl Record Player; Description: Classic design with modern connectivity; Price: $199.99; Category: Home Entertainment; Availability: True; SKU: VRP-78RPM"
    ]

    product_schema_description = json.dumps(Product.model_json_schema(), indent=2)

    extracted_products = []

    for i, listing in enumerate(unstructured_product_listings):
        print(f"--- Processing Listing {i+1} ---")
        print(f"Unstructured Input: {listing}")

        prompt = craft_extraction_prompt(listing, product_schema_description)

        genai_json_output = simulate_genai_extraction(prompt)
        print(f"Simulated GenAI JSON Output: {genai_json_output}")

        try:
            product_obj = Product.model_validate_json(genai_json_output)
            extracted_products.append(product_obj)
            print(f"Successfully Extracted and Validated Product:\n{json.dumps(product_obj.model_dump(), indent=2)}")
        except Exception as e:
            print(f"Error parsing or validating GenAI output: {e}")
            print(f"Problematic JSON: {genai_json_output}")
        print("-" * 40)

    print("\nAll Extracted and Validated Products:")
    for product in extracted_products:
        print(json.dumps(product.model_dump(), indent=2))