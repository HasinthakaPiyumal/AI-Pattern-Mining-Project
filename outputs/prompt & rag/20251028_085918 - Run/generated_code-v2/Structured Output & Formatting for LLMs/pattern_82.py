from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import json

app = FastAPI()

class ProductInput(BaseModel):
    product_name: str = Field(..., example="Wireless Bluetooth Earbuds")
    key_features: List[str] = Field(..., example=["Noise-cancelling", "Long battery life", "Ergonomic design"])
    target_audience: str = Field(..., example="Commuters, fitness enthusiasts")

class ProductDescriptionOutput(BaseModel):
    product_name: str
    short_description: str
    long_description: str
    key_features: List[str]
    SEO_keywords: List[str]

def _simulate_llm_response(prompt: str) -> str:
    # In a real application, this would be an API call to an actual LLM (e.g., OpenAI, Hugging Face)
    # For demonstration, we return a structured JSON string.
    
    # Extract product name from prompt for a slightly dynamic simulation
    try:
        input_data = json.loads(prompt.split("```json\n")[1].split("\n```")[0])
        product_name = input_data.get("product_name", "Generic Product")
        key_features = input_data.get("key_features", [])
    except:
        product_name = "Generic Product"
        key_features = ["feature1", "feature2"]

    return json.dumps({
        "product_name": product_name,
        "short_description": f"Experience premium audio with our new {product_name}. Perfect for {key_features[0].lower()} and on-the-go.",
        "long_description": f"Dive into an immersive sound experience with the {product_name}. Boasting {', '.join(key_features)} and a sleek, comfortable design, these earbuds are engineered for your active lifestyle. Enjoy crystal-clear calls and powerful bass all day long.",
        "key_features": key_features,
        "SEO_keywords": [product_name.lower().replace(" ", "-"), "wireless-earbuds", "bluetooth-headphones", "premium-audio", "noise-cancellation"]
    })

@app.post("/generate-description", response_model=ProductDescriptionOutput, summary="Generate structured product descriptions")
async def generate_product_description(product_details: ProductInput):
    """
    Generates a structured product description in JSON format based on provided product details.

    - **product_name**: Name of the product (e.g., "Wireless Bluetooth Earbuds")
    - **key_features**: List of main features (e.g., ["Noise-cancelling", "Long battery life"])
    - **target_audience**: Who the product is for (e.g., "Commuters, fitness enthusiasts")
    """
    prompt_template = """Generate a detailed product description for the following product details. The output must be a JSON object strictly adhering to the specified schema.```json
{{
    "product_name": "<product_name>",
    "short_description": "<a concise summary>",
    "long_description": "<a comprehensive description>",
    "key_features": ["<feature1>", "<feature2>", ...],
    "SEO_keywords": ["<keyword1>", "<keyword2>", ...]
}}
```

Product Details:
```json
{{
    "product_name": "{product_details.product_name}",
    "key_features": {product_details.key_features},
    "target_audience": "{product_details.target_audience}"
}}
```

Generated JSON Description:"""

    # For a real LLM, you would send prompt_template to the LLM API
    # and then parse its response.
    llm_response_json_str = _simulate_llm_response(prompt_template.format(product_details=product_details))
    
    try:
        parsed_description = json.loads(llm_response_json_str)
        return ProductDescriptionOutput(**parsed_description)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse LLM response as JSON.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing LLM response: {e}")

