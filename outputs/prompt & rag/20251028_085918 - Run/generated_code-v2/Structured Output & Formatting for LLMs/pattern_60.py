from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import json

app = FastAPI()

class ProductFeatures(BaseModel):
    product_name: str
    category: str
    key_features: List[str]
    target_audience: str

class ProductDescriptionOutput(BaseModel):
    product_name: str
    description: str
    seo_keywords: List[str]
    short_summary: str

def generate_llm_response(features: ProductFeatures) -> str:
    prompt = f"""Generate an SEO-optimized product description in JSON format for the following product:
Product Name: {features.product_name}
Category: {features.category}
Key Features: {', '.join(features.key_features)}
Target Audience: {features.target_audience}

Your output must be a JSON object matching this schema:
{{
    "product_name": "string",
    "description": "string",
    "seo_keywords": ["string", "string"],
    "short_summary": "string"
}}
"""

    # Simulate LLM response for demonstration purposes
    simulated_description = f"Discover the ultimate {features.product_name}, a game-changer in the {features.category} category. Perfect for {features.target_audience}."
    simulated_seo_keywords = [f"{features.product_name} {features.category}", f"best {features.category}"] + [f.replace(' ', '-') for f in features.key_features[:2]]
    simulated_short_summary = f"A premium {features.category} for {features.target_audience}."

    mock_response = {
        "product_name": features.product_name,
        "description": simulated_description,
        "seo_keywords": simulated_seo_keywords,
        "short_summary": simulated_short_summary
    }
    return json.dumps(mock_response)

@app.post("/generate-description", response_model=ProductDescriptionOutput)
async def generate_product_description(features: ProductFeatures):
    llm_output_json_string = generate_llm_response(features)
    parsed_output = json.loads(llm_output_json_string)
    return ProductDescriptionOutput(**parsed_output)