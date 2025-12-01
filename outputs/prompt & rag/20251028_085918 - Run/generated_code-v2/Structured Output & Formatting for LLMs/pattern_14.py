from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from typing import List, Optional
import json

app = FastAPI()

class ProductDescriptionInput(BaseModel):
    product_description: str = Field(..., example="A stylish blue denim jacket, available in S, M, L. Made from 100% cotton. Price: $75.00. Brand: DenimCo.")

class ProductInfoOutput(BaseModel):
    product_name: Optional[str] = None
    price: Optional[float] = None
    currency: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    brand: Optional[str] = None
    available_sizes: Optional[List[str]] = None

def genai_model_predict(prompt: str) -> str:
    # This is a placeholder for a real GenAI model interaction.
    # In a real application, you would integrate with an LLM API (e.g., OpenAI, Hugging Face Transformers).
    # For demonstration, we'll simulate a response based on keywords.
    
    if "blue denim jacket" in prompt.lower():
        return json.dumps({
            "product_name": "Stylish Blue Denim Jacket",
            "price": 75.00,
            "currency": "USD",
            "description": "A stylish blue denim jacket, made from 100% cotton.",
            "category": "Apparel",
            "brand": "DenimCo",
            "available_sizes": ["S", "M", "L"]
        })
    elif "red running shoes" in prompt.lower():
        return json.dumps({
            "product_name": "Red Running Shoes",
            "price": 120.50,
            "currency": "USD",
            "description": "Lightweight red running shoes with good sole support.",
            "category": "Footwear",
            "brand": "SpeedyFeet",
            "available_sizes": ["US 8", "US 9", "US 10", "US 11"]
        })
    else:
        return json.dumps({
            "product_name": None,
            "price": None,
            "currency": None,
            "description": "Could not extract detailed information.",
            "category": None,
            "brand": None,
            "available_sizes": None
        })

@app.post("/extract_product_info", response_model=ProductInfoOutput)
async def extract_product_info(input_data: ProductDescriptionInput):
    product_description = input_data.product_description

    prompt = f"""Extract the following product information from the text below and return it as a JSON object. If a field is not found, use `null`. 
Expected JSON format: 
{{
  "product_name": "string",
  "price": float,
  "currency": "string",
  "description": "string",
  "category": "string",
  "brand": "string",
  "available_sizes": ["string"]
}}
Product Description: {product_description}"""

    try:
        genai_response_str = genai_model_predict(prompt)
        extracted_info = json.loads(genai_response_str)
        return ProductInfoOutput(**extracted_info)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse GenAI model response as JSON.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during extraction: {str(e)}")