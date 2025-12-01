from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()

app = FastAPI()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ProductRequest(BaseModel):
    product_text: str

class ProductOutput(BaseModel):
    name: str
    description: str
    price: str
    features: list[str]
    customer_reviews_summary: str

@app.post("/extract_product_info", response_model=ProductOutput)
async def extract_product_info(request: ProductRequest):
    if not openai_client.api_key:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured. Please set OPENAI_API_KEY in your .env file.")

    prompt_template = """
    You are an expert e-commerce product information extractor. 
    Extract the following information from the provided product text and format it as a JSON object.

    Product Text:
    {product_text}

    Expected JSON format (strictly adhere to this schema):
    {{
        "name": "<product_name>",
        "description": "<product_description>",
        "price": "<product_price>",
        "features": [
            "<feature_1>",
            "<feature_2>",
            "..."
        ],
        "customer_reviews_summary": "<summary_of_customer_reviews>"
    }}

    Ensure all fields are present and correctly formatted. If a field cannot be found, use "N/A" or an empty list for features.
    """

    formatted_prompt = prompt_template.format(product_text=request.product_text)

    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",  # or "gpt-4" for potentially better results
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a helpful assistant designed to output JSON."}, 
                {"role": "user", "content": formatted_prompt}
            ]
        )
        
        llm_output = response.choices[0].message.content
        parsed_output = json.loads(llm_output)
        
        validated_output = ProductOutput(**parsed_output)
        
        return validated_output

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse JSON response from LLM. It did not return valid JSON.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
