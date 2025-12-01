import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI
import json

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()

# Pydantic model for input validation
class ProductDetails(BaseModel):
    product_name: str = Field(..., example="Men's Leather Wallet")
    keywords: list[str] = Field(..., example=["genuine leather", "slim design", "RFID blocking"])
    target_audience: str = Field("men", example="men")
    description_length: str = Field("medium", example="medium") # short, medium, long

# Pydantic model for structured product description output
class ProductDescriptionOutput(BaseModel):
    product_name: str = Field(..., description="The name of the product.")
    tagline: str = Field(..., description="A catchy tagline for the product.")
    description: str = Field(..., description="A detailed description of the product.")
    features: list[str] = Field(..., description="A list of key features.")
    materials: list[str] = Field(..., description="Materials used in the product.")
    care_instructions: str = Field(..., description="Instructions for product care.")

# Core logic for generating the prompt
def create_prompt(details: ProductDetails) -> str:
    keywords_str = ", ".join(details.keywords)
    prompt = f"""Generate a product description for an e-commerce website in JSON format. The product is a '{details.product_name}'.
Target audience: {details.target_audience}.
Key characteristics/keywords: {keywords_str}.
Desired length: {details.description_length}.

Output should be a JSON object with the following keys:
- 'product_name': The name of the product.
- 'tagline': A catchy tagline for the product.
- 'description': A detailed description of the product.
- 'features': A list of key features.
- 'materials': Materials used in the product.
- 'care_instructions': Instructions for product care.

Ensure the JSON is perfectly valid and follows the specified schema. Do not include any additional text outside the JSON object.
"""
    return prompt

@app.post("/generate-description", response_model=ProductDescriptionOutput)
async def generate_product_description(details: ProductDetails):
    try:
        prompt = create_prompt(details)
        
        chat_completion = client.chat.completions.create(
            model="gpt-3.5-turbo-0125", # Or "gpt-4-turbo-preview" or other suitable model
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": "You are a helpful assistant designed to output JSON."}, # Explicitly tell the model to output JSON
                {"role": "user", "content": prompt}
            ]
        )
        
        llm_output_str = chat_completion.choices[0].message.content
        
        # Parse and validate the LLM output using Pydantic
        product_description_data = json.loads(llm_output_str)
        validated_description = ProductDescriptionOutput(**product_description_data)
        
        return validated_description
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode JSON from LLM: {e}. Raw output: {llm_output_str}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# To run the application:
# 1. Save this code as main.py
# 2. Create a .env file in the same directory with your OpenAI API key: OPENAI_API_KEY="your_openai_api_key_here"
# 3. Install dependencies: pip install fastapi uvicorn pydantic openai python-dotenv
# 4. Run: uvicorn main:app --reload