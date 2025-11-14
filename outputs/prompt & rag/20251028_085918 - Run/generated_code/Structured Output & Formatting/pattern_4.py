import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Literal
import instructor
from openai import OpenAI

# Load environment variables
load_dotenv()

# --- 1. Data Model (Pydantic) ---
class ProductReviewSummary(BaseModel):
    product_id: str = Field(..., description="The ID of the product being reviewed.")
    overall_sentiment: Literal["positive", "negative", "neutral"] = Field(..., description="Overall sentiment based on the aggregate of reviews.")
    key_pros: List[str] = Field(..., description="A list of common positive points extracted from reviews.")
    key_cons: List[str] = Field(..., description="A list of common negative points extracted from reviews.")
    common_themes: List[str] = Field(..., description="A list of recurring topics or features mentioned in reviews.")
    summary_text: str = Field(..., description="A brief, overarching textual summary of the reviews.")

# --- 2. LLM Interaction Layer (Instructor & OpenAI) ---
# Configure OpenAI client with instructor for structured output
try:
    client = instructor.patch(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))
except Exception as e:
    raise RuntimeError(f"Failed to initialize OpenAI client. Make sure OPENAI_API_KEY is set: {e}")

# --- 3. Review Summarization Service ---
async def summarize_reviews_service(
    product_id: str,
    reviews: List[str]
) -> ProductReviewSummary:
    if not reviews:
        raise ValueError("No reviews provided for summarization.")

    review_text = "\n".join([f"- {review}" for review in reviews])
    
    prompt = f"""Given the following customer reviews for product ID {product_id}, please provide a structured summary.
    Extract the overall sentiment, key positive points (pros), key negative points (cons), and common themes mentioned.
    Finally, provide a brief overarching textual summary.

    Customer Reviews:
    {review_text}

    Ensure the output strictly adheres to the JSON schema defined for ProductReviewSummary.
    """

    try:
        # Use instructor to force the LLM to return a ProductReviewSummary object
        summary: ProductReviewSummary = await client.chat.completions.create(
            model="gpt-3.5-turbo",  # or "gpt-4" for potentially better results
            response_model=ProductReviewSummary,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes product reviews into a structured JSON format."},
                {"role": "user", "content": prompt}
            ]
        )
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error summarizing reviews: {e}")

# --- 4. API Endpoint (FastAPI) ---
app = FastAPI(
    title="E-commerce Product Review Summarizer",
    description="API for summarizing product reviews into a structured JSON format."
)

class ReviewRequest(BaseModel):
    product_id: str = Field(..., example="PROD123", description="Unique identifier for the product.")
    reviews: List[str] = Field(..., example=["Great product, highly recommend!", "The battery life is terrible."], description="List of customer reviews for the product.")


@app.post("/summarize-reviews", response_model=ProductReviewSummary)
async def summarize_product_reviews(request: ReviewRequest):
    """Summarize a list of customer reviews for a given product into a structured JSON format."""
    try:
        summary = await summarize_reviews_service(request.product_id, request.reviews)
        return summary
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

# To run the application:
# 1. Save the code as main.py
# 2. Create a .env file in the same directory with OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
# 3. Install dependencies: pip install fastapi uvicorn pydantic instructor openai python-dotenv
# 4. Run from your terminal: uvicorn main:app --reload
# 5. Access the API documentation at http://127.0.0.1:8000/docs
