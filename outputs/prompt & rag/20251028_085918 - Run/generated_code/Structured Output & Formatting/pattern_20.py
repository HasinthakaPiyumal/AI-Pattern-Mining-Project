from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Literal
import instructor
import openai
import os

# --- Pydantic Models ---

class KeyAspect(BaseModel):
    aspect_name: str = Field(..., description="Name of the key aspect identified in the review (e.g., 'delivery', 'product quality', 'price').")
    aspect_sentiment: Literal["positive", "negative", "neutral"] = Field(..., description="Sentiment of this specific aspect.")

class SentimentAnalysisResult(BaseModel):
    overall_sentiment: Literal["positive", "negative", "neutral", "mixed"] = Field(..., description="Overall sentiment of the product review.")
    sentiment_score: float = Field(..., ge=0.0, le=1.0, description="A numerical score representing the confidence or intensity of the overall sentiment (0.0 to 1.0).")
    key_aspects: List[KeyAspect] = Field(..., description="A list of key aspects mentioned in the review and their individual sentiments.")
    summary: str = Field(..., description="A concise summary of the product review.")

class ReviewInput(BaseModel):
    review_text: str = Field(..., min_length=1, description="The product review text to be analyzed.")

# --- FastAPI App Setup ---

app = FastAPI(
    title="E-commerce Product Review Sentiment Analyzer",
    description="API to analyze product reviews and return structured sentiment data."
)

# --- LLM Client Setup ---

# Initialize OpenAI client with instructor patch
# For local development, ensure OPENAI_API_KEY is set in your environment variables.
# If you don't have an OpenAI API key, this will raise an error. 
# For demonstration without an actual API call, you can comment out the instructor.patch line 
# and use a mock function for llm_call_with_structured_output.

try:
    # Patch the OpenAI client to enable structured output with Pydantic models
    client = instructor.patch(openai.OpenAI())
except Exception as e:
    print(f"Warning: Could not initialize OpenAI client with instructor. Error: {e}")
    print("Proceeding with a mock LLM function for demonstration purposes.")
    client = None

async def llm_call_with_structured_output(review_text: str) -> SentimentAnalysisResult:
    if client:
        # Real LLM call with structured output
        try:
            completion = await client.chat.completions.create(
                model="gpt-3.5-turbo",  # Or "gpt-4", "gpt-4o", etc.
                response_model=SentimentAnalysisResult,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant designed to extract sentiment and key aspects from e-commerce product reviews and output them in a structured JSON format. Provide an overall sentiment, a sentiment score, a list of key aspects with their sentiments, and a concise summary."},
                    {"role": "user", "content": f"Analyze the following product review: {review_text}"}
                ]
            )
            return completion
        except Exception as e:
            print(f"Error during LLM call: {e}")
            # Fallback to mock or raise error
            pass

    # Mock LLM response for demonstration or if API key is not available/failed
    print("Using mock LLM response.")
    if "good" in review_text.lower() or "excellent" in review_text.lower():
        sentiment = "positive"
        score = 0.9
    elif "bad" in review_text.lower() or "terrible" in review_text.lower():
        sentiment = "negative"
        score = 0.1
    else:
        sentiment = "neutral"
        score = 0.5
    
    return SentimentAnalysisResult(
        overall_sentiment=sentiment,
        sentiment_score=score,
        key_aspects=[
            KeyAspect(aspect_name="overall experience", aspect_sentiment=sentiment)
        ],
        summary=f"Mock summary: The review seems generally {sentiment}."
    )

# --- API Endpoint ---

@app.post("/analyze-review", response_model=SentimentAnalysisResult)
async def analyze_product_review(review_input: ReviewInput):
    """
    Analyzes a product review and returns structured sentiment information.
    """
    print(f"Received review for analysis: {review_input.review_text[:50]}...")
    structured_result = await llm_call_with_structured_output(review_input.review_text)
    return structured_result

# To run this application:
# 1. Save the code as main.py
# 2. Install dependencies: pip install fastapi uvicorn pydantic instructor openai
# 3. If using OpenAI, set your API key: export OPENAI_API_KEY="your_openai_api_key"
# 4. Run the Uvicorn server: uvicorn main:app --reload
# 5. Access the API at http://127.0.0.1:8000/docs for interactive documentation (Swagger UI).
