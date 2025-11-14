from enum import Enum
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains.structured_output import create_structured_output_runnable

# --- 1. models.py content ---

class Sentiment(str, Enum):
    positive = "positive"
    negative = "negative"
    neutral = "neutral"

class ReviewInput(BaseModel):
    review_text: str = Field(..., description="The raw customer review text.")

class StructuredReviewOutput(BaseModel):
    sentiment: Sentiment = Field(..., description="Overall sentiment of the review.")
    key_features: List[str] = Field(..., description="List of key product features mentioned in the review.")
    common_complaints: List[str] = Field(..., description="List of common complaints or issues raised in the review.")
    star_rating: int = Field(..., ge=1, le=5, description="Star rating given in the review (1 to 5).")

# --- 2. review_analyzer.py content ---

class ReviewAnalyzer:
    def __init__(self, openai_api_key: Optional[str] = None):
        # Initialize the LLM. For local models, replace ChatOpenAI with HuggingFacePipeline
        # Example for local model (requires transformers and a suitable model):
        # from langchain_community.llms import HuggingFacePipeline
        # from transformers import pipeline
        # self.llm = HuggingFacePipeline(pipeline=pipeline("text-generation", model="distilgpt2", device=0))
        
        if openai_api_key:
            self.llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125", openai_api_key=openai_api_key)
        else:
            # Fallback or raise error if API key not provided for OpenAI
            raise ValueError("OpenAI API key must be provided.")

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are an expert e-commerce product review analyzer. Extract structured information from the provided customer review."),
                ("human", "Analyze the following product review and extract the sentiment, key features, common complaints, and star rating. Format the output as a JSON object strictly following the provided schema.\n\nReview: {review_text}")
            ]
        )

        # Create a runnable that enforces structured output using the Pydantic model
        self.structured_extractor = create_structured_output_runnable(
            output_schema=StructuredReviewOutput,
            llm=self.llm,
            prompt=self.prompt,
        )

    async def analyze_review(self, review_text: str) -> StructuredReviewOutput:
        try:
            # Invoke the runnable with the review text
            structured_output = await self.structured_extractor.ainvoke({"review_text": review_text})
            return structured_output
        except Exception as e:
            print(f"Error analyzing review: {e}")
            # In a real application, you might want to return a specific error structure
            raise

# --- 3. main.py content ---

app = FastAPI(
    title="E-commerce Product Review Analyzer",
    description="API to analyze customer reviews and extract structured information."
)

# Initialize the analyzer. For a production environment, use environment variables for API keys.
# Example: os.getenv("OPENAI_API_KEY")
# Make sure to set your OPENAI_API_KEY environment variable or pass it directly.
# For demonstration, a placeholder is used.
# review_analyzer = ReviewAnalyzer(openai_api_key="YOUR_OPENAI_API_KEY") # Replace with your actual key

# For simplicity in a single-file example, let's assume the API key is set in environment.
# In a real app, you'd load from .env or config.
import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 

if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY environment variable not set. Please set it to run the application.")
    print("You can run: export OPENAI_API_KEY='your_api_key_here'")
    # Optionally, raise an error or exit if the key is critical
    # raise ValueError("OPENAI_API_KEY environment variable is not set.")

# Initialize ReviewAnalyzer only if API key is potentially available
review_analyzer = None
if OPENAI_API_KEY:
    try:
        review_analyzer = ReviewAnalyzer(openai_api_key=OPENAI_API_KEY)
    except ValueError as e:
        print(f"Error initializing ReviewAnalyzer: {e}")


@app.post("/analyze-review", response_model=StructuredReviewOutput)
async def analyze_product_review(review_input: ReviewInput):
    """
    Analyzes a customer product review and extracts structured data.
    """
    if not review_analyzer:
        raise HTTPException(status_code=500, detail="Review analyzer not initialized. Check API key configuration.")
    
    structured_output = await review_analyzer.analyze_review(review_input.review_text)
    return structured_output

# To run this application:
# 1. Save the code as ecommerce_review_analyzer.py
# 2. Install necessary libraries: pip install fastapi uvicorn pydantic "langchain_openai>=0.1.0" "langchain>=0.1.0"
# 3. Set your OpenAI API key as an environment variable: export OPENAI_API_KEY='YOUR_API_KEY_HERE'
# 4. Run the application: uvicorn ecommerce_review_analyzer:app --reload
# 5. Access the API documentation at http://127.0.0.1:8000/docs