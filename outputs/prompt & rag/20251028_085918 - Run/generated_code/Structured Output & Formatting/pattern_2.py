from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import re

# Initialize FastAPI app
app = FastAPI(
    title="E-commerce Product Review Sentiment Analyzer",
    description="API to analyze customer product reviews for sentiment and key aspects, returning structured JSON."
)

# --- Sentiment Analysis Module ---
# Load a pre-trained sentiment analysis model from Hugging Face Transformers
# This model classifies text as 'POSITIVE' or 'NEGATIVE'
try:
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        tokenizer="distilbert-base-uncased-finetuned-sst-2-english"
    )
except Exception as e:
    print(f"Error loading sentiment pipeline: {e}")
    sentiment_pipeline = None # Handle case where model loading fails

def analyze_sentiment(text: str) -> str:
    """
    Analyzes the sentiment of the given text using a pre-trained model.
    Maps model output ('POSITIVE', 'NEGATIVE') to 'positive', 'negative', 'neutral'.
    """
    if not sentiment_pipeline:
        return "error_loading_model"

    result = sentiment_pipeline(text)[0]
    label = result["label"]
    score = result["score"]

    # The 'distilbert-base-uncased-finetuned-sst-2-english' model typically gives POSITIVE/NEGATIVE.
    # We can introduce a threshold or simply map directly.
    if label == "POSITIVE":
        return "positive"
    elif label == "NEGATIVE":
        return "negative"
    else:
        # In case the model produces something unexpected, or if a neutral class was desired.
        # For sst-2, it's usually binary.
        return "neutral"

# --- Aspect Extraction Module ---
# A dictionary of keywords mapped to their respective aspects
ASPECT_KEYWORDS = {
    "price": ["price", "cost", "expensive", "cheap", "value", "affordable", "pricy"],
    "quality": ["quality", "durable", "sturdy", "flimsy", "well-made", "material"],
    "delivery": ["delivery", "shipping", "shipped", "delivered", "fast delivery", "late delivery"],
    "customer service": ["customer service", "support", "helpline", "representative"],
    "features": ["features", "functionality", "specs", "performance", "works"],
    "battery life": ["battery", "charge", "lasts", "power"],
    "ease of use": ["easy to use", "setup", "user-friendly", "complicated"],
    "design": ["design", "look", "appearance", "style"]
}

def extract_aspects(text: str) -> list[str]:
    """
    Extracts key aspects from the text based on a predefined list of keywords.
    """
    found_aspects = set()
    text_lower = text.lower()

    for aspect, keywords in ASPECT_KEYWORDS.items():
        for keyword in keywords:
            # Use regex for whole word matching to avoid partial matches (e.g., 'pri' in 'primary')
            if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                found_aspects.add(aspect)
                break # Move to the next aspect once a keyword is found for current aspect

    return sorted(list(found_aspects))

# --- Pydantic Models for Request and Response ---
class ReviewRequest(BaseModel):
    review_text: str

    class Config:
        json_schema_extra = {
            "example": {
                "review_text": "This product is amazing! The quality is superb and the price is very reasonable. Shipping was also incredibly fast."
            }
        }

class ReviewResponse(BaseModel):
    sentiment: str
    aspects: list[str]

    class Config:
        json_schema_extra = {
            "example": {
                "sentiment": "positive",
                "aspects": ["delivery", "price", "quality"]
            }
        }

# --- API Endpoint ---
@app.post("/analyze_review", response_model=ReviewResponse, summary="Analyze product review sentiment and aspects")
async def analyze_review(request: ReviewRequest):
    """
    Analyzes an e-commerce product review to determine its sentiment 
    (positive, negative, or neutral) and extract key product aspects mentioned.
    """
    review_text = request.review_text

    # Perform sentiment analysis
    sentiment = analyze_sentiment(review_text)

    # Extract aspects
    aspects = extract_aspects(review_text)

    return ReviewResponse(sentiment=sentiment, aspects=aspects)

# To run this application:
# 1. Save the code as `main.py`
# 2. Install necessary libraries: `pip install fastapi "uvicorn[standard]" transformers pydantic`
# 3. Run from your terminal: `uvicorn main:app --reload`
# 4. Access the API documentation at `http://127.0.0.1:8000/docs`
