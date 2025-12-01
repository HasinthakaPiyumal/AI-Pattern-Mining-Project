from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from transformers import pipeline

app = FastAPI()

# Load NLP models
# Using smaller, efficient models for demonstration
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

class ReviewInput(BaseModel):
    review_id: str
    text: str

class ProductReviewsRequest(BaseModel):
    product_id: str
    reviews: List[ReviewInput]

class ReviewSummary(BaseModel):
    review_id: str
    summary: str
    sentiment: str
    themes: List[str]

class ProductReviewSummary(BaseModel):
    product_id: str
    summarized_reviews: List[ReviewSummary]

@app.post("/summarize_reviews", response_model=ProductReviewSummary)
async def summarize_reviews(request: ProductReviewsRequest):
    all_summaries = []
    for review in request.reviews:
        # Summarization
        summary_result = summarizer(review.text, max_length=50, min_length=10, do_sample=False)
        summary_text = summary_result[0]["summary_text"]

        # Sentiment Analysis
        sentiment_result = sentiment_analyzer(review.text)
        sentiment_label = sentiment_result[0]["label"]

        # Theme Extraction (simplified - often requires more advanced techniques or a more capable LLM)
        # For this example, let's just pick some keywords if they appear
        themes = []
        if "delivery" in review.text.lower():
            themes.append("delivery")
        if "quality" in review.text.lower():
            themes.append("product quality")
        if "price" in review.text.lower() or "cost" in review.text.lower():
            themes.append("pricing")
        if "customer service" in review.text.lower() or "support" in review.text.lower():
            themes.append("customer service")
        if not themes:
            themes.append("general feedback") # Default theme if no keywords found

        all_summaries.append(ReviewSummary(
            review_id=review.review_id,
            summary=summary_text,
            sentiment=sentiment_label,
            themes=themes
        ))

    return ProductReviewSummary(
        product_id=request.product_id,
        summarized_reviews=all_summaries
    )

# To run this application:
# 1. Save the code as `main.py`
# 2. Install dependencies: `pip install fastapi uvicorn transformers pydantic`
# 3. Run from your terminal: `uvicorn main:app --reload`
# 4. Access the API at http://127.0.0.1:8000/docs for interactive documentation.