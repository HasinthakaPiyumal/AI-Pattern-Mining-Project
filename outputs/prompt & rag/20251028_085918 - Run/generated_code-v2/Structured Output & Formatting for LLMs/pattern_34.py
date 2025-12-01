from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import random

class ProductSummary(BaseModel):
    product_id: str
    overall_sentiment: str
    key_features_mentioned: List[str]
    common_complaints: List[str]

class ReviewRequest(BaseModel):
    product_id: str
    reviews: List[str]

class ReviewSummarizer:
    def summarize_reviews(self, product_id: str, reviews: List[str]) -> ProductSummary:
        # Simulated AI logic for demonstration of output formatting
        # In a real application, this would involve NLP models for summarization, 
        # sentiment analysis, and aspect extraction.

        overall_sentiment = "neutral"
        key_features = []
        complaints = []

        positive_keywords = ["great", "excellent", "love", "amazing", "good", "fantastic"]
        negative_keywords = ["bad", "poor", "disappointed", "terrible", "problem", "issue"]
        feature_keywords = ["battery life", "camera", "screen", "performance", "design", "software"]

        positive_count = 0
        negative_count = 0

        for review in reviews:
            review_lower = review.lower()
            for keyword in positive_keywords:
                if keyword in review_lower:
                    positive_count += 1
            for keyword in negative_keywords:
                if keyword in review_lower:
                    negative_count += 1
            for feature in feature_keywords:
                if feature in review_lower and feature not in key_features:
                    key_features.append(feature)

            # Simulate common complaints for variety
            if random.random() < 0.2 and "slow" in review_lower and "slowness" not in complaints:
                complaints.append("slowness")
            if random.random() < 0.1 and "expensive" in review_lower and "price" not in complaints:
                complaints.append("high price")

        if positive_count > negative_count:
            overall_sentiment = "positive"
        elif negative_count > positive_count:
            overall_sentiment = "negative"
        else:
            overall_sentiment = "neutral"

        # Ensure lists are not empty for demonstration, add defaults if necessary
        if not key_features:
            key_features = ["general usability"]
        if not complaints:
            complaints = ["minor issues"]
            
        # Trim to a few common ones for brevity
        key_features = list(set(key_features))[:3]
        complaints = list(set(complaints))[:3]

        return ProductSummary(
            product_id=product_id,
            overall_sentiment=overall_sentiment,
            key_features_mentioned=key_features,
            common_complaints=complaints
        )

app = FastAPI()
summarizer = ReviewSummarizer()

@app.post("/summarize_product_reviews", response_model=ProductSummary)
async def summarize_product_reviews(request: ReviewRequest):
    summary = summarizer.summarize_reviews(request.product_id, request.reviews)
    return summary
