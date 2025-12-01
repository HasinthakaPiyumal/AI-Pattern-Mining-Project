from fastapi import FastAPI
from pydantic import BaseModel
import random

def evaluate_review_quality(review_text: str) -> dict:
    min_score = 1
    max_score = 5
    base_score = 3
    explanation_parts = []

    # Length heuristic
    review_length = len(review_text)
    if review_length < 30:
        base_score -= 2
        explanation_parts.append("The review is very short and lacks detail.")
    elif 30 <= review_length < 70:
        base_score -= 1
        explanation_parts.append("The review is somewhat short.")
    elif 150 <= review_length < 300:
        base_score += 1
        explanation_parts.append("The review is reasonably detailed.")
    elif review_length >= 300:
        base_score += 2
        explanation_parts.append("The review is very detailed and comprehensive.")
    else:
        explanation_parts.append("The review has an average length.")

    # Keyword sentiment heuristic
    positive_keywords = ["great", "excellent", "love", "amazing", "fantastic", "good", "happy", "useful", "satisfied", "recommend", "perfect", "sturdy", "efficient"]
    negative_keywords = ["bad", "poor", "terrible", "disappointed", "hate", "broken", "faulty", "unhappy", "not recommend", "flimsy", "slow", "difficult"]

    positive_count = sum(1 for word in positive_keywords if word in review_text.lower())
    negative_count = sum(1 for word in negative_keywords if word in review_text.lower())

    if positive_count > negative_count:
        base_score += 1
        explanation_parts.append("It contains positive sentiment.")
    elif negative_count > positive_count:
        base_score -= 1
        explanation_parts.append("It contains negative sentiment.")
    else:
        explanation_parts.append("It has neutral or mixed sentiment.")

    # Add a touch of randomness for 