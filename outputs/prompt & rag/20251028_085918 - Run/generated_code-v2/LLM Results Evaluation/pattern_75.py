import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Literal, Optional

class Review(BaseModel):
    review_id: str
    product_id: str
    user_id: str
    text: str
    rating: int = Field(..., ge=1, le=5)
    timestamp: str

class EvaluationResult(BaseModel):
    review_id: str
    overall_quality: Literal["excellent", "good", "fair", "poor"] = Field(
        ..., description="Overall assessment of the review's quality and coherence."
    )
    sentiment: Literal["positive", "neutral", "negative", "mixed"] = Field(
        ..., description="The overall sentiment expressed in the review."
    )
    helpful_score: Literal["very_helpful", "helpful", "somewhat_helpful", "not_helpful"] = Field(
        ..., description="Assessment of how helpful the review is for potential buyers."
    )
    is_spam: bool = Field(
        ..., description="True if the review appears to be spam, bot-generated, or irrelevant."
    )
    concerns: Optional[str] = Field(
        None, description="Any specific concerns or issues identified in the review text."
    )
    summary: str = Field(
        ..., description="A concise summary of the review's main points."
    )

class LLMEvaluator:
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def evaluate_review(self, review: Review) -> EvaluationResult:
        prompt = f"""You are an expert e-commerce product review evaluator. Your task is to analyze the provided customer review and provide a structured evaluation in JSON format.

Review Details:
Product ID: {review.product_id}
User ID: {review.user_id}
Rating: {review.rating} stars
Review Text:
---
{review.text}
---

Please evaluate the review based on the following criteria and provide your output as a JSON object that adheres to the following schema:

{{
    "review_id": "{review.review_id}",
    "overall_quality": "string (excellent, good, fair, poor)",
    "sentiment": "string (positive, neutral, negative, mixed)",
    "helpful_score": "string (very_helpful, helpful, somewhat_helpful, not_helpful)",
    "is_spam": "boolean",
    "concerns": "string (optional, any specific issues)",
    "summary": "string (concise summary of main points)"
}}

Constraint: The output MUST be a valid JSON object. Do not include any other text or formatting outside the JSON.
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            json_output = response.choices[0].message.content
            evaluation_data = json.loads(json_output)
            
            return EvaluationResult(**evaluation_data)

        except Exception as e:
            print(f"Error evaluating review {review.review_id}: {e}")
            raise

def main():
    load_dotenv()
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please create a .env file or set the environment variable.")
        return

    evaluator = LLMEvaluator(api_key=openai_api_key)

    sample_reviews = [
        Review(
            review_id="rev_001",
            product_id="prod_A123",
            user_id="user_X789",
            text="This product is absolutely fantastic! The quality is superb and it exceeded my expectations. Highly recommend it to everyone. Worth every penny!",
            rating=5,
            timestamp="2023-10-26T10:00:00Z"
        ),
        Review(
            review_id="rev_002",
            product_id="prod_B456",
            user_id="user_Y012",
            text="It's okay, I guess. Nothing special. The battery life is a bit disappointing, but it works. Not sure if I'd buy it again.",
            rating=3,
            timestamp="2023-10-26T11:15:00Z"
        ),
        Review(
            review_id="rev_003",
            product_id="prod_C789",
            user_id="user_Z345",
            text="BUY THIS NOW!!!! FREE GIFT CARD IF YOU CLICK LINK!!!! www.scamlink.com Best product ever!!!!",
            rating=1,
            timestamp="2023-10-26T12:30:00Z"
        ),
        Review(
            review_id="rev_004",
            product_id="prod_A123",
            user_id="user_W987",
            text="The color is great and it fits perfectly. However, the material feels a bit cheap, and it started pilling after just one wash. Mixed feelings.",
            rating=3,
            timestamp="2023-10-26T13:45:00Z"
        )
    ]

    print("Starting review evaluation...\n")
    for review in sample_reviews:
        print(f"Evaluating review ID: {review.review_id}")
        try:
            evaluation = evaluator.evaluate_review(review)
            print("Evaluation Result (JSON):")
            print(evaluation.model_dump_json(indent=2))
            print("-" * 50)
        except Exception as e:
            print(f"Failed to evaluate review {review.review_id}: {e}")
            print("-" * 50)

if __name__ == "__main__":
    main()