from pydantic import BaseModel, Field, ValidationError
from typing import List, Literal
import json

# 3. Output Structure Definition (Pydantic)
class ReviewEvaluation(BaseModel):
    review_id: str
    sentiment: Literal["positive", "negative", "neutral"]
    quality_score: int = Field(..., ge=1, le=5)
    reasoning: str
    flags: List[str]

# 2. LLM Integration (Simulated/Placeholder)
class SimulatedLLM:
    def evaluate_review(self, review_text: str, review_id: str) -> str:
        # Simulate LLM's structured JSON output based on review content
        if "excellent" in review_text.lower() or "love" in review_text.lower():
            sentiment = "positive"
            quality = 5
            reason = "Highly positive language indicating satisfaction."
            flags = []
        elif "bad" in review_text.lower() or "disappointed" in review_text.lower():
            sentiment = "negative"
            quality = 2
            reason = "Negative sentiment due to product shortcomings."
            flags = []
            if "spam" in review_text.lower():
                flags.append("potential_spam")
        elif "mediocre" in review_text.lower() or "okay" in review_text.lower():
            sentiment = "neutral"
            quality = 3
            reason = "Neutral feedback, neither strongly positive nor negative."
            flags = []
        elif "profane" in review_text.lower() or "damn" in review_text.lower():
            sentiment = "negative"
            quality = 1
            reason = "Contains profanity, indicating strong dissatisfaction."
            flags = ["profanity", "low_quality"]
        else:
            sentiment = "neutral"
            quality = 4
            reason = "General positive tone, helpful."
            flags = []

        output_data = {
            "review_id": review_id,
            "sentiment": sentiment,
            "quality_score": quality,
            "reasoning": reason,
            "flags": flags,
        }
        return json.dumps(output_data)

# 4. Review Evaluator Module
class ReviewEvaluator:
    def __init__(self):
        self.llm = SimulatedLLM()

    def evaluate_review(self, review_id: str, review_text: str) -> ReviewEvaluation:
        llm_raw_output = self.llm.evaluate_review(review_text, review_id)
        try:
            evaluation_data = json.loads(llm_raw_output)
            validated_evaluation = ReviewEvaluation(**evaluation_data)
            return validated_evaluation
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from LLM for review {review_id}: {e}")
            raise
        except ValidationError as e:
            print(f"Validation error for review {review_id}: {e.json()}")
            raise

# 5. Main Application Script
if __name__ == "__main__":
    # 1. Data Ingestion (Simulated)
    sample_reviews = [
        "This product is excellent! I love everything about it.",
        "I'm really disappointed with the quality. It broke after a week.",
        "It's just okay, nothing special. The features are mediocre.",
        "Beware! This is a scam and utterly useless. Damn it!",
        "Works as expected, pretty good overall.",
        "The worst experience ever. Totally regret buying this. This is spam."
    ]

    evaluator = ReviewEvaluator()

    print("--- Starting Review Evaluation ---")
    for i, review_text in enumerate(sample_reviews):
        review_id = f"review_{i+1}"
        print(f"\nEvaluating Review ID: {review_id}")
        print(f"Review Text: '{review_text}'")
        try:
            evaluation_result = evaluator.evaluate_review(review_id, review_text)
            print("Structured Evaluation Result (JSON):")
            print(evaluation_result.json(indent=2))
        except (json.JSONDecodeError, ValidationError) as e:
            print(f"Failed to evaluate review {review_id} due to error: {e}")
    print("\n--- Review Evaluation Finished ---")