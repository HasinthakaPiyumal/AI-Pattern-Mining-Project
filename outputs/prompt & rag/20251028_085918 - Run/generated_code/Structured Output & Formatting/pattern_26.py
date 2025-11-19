import json
from typing import List, Optional
from pydantic import BaseModel, ValidationError

class ReviewSummary(BaseModel):
    sentiment: str
    features_mentioned: List[str]
    star_rating: Optional[int]

def _simulate_llm_call(review: str) -> str:
    review = review.lower()
    sentiment = "neutral"
    features = []
    rating = None

    if "great" in review or "excellent" in review or "love" in review or "fantastic" in review:
        sentiment = "positive"
    elif "bad" in review or "terrible" in review or "hate" in review or "disappointed" in review:
        sentiment = "negative"

    if "battery life" in review:
        features.append("battery life")
    if "camera" in review:
        features.append("camera")
    if "screen" in review or "display" in review:
        features.append("screen")
    if "performance" in review:
        features.append("performance")
    if "price" in review or "cost" in review:
        features.append("price")

    for i in range(1, 6):
        if f"{i} star" in review or f"{i}/5" in review:
            rating = i
            break

    if rating is None:
        # Simple heuristic if no explicit star found, based on sentiment
        if sentiment == "positive":
            rating = 5
        elif sentiment == "negative":
            rating = 1
        else:
            rating = 3

    output_data = {
        "sentiment": sentiment,
        "features_mentioned": features,
        "star_rating": rating
    }
    return json.dumps(output_data)

def summarize_review(review: str) -> Optional[ReviewSummary]:
    try:
        llm_output_str = _simulate_llm_call(review)
        llm_output_json = json.loads(llm_output_str)
        summary = ReviewSummary(**llm_output_json)
        return summary
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from LLM output: {e}")
        return None
    except ValidationError as e:
        print(f"Error validating LLM output against Pydantic model: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

if __name__ == "__main__":
    reviews = [
        "This phone has an amazing camera and the battery life is excellent. Definitely a 5-star product!",
        "The screen is terrible and the performance is sluggish. Very disappointed, 1/5 stars.",
        "It's an okay product, nothing special. The price could be better.",
        "Love the new display! Fast shipping too.",
        "Absolutely terrible, don't buy this. Horrible experience."
    ]

    for i, review_text in enumerate(reviews):
        print(f"\n--- Review {i+1} ---")
        print(f"Input: {review_text}")
        summary_obj = summarize_review(review_text)

        if summary_obj:
            print(f"Summary: {summary_obj.model_dump_json(indent=2)}")
        else:
            print("Failed to summarize review.")