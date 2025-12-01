
LIKERT_SCALE = {
    "Strongly Disagree": {"value": 1, "description": "Very Negative"},
    "Disagree": {"value": 2, "description": "Negative"},
    "Neutral": {"value": 3, "description": "No strong positive or negative sentiment"},
    "Agree": {"value": 4, "description": "Positive"},
    "Strongly Agree": {"value": 5, "description": "Very Positive"}
}

def simulate_llm_sentiment_analysis(review_text: str) -> dict:
    """
    Simulates an LLM's assessment of sentiment for a given review text and maps it to a Likert scale category.
    In a real application, this would involve an actual API call to an LLM.
    """
    review_text_lower = review_text.lower()

    if any(keyword in review_text_lower for keyword in ["terrible", "horrible", "awful", "hate", "worst", "extremely disappointed", "regret", "bad quality"]):
        return LIKERT_SCALE["Strongly Disagree"]
    elif any(keyword in review_text_lower for keyword in ["bad", "disappointed", "poor", "not good", "unhappy", "issues", "problem", "could be better"]):
        return LIKERT_SCALE["Disagree"]
    elif any(keyword in review_text_lower for keyword in ["ok", "average", "decent", "neutral", "neither good nor bad", "fine", "as expected"]):
        return LIKERT_SCALE["Neutral"]
    elif any(keyword in review_text_lower for keyword in ["good", "satisfied", "happy", "like", "nice", "recommend", "worth it", "works well"]):
        return LIKERT_SCALE["Agree"]
    elif any(keyword in review_text_lower for keyword in ["excellent", "amazing", "love", "fantastic", "perfect", "best", "highly recommend", "incredible", "superb"]):
        return LIKERT_SCALE["Strongly Agree"]
    else:
        # Default to neutral if no strong keywords are found
        return LIKERT_SCALE["Neutral"]

def evaluate_review(review_text: str) -> dict:
    """
    Evaluates a product review using the simulated LLM sentiment analysis and returns a structured result.
    """
    sentiment_result = simulate_llm_sentiment_analysis(review_text)
    # Find the category name based on the returned sentiment_result dictionary
    sentiment_category_name = next(key for key, value in LIKERT_SCALE.items() if value == sentiment_result)

    return {
        "review": review_text,
        "sentiment_category": sentiment_category_name,
        "sentiment_score": sentiment_result["value"],
        "sentiment_description": sentiment_result["description"]
    }

if __name__ == "__main__":
    sample_reviews = [
        "This product is absolutely amazing! I highly recommend it to everyone. The quality is superb.",
        "It's okay, nothing special. I don't love it, but I don't hate it either. It works as expected.",
        "I am extremely disappointed with this purchase. The quality is terrible and it broke on first use.",
        "The item arrived quickly and works as expected. I'm satisfied with my purchase.",
        "This is the worst product I've ever bought. A complete waste of money, bad quality.",
        "Pretty good for the price. I would buy it again, it's a nice item.",
        "It has some minor issues with the battery life, but overall it's acceptable.",
        "Absolutely incredible experience, truly top-notch! This is the best in its category."
    ]

    print("--- E-commerce Product Review Assistant (Likert Scale Sentiment Analysis) ---")
    print("\nProcessing Sample Reviews:\n")

    for i, review in enumerate(sample_reviews):
        result = evaluate_review(review)
        print(f"Review {i+1}: \"{result['review']}\"")
        print(f"  Sentiment Category: {result['sentiment_category']} ({result['sentiment_score']}/5)")
        print(f"  Description: {result['sentiment_description']}")
        print("-" * 70)

    print("\n--- End of Review Analysis ---")
