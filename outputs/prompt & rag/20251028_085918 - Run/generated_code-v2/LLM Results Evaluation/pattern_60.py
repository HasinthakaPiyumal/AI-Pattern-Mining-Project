import random

# Define the Likert scale categories
LIKERT_SCALE_CATEGORIES = [
    "Very Negative",
    "Negative",
    "Neutral",
    "Positive",
    "Very Positive"
]

def simulate_llm_sentiment_analysis(review_text: str, scale_categories: list) -> str:
    """
    Simulates an LLM's sentiment analysis using a Likert Scale.
    In a real application, this would involve an API call to a Generative AI model
    like Gemini, GPT, etc., prompting it with the review and the scale categories.

    For demonstration, this function uses a very basic keyword-based heuristic.
    """
    review_text_lower = review_text.lower()

    if "terrible" in review_text_lower or "awful" in review_text_lower or "worst" in review_text_lower:
        return scale_categories[0] # Very Negative
    elif "bad" in review_text_lower or "disappointing" in review_text_lower:
        return scale_categories[1] # Negative
    elif "ok" in review_text_lower or "average" in review_text_lower or "neutral" in review_text_lower:
        return scale_categories[2] # Neutral
    elif "good" in review_text_lower or "great" in review_text_lower or "love" in review_text_lower:
        return scale_categories[3] # Positive
    elif "excellent" in review_text_lower or "amazing" in review_text_lower or "best product" in review_text_lower or "incredible" in review_text_lower:
        return scale_categories[4] # Very Positive
    else:
        # If no strong keywords, assign a neutral or slightly positive/negative sentiment randomly
        return random.choice(scale_categories[1:4]) # Randomly choose from Negative, Neutral, Positive

def main():
    print("E-commerce Product Review Sentiment Analyzer with Likert Scale Evaluation\n")

    # Example product reviews
    product_reviews = [
        "This product is absolutely amazing! I love it so much, best purchase ever.",
        "It's okay, not bad, not great. Just average for the price.",
        "Terrible quality, completely broke after a week. Very disappointing.",
        "Good product for the price. Would recommend to others.",
        "I hate this item, it's the worst thing I've ever bought.",
        "A perfectly neutral experience, nothing stood out either positively or negatively.",
        "Very good value and arrived quickly. Happy with my purchase.",
        "Disappointing, expected more given the description.",
        "Incredible! Exceeded all my expectations. Five stars!",
        "The item was fine."
    ]

    print(f"Likert Scale Categories: {', '.join(LIKERT_SCALE_CATEGORIES)}\n")

    for i, review in enumerate(product_reviews):
        sentiment = simulate_llm_sentiment_analysis(review, LIKERT_SCALE_CATEGORIES)
        print(f"Review {i+1}: \"{review}\"")
        print(f"Sentiment: {sentiment}\n")

if __name__ == "__main__":
    main()