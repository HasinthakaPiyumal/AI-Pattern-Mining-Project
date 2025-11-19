import json

def summarize_reviews(reviews: list) -> list:
    summaries = []
    for i, review in enumerate(reviews):
        # Simulate LLM interaction and structured JSON output
        # In a real application, this would be an API call to an LLM
        # with a prompt engineered to return JSON.
        if i % 2 == 0:
            simulated_llm_response_json = json.dumps({
                "overall_sentiment": "positive",
                "positive_aspects": ["comfortable to wear", "good sound quality"],
                "negative_aspects": ["battery life could be better"],
                "frequently_mentioned_features": ["comfort", "sound", "battery"]
            })
        else:
            simulated_llm_response_json = json.dumps({
                "overall_sentiment": "negative",
                "positive_aspects": ["sleek design"],
                "negative_aspects": ["slow performance", "overheats quickly"],
                "frequently_mentioned_features": ["performance", "design", "heating"]
            })

        try:
            structured_summary = json.loads(simulated_llm_response_json)
            # Basic validation: check for required keys
            required_keys = ["overall_sentiment", "positive_aspects", "negative_aspects", "frequently_mentioned_features"]
            if all(key in structured_summary for key in required_keys):
                summaries.append(structured_summary)
            else:
                print(f"Validation failed for review {i}: Missing required keys in LLM output.")
        except json.JSONDecodeError:
            print(f"JSON decoding error for review {i}: Invalid JSON from LLM.")
        except Exception as e:
            print(f"An unexpected error occurred for review {i}: {e}")

    return summaries

if __name__ == "__main__":
    customer_reviews = [
        "These headphones are super comfortable and the sound quality is amazing, but the battery dies too fast.",
        "I bought this laptop last week. It looks great, but it's really slow and gets hot fast. Disappointing performance.",
        "The camera on this phone is incredible! Pictures are so clear. Love it.",
        "The software is buggy and constantly crashes. I regret buying this product."
    ]

    product_summaries = summarize_reviews(customer_reviews)

    if product_summaries:
        print("\n--- Structured Product Review Summaries ---")
        for i, summary in enumerate(product_summaries):
            print(f"\nReview {i+1} Summary:")
            print(f"  Overall Sentiment: {summary['overall_sentiment']}")
            print(f"  Positive Aspects: {', '.join(summary['positive_aspects'])}")
            print(f"  Negative Aspects: {', '.join(summary['negative_aspects'])}")
            print(f"  Frequently Mentioned Features: {', '.join(summary['frequently_mentioned_features'])}")
    else:
        print("No summaries generated.")