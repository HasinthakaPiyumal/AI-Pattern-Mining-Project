import json

def summarize_product_review(review_text: str) -> dict:
    """
    Summarizes a product review, extracting sentiment, key features, and issues,
    and formats the output as a structured dictionary (JSON-compatible).

    Args:
        review_text: The raw, unstructured product review text.

    Returns:
        A dictionary containing the structured summary.
    """
    # --- Simulated Sentiment Analysis ---
    # In a real application, this would use an sophisticated NLP model.
    # For this demonstration, we use simple keyword matching to adhere to
    # the constraint of using only built-in Python libraries.
    positive_keywords = ["love", "great", "excellent", "good", "happy", "satisfied", "recommend", "best", "perfect", "amazing", "stunning", "vibrant"]
    negative_keywords = ["hate", "bad", "poor", "terrible", "disappointed", "issue", "problem", "broken", "worst", "useless"]

    sentiment_score = 0
    review_lower = review_text.lower()

    for keyword in positive_keywords:
        if keyword in review_lower:
            sentiment_score += 1
    for keyword in negative_keywords:
        if keyword in review_lower:
            sentiment_score -= 1

    sentiment = "neutral"
    if sentiment_score > 0:
        sentiment = "positive"
    elif sentiment_score < 0:
        sentiment = "negative"

    # --- Simulated Feature and Issue Extraction ---
    # In a real application, this would use more advanced NLP techniques
    # like topic modeling, named entity recognition, or dependency parsing.
    # For this demo, we perform simple word frequency analysis after basic cleaning
    # to adhere to the constraint of using only built-in Python libraries.

    # Basic tokenization and cleaning
    # Remove punctuation and split into words
    cleaned_review = ''.join(char if char.isalnum() or char.isspace() else ' ' for char in review_lower)
    words = cleaned_review.split()

    # Remove common stop words (a very basic set for demonstration)
    stop_words = {"a", "an", "the", "is", "are", "was", "were", "and", "but", "or", "for", "with", "it", "this", "that", "i", "my", "me", "you", "your", "of", "to", "in", "on", "at", "not", "very", "so", "much", "had", "small", "just", "its", "after", "only", "week", "customer", "service", "performance", "design"}
    filtered_words = [word for word in words if word not in stop_words and len(word) > 2]

    word_counts = {}
    for word in filtered_words:
        word_counts[word] = word_counts.get(word, 0) + 1

    # Identify top features/issues based on frequency and specific issue keywords
    common_features = []
    common_issues = []
    feature_issue_threshold = 1 # words appearing at least this many times (low for diverse examples)
    specific_issue_keywords = ["problem", "issue", "broken", "defect", "fail", "difficult", "slow"]

    # Sort words by frequency to prioritize more mentioned items
    sorted_word_counts = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)

    for word, count in sorted_word_counts:
        if count >= feature_issue_threshold:
            # Check if the word is an explicit issue keyword or contains one
            if any(ik in word for ik in specific_issue_keywords):
                if word not in common_issues:
                    common_issues.append(word)
            else:
                if word not in common_features:
                    common_features.append(word)
        
        # Limit the number of extracted items for conciseness
        if len(common_features) >= 3 and len(common_issues) >= 2:
            break

    # Ensure we have at least some items if possible, padding with top words if needed
    # (This part can be more sophisticated for a real app)
    if len(common_features) == 0 and filtered_words:
        for word in sorted_word_counts:
            if word[0] not in common_features and word[0] not in specific_issue_keywords:
                common_features.append(word[0])
                if len(common_features) >= 2: break
    
    if len(common_issues) == 0 and filtered_words:
        for word in sorted_word_counts:
            if word[0] in specific_issue_keywords and word[0] not in common_issues:
                common_issues.append(word[0])
                if len(common_issues) >= 1: break

    # --- Format Output as JSON-compatible dictionary ---
    structured_summary = {
        "review_id": "generated_summary_001", # Placeholder ID
        "original_review": review_text,
        "summary": {
            "overall_sentiment": sentiment,
            "key_features_mentioned": common_features[:3], # Limit to top 3
            "common_issues_identified": common_issues[:2] # Limit to top 2
        }
    }

    return structured_summary

if __name__ == "__main__":
    # Example Usage
    review1 = "I absolutely love this product! The battery life is amazing and the camera takes stunning photos. The screen is also very vibrant. Highly recommend it."
    review2 = "This product is terrible. It broke after only a week, and the customer service was useless. I had a lot of problems with its performance. Very disappointed."
    review3 = "It's okay, nothing special. The design is nice, but the performance is just average. Had a small issue with shipping, but it arrived eventually."
    review4 = "The sound quality is great, but connection drops frequently. Price is a bit high."

    print("--- Review 1 Summary ---")
    summary1 = summarize_product_review(review1)
    print(json.dumps(summary1, indent=2))
    print("\n")

    print("--- Review 2 Summary ---")
    summary2 = summarize_product_review(review2)
    print(json.dumps(summary2, indent=2))
    print("\n")

    print("--- Review 3 Summary ---")
    summary3 = summarize_product_review(review3)
    print(json.dumps(summary3, indent=2))
    print("\n")

    print("--- Review 4 Summary ---")
    summary4 = summarize_product_review(review4)
    print(json.dumps(summary4, indent=2))
    print("\n")