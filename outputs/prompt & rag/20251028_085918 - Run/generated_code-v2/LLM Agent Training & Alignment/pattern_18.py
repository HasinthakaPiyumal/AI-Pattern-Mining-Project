import random

def generate_descriptions(product_details, num_samples):
    product_name = product_details["product_name"]
    category = product_details["category"]
    features = ", ".join(product_details["features"])
    
    descriptions = []
    for i in range(num_samples):
        variation_phrases = [
            f"Experience superior sound with the {product_name}.",
            f"Elevate your listening with these {product_name}.",
            f"The ultimate {product_name} for unparalleled audio.",
            f"Discover the freedom of {product_name}.",
            f"Unleash crystal-clear audio with the {product_name}."
        ]
        opening = random.choice(variation_phrases)
        closing = random.choice([
            "Perfect for on-the-go or relaxing at home.",
            "Designed for comfort and long-lasting performance.",
            "Immerse yourself in your favorite tunes.",
            "Get yours today and transform your audio experience."
        ])
        description = f"{opening} This {category} features {features}. {closing}"
        descriptions.append(description)
    return descriptions

def score_description(description, product_details):
    score = 0
    product_name = product_details["product_name"].lower()
    keywords = [kw.lower() for kw in product_details["keywords"]]
    features = [feat.lower() for feat in product_details["features"]]

    desc_len = len(description)
    if 100 <= desc_len <= 200:
        score += 5
    elif 50 <= desc_len < 100 or 200 < desc_len <= 300:
        score += 2
    else:
        score -= 1

    for keyword in keywords:
        if keyword in description.lower():
            score += 3

    if product_name in description.lower():
        score += 2

    all_features_present = True
    for feature in features:
        if feature not in description.lower():
            all_features_present = False
            break
    if all_features_present:
        score += 5

    positive_words = ["superior", "premium", "enjoy", "elevate", "ultimate", "unparalleled", "freedom", "crystal-clear", "perfect", "comfort", "transform"]
    for word in positive_words:
        if word in description.lower():
            score += 1

    negative_words = ["bad", "poor", "unreliable", "low quality"]
    for word in negative_words:
        if word in description.lower():
            score -= 5

    return score

def optimize_product_description(product_details, num_samples):
    candidate_descriptions = generate_descriptions(product_details, num_samples)
    scored_descriptions = []
    for desc in candidate_descriptions:
        score = score_description(desc, product_details)
        scored_descriptions.append({"description": desc, "score": score})

    best_description_info = max(scored_descriptions, key=lambda x: x["score"])
    return best_description_info["description"]

if __name__ == "__main__":
    product_details = {
        "product_name": "Wireless Bluetooth Headphones",
        "category": "Electronics",
        "features": ["Noise-cancelling", "Long battery life", "Comfortable fit"],
        "keywords": ["audio", "headset", "portable"]
    }
    N_SAMPLES = 5
    optimized_description = optimize_product_description(product_details, N_SAMPLES)
    print(f"Optimized Product Description:\n{optimized_description}")