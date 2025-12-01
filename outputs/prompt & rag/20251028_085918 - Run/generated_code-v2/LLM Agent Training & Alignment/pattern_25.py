import random

def simulate_llm_description_generator(product_info, num_samples):
    """
    Simulates a Large Language Model generating multiple product descriptions.
    In a real application, this would call an actual LLM API or local model.
    """
    product_name = product_info.get("name", "product")
    features = product_info.get("features", [])
    category = product_info.get("category", "item")

    base_descriptions = [
        f"Discover the amazing {product_name}! Perfect for {category} enthusiasts.",
        f"Elevate your experience with our new {product_name}. Featuring {' and '.join(features)}.",
        f"Get your hands on the {product_name}, a must-have for every {category}."
    ]

    candidate_descriptions = []
    for _ in range(num_samples):
        base = random.choice(base_descriptions)
        # Add some variation to simulate different LLM outputs
        variation_phrases = [
            " You won't regret it.",
            " Limited stock available!",
            " Designed for ultimate performance.",
            " Experience the difference.",
            " Order now!"
        ]
        description = base + random.choice(variation_phrases) if random.random() > 0.3 else base
        if features and random.random() > 0.5:
            feature_to_add = random.choice(features)
            if feature_to_add not in description:
                description += f" Key feature: {feature_to_add}."
        candidate_descriptions.append(description)
    return candidate_descriptions

def simulate_reward_model(description, product_info):
    """
    Simulates a Reward Model scoring a product description.
    In a real application, this would be a sophisticated model (e.g., BERT-based classifier,
    fine-tuned sentence-transformer) evaluating multiple quality criteria.

    Score criteria (simplified for simulation):
    - Presence of product name: +20
    - Presence of at least one feature: +10
    - Length between 50 and 150 characters: +15
    - Contains positive sentiment words (simple check): +5
    - Random bonus/penalty to simulate nuanced scoring: +/- 0-10
    """
    score = 0
    product_name = product_info.get("name", "").lower()
    features = [f.lower() for f in product_info.get("features", [])]
    description_lower = description.lower()

    # 1. Product Name presence
    if product_name and product_name in description_lower:
        score += 20

    # 2. Feature presence
    if any(feature in description_lower for feature in features):
        score += 10

    # 3. Length criteria
    desc_len = len(description)
    if 50 <= desc_len <= 150:
        score += 15
    elif 30 <= desc_len < 50 or 150 < desc_len <= 200:
        score += 5

    # 4. Positive sentiment (very basic check)
    positive_words = ["amazing", "excellent", "great", "best", "superior", "high-quality", "innovative", "perfect", "elevate", "must-have"]
    if any(word in description_lower for word in positive_words):
        score += 5

    # 5. Random bonus/penalty for nuance
    score += random.randint(-5, 10) # Introduce some randomness

    # Penalize if it's too short or seems incomplete
    if desc_len < 30:
        score -= 10

    return max(0, score) # Ensure score is not negative

def generate_best_product_description(product_info, N=5):
    """
    Generates N candidate product descriptions using the simulated LLM,
    scores them using the simulated Reward Model, and returns the best one.
    This implements the 'Rejection Sampling (Best-of-N)' pattern.
    """
    print(f"\nGenerating {N} candidate descriptions for '{product_info.get('name')}'...")
    candidate_descriptions = simulate_llm_description_generator(product_info, N)

    best_description = ""
    highest_score = -1

    print("\n--- Candidate Descriptions and Scores ---")
    for i, desc in enumerate(candidate_descriptions):
        score = simulate_reward_model(desc, product_info)
        print(f"Description {i+1}: '{desc}'\nScore: {score}\n")
        if score > highest_score:
            highest_score = score
            best_description = desc

    print("----------------------------------------")
    return best_description, highest_score

if __name__ == "__main__":
    # Example Usage:
    sample_product_info = {
        "name": "Wireless Noise-Cancelling Headphones",
        "features": ["Active Noise Cancellation", "40-hour Battery Life", "Comfortable Earcups", "Bluetooth 5.2"],
        "category": "Audio Electronics",
        "brand": "SoundBlast"
    }

    N_samples = 7  # Number of descriptions to generate and evaluate

    final_description, final_score = generate_best_product_description(sample_product_info, N_samples)

    print(f"\n>>> Final Best Product Description (Score: {final_score}):\n{final_description}")

    sample_product_info_2 = {
        "name": "Smart Home Security Camera",
        "features": ["1080p HD Video", "Motion Detection", "Two-Way Audio", "Cloud Storage"],
        "category": "Home Automation",
        "brand": "GuardianTech"
    }

    final_description_2, final_score_2 = generate_best_product_description(sample_product_info_2, 10)
    print(f"\n>>> Final Best Product Description (Score: {final_score_2}):\n{final_description_2}")
