import random

class LanguageModel:
    """Simulates a Language Model that generates product descriptions."""
    def generate_descriptions(self, product_name: str, num_candidates: int) -> list[str]:
        candidates = []
        for i in range(num_candidates):
            # Simulate generating a description. In a real scenario, this would involve
            # calling a pre-trained LLM (e.g., using transformers library).
            description = f"A fantastic {product_name} that offers unparalleled features and benefits. This is candidate {i+1}."
            if random.random() > 0.7: # Introduce some variety for scoring
                description += " It's highly recommended for all users."
            else:
                description += " It's a great choice."
            candidates.append(description)
        return candidates

class RewardModel:
    """Simulates a Reward Model that scores product descriptions."""
    def score(self, description: str) -> float:
        # Simulate scoring a description. In a real scenario, this would involve
        # a trained model that evaluates engagement, SEO keywords, readability, etc.
        score = 0.0
        # Example scoring logic: longer descriptions might get a slightly higher base score
        score += len(description) / 100.0

        # Reward for positive keywords (simulated engagement/SEO)
        if "fantastic" in description.lower():
            score += 0.5
        if "unparalleled" in description.lower():
            score += 0.7
        if "highly recommended" in description.lower():
            score += 1.0
        if "great choice" in description.lower():
            score += 0.2
        
        # Add some randomness to simulate real model variability
        score += random.uniform(-0.1, 0.1)

        return round(score, 2)

def apply_rejection_sampling_best_of_n(
    product_name: str,
    num_candidates: int = 5,
    llm: LanguageModel = None,
    reward_model: RewardModel = None
) -> tuple[str, float]:
    """
    Applies the Rejection Sampling (Best-of-N) pattern to generate
    and select the best product description.

    Args:
        product_name (str): The name of the product for which to generate descriptions.
        num_candidates (int): The number of candidate descriptions to generate.
        llm (LanguageModel): An instance of the LanguageModel.
        reward_model (RewardModel): An instance of the RewardModel.

    Returns:
        tuple[str, float]: The best description and its corresponding reward score.
    """
    if llm is None:
        llm = LanguageModel()
    if reward_model is None:
        reward_model = RewardModel()

    print(f"\n--- Generating {num_candidates} candidate descriptions for '{product_name}' ---")
    candidate_descriptions = llm.generate_descriptions(product_name, num_candidates)

    best_description = ""
    highest_score = -1.0

    print("\n--- Scoring candidates ---")
    for i, desc in enumerate(candidate_descriptions):
        score = reward_model.score(desc)
        print(f"Candidate {i+1}: '{desc}' (Score: {score})")
        if score > highest_score:
            highest_score = score
            best_description = desc
            
    return best_description, highest_score

if __name__ == "__main__":
    # Example usage for an e-commerce platform
    product = "Smartwatch Pro X"
    num_samples = 10  # Number of descriptions to sample

    best_desc, best_score = apply_rejection_sampling_best_of_n(
        product_name=product,
        num_candidates=num_samples
    )

    print(f"\n--- Final Selected Description for '{product}' ---")
    print(f"Description: '{best_desc}'")
    print(f"Score: {best_score}")

    product_2 = "Wireless Earbuds Elite"
    num_samples_2 = 7
    best_desc_2, best_score_2 = apply_rejection_sampling_best_of_n(
        product_name=product_2,
        num_candidates=num_samples_2
    )

    print(f"\n--- Final Selected Description for '{product_2}' ---")
    print(f"Description: '{best_desc_2}'")
    print(f"Score: {best_score_2}")
