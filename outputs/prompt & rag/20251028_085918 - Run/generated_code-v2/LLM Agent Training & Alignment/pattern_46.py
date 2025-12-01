import numpy as np
from typing import List, Dict

class MockLanguageModel:
    def __init__(self):
        self.base_descriptions = [
            "Discover our amazing {product_name} with features like {feature1} and {feature2}. Perfect for {target_audience}.",
            "Elevate your experience with the new {product_name}. Boasting {feature1} and {feature2}, it's designed for {target_audience}.",
            "Unleash the power of {product_name}. Featuring {feature1} and {feature2}, a must-have for {target_audience}.",
            "Experience convenience with {product_name}. Its {feature1} and {feature2} make it ideal for {target_audience}.",
            "Revolutionize your daily routine with {product_name}. Enjoy {feature1} and {feature2}, crafted for {target_audience}."
        ]

    def generate_candidates(self, product_details: Dict[str, str], num_candidates: int) -> List[str]:
        candidates = []
        for i in range(num_candidates):
            # Simple substitution for demonstration. A real LM would generate diverse text.
            description_template = self.base_descriptions[i % len(self.base_descriptions)]
            formatted_description = description_template.format(
                product_name=product_details.get("product_name", "product"),
                feature1=product_details.get("feature1", "key feature 1"),
                feature2=product_details.get("feature2", "key feature 2"),
                target_audience=product_details.get("target_audience", "everyone")
            )
            candidates.append(formatted_description)
        return candidates

class MockRewardModel:
    def score(self, description: str) -> Dict[str, float]:
        # Simulate scores based on some arbitrary logic or randomness
        # In a real scenario, this would involve NLP models, keyword analysis, etc.
        np.random.seed(hash(description) % (2**32 - 1)) # Seed for reproducible random scores per description
        engagement_score = np.random.uniform(0.5, 1.0)
        seo_score = np.random.uniform(0.4, 0.95)
        persuasiveness_score = np.random.uniform(0.6, 1.0)
        
        # A very basic rule: longer descriptions might have slightly better SEO potential
        if len(description) > 100: # Arbitrary threshold
            seo_score += 0.05
        
        # Another rule: presence of certain keywords might boost scores
        if "amazing" in description.lower() or "elevate" in description.lower():
            engagement_score += 0.03
            persuasiveness_score += 0.03

        return {"engagement": min(engagement_score, 1.0), "seo": min(seo_score, 1.0), "persuasiveness": min(persuasiveness_score, 1.0)}

class ProductDescriptionGenerator:
    def __init__(self, num_samples: int = 5):
        self.language_model = MockLanguageModel()
        self.reward_model = MockRewardModel()
        self.num_samples = num_samples

    def generate_optimized_description(self, product_details: Dict[str, str]) -> Dict[str, str]:
        print(f"Generating {self.num_samples} candidate descriptions...")
        candidate_descriptions = self.language_model.generate_candidates(product_details, self.num_samples)
        
        scored_descriptions = []
        for i, desc in enumerate(candidate_descriptions):
            scores = self.reward_model.score(desc)
            # Aggregate scores - simple average for demonstration
            total_score = np.mean(list(scores.values()))
            scored_descriptions.append({"description": desc, "scores": scores, "total_score": total_score})
            print(f"Candidate {i+1}:\n  Description: {desc}\n  Scores: {scores}\n  Total Score: {total_score:.2f}\n")

        if not scored_descriptions:
            return {"error": "No descriptions were generated."}

        best_description_data = max(scored_descriptions, key=lambda x: x["total_score"])
        
        return {
            "product_name": product_details.get("product_name", "Unknown Product"),
            "optimized_description": best_description_data["description"],
            "final_score": best_description_data["total_score"],
            "detailed_scores": best_description_data["scores"]
        }

# Example Usage:
if __name__ == "__main__":
    generator = ProductDescriptionGenerator(num_samples=5)

    product_input = {
        "product_name": "Smartwatch Pro X",
        "feature1": "ECG monitoring",
        "feature2": "5-day battery life",
        "target_audience": "health-conscious tech enthusiasts"
    }

    print("\n--- Generating description for Smartwatch Pro X ---")
    result = generator.generate_optimized_description(product_input)
    
    print("\n--- Final Optimized Product Description --- ")
    print(f"Product: {result.get('product_name')}")
    print(f"Description: {result.get('optimized_description')}")
    print(f"Final Score: {result.get('final_score'):.2f}")
    print(f"Detailed Scores: {result.get('detailed_scores')}")

    print("\n--- Generating description for Wireless Earbuds Z ---")
    product_input_2 = {
        "product_name": "Wireless Earbuds Z",
        "feature1": "Active Noise Cancellation",
        "feature2": "Crystal Clear Audio",
        "target_audience": "music lovers and commuters"
    }
    result_2 = generator.generate_optimized_description(product_input_2)

    print("\n--- Final Optimized Product Description --- ")
    print(f"Product: {result_2.get('product_name')}")
    print(f"Description: {result_2.get('optimized_description')}")
    print(f"Final Score: {result_2.get('final_score'):.2f}")
    print(f"Detailed Scores: {result_2.get('detailed_scores')}")
