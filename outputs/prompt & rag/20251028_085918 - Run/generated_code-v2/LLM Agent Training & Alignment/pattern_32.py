from transformers import pipeline

class ProductDescriptionGenerator:
    def __init__(self, llm_model_name="gpt2", reward_model_threshold=0.5):
        self.llm_generator = pipeline("text-generation", model=llm_model_name)
        self.reward_model_threshold = reward_model_threshold

    def _generate_candidate_description(self, product_details):
        prompt = f"Write a compelling e-commerce product description for {product_details['name']}. Category: {product_details['category']}. Features: {', '.join(product_details['features'])}.\nDescription:"
        generated_text = self.llm_generator(prompt, max_length=150, num_return_sequences=1, do_sample=True, temperature=0.7, top_k=50, truncation=True)[0]['generated_text']
        # Extract only the generated description part after the prompt
        description_start_index = generated_text.find("Description:") + len("Description:")
        return generated_text[description_start_index:].strip()

    def _score_description(self, description):
        # Dummy reward model: scores based on length and presence of keywords
        score = 0.0
        if len(description.split()) > 30: # Good length
            score += 0.4
        if any(keyword in description.lower() for keyword in ["innovative", "high-quality", "exclusive", "premium", "durable"]):
            score += 0.3
        if "seo" in description.lower() or "optimize" in description.lower(): # Placeholder for SEO aspect
            score += 0.2
        if "buy now" in description.lower() or "shop today" in description.lower(): # Placeholder for persuasiveness
            score += 0.1
        return score

    def generate_best_description(self, product_details, num_samples=5):
        candidate_descriptions = []
        for _ in range(num_samples):
            desc = self._generate_candidate_description(product_details)
            candidate_descriptions.append(desc)

        scored_descriptions = []
        for desc in candidate_descriptions:
            score = self._score_description(desc)
            scored_descriptions.append((desc, score))

        # Select the description with the highest score
        best_description, best_score = max(scored_descriptions, key=lambda item: item[1])

        return best_description, best_score

if __name__ == "__main__":
    # Example Usage
    generator = ProductDescriptionGenerator(llm_model_name="distilgpt2") # Using distilgpt2 for faster local execution

    product_info = {
        "name": "Smart Home Assistant",
        "category": "Electronics",
        "features": ["voice control", "AI-powered", "multi-room audio", "smart home integration", "privacy features"]
    }

    print("\nGenerating product descriptions using Rejection Sampling (Best-of-N)...\n")
    best_desc, score = generator.generate_best_description(product_info, num_samples=5)

    print(f"Selected Best Description (Score: {score:.2f}):\n{best_desc}\n")

    product_info_2 = {
        "name": "Organic Herbal Tea Blend",
        "category": "Food & Beverages",
        "features": ["all-natural", "caffeine-free", "relaxing effect", "sustainable sourcing", "biodegradable packaging"]
    }

    print("\nGenerating product descriptions for another product...\n")
    best_desc_2, score_2 = generator.generate_best_description(product_info_2, num_samples=7)

    print(f"Selected Best Description (Score: {score_2:.2f}):\n{best_desc_2}\n")