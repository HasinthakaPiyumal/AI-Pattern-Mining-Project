import numpy as np
import random
import time
from sentence_transformers import SentenceTransformer, util

class ProductDescriptionGenerator:
    def __init__(self, mock_responses=None):
        self.mock_responses = mock_responses if mock_responses is not None else [
            "Experience the ultimate comfort with our premium cotton t-shirt. Perfect for everyday wear.",
            "Our cutting-edge smartphone boasts a stunning display and powerful processor for seamless multitasking.",
            "Transform your living space with this elegant minimalist lamp. A perfect blend of style and function."
        ]

    def generate_description(self, prompt, product_features):
        # Simulate LLM generation with a random mock response
        time.sleep(0.1)  # Simulate API call delay
        feature_str = ", ".join([f"{k}: {v}" for k, v in product_features.items()])
        return f"[Generated based on: {prompt} and features: {feature_str}] {random.choice(self.mock_responses)}"

class PromptScorer:
    def __init__(self, exemplar_descriptions):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.exemplar_embeddings = self.model.encode(exemplar_descriptions, convert_to_tensor=True)
        self.exemplar_descriptions = exemplar_descriptions

    def score_prompt(self, generated_description):
        generated_embedding = self.model.encode(generated_description, convert_to_tensor=True)
        cosine_scores = util.cos_sim(generated_embedding, self.exemplar_embeddings)[0]
        # Return the average or max similarity to exemplars
        return torch.max(cosine_scores).item() # Use torch.max since embeddings are tensors

class PromptOptimizer:
    def __init__(self):
        self.variation_templates = [
            "Generate a detailed product description for {features}. Focus on {focus_area}. ",
            "Write a compelling product overview for {features}. Highlight {highlight_feature}. ",
            "Craft an engaging description for an e-commerce product with {features}. Emphasize {emphasis}. ",
            "Create a concise and impactful product description for {features}. What makes it unique? {unique_aspect}. "
        ]
        self.focus_areas = ["comfort", "performance", "design", "value", "durability"]
        self.highlight_features = ["key benefits", "innovative technology", "user experience"]
        self.emphasis_options = ["customer benefits", "unique selling points", "emotional appeal"]
        self.unique_aspects = ["its innovative features", "its superior quality", "its eco-friendliness"]

    def generate_variations(self, best_prompt, num_variations=3):
        variations = []
        for _ in range(num_variations):
            # Simple string manipulation and template filling for variations
            template = random.choice(self.variation_templates)
            features_placeholder = "product details"
            focus_area = random.choice(self.focus_areas)
            highlight_feature = random.choice(self.highlight_features)
            emphasis = random.choice(self.emphasis_options)
            unique_aspect = random.choice(self.unique_aspects)
            
            variation = template.format(
                features=features_placeholder, 
                focus_area=focus_area, 
                highlight_feature=highlight_feature, 
                emphasis=emphasis, 
                unique_aspect=unique_aspect
            )
            if "{features}" in best_prompt:
                variation = best_prompt.replace("{features}", features_placeholder) # Preserve some structure
            variations.append(variation.strip())
        return variations

class AutomaticPromptEngineer:
    def __init__(self, exemplar_descriptions, initial_prompts, product_features, max_iterations=10, top_k=2, num_variations_per_prompt=3):
        self.product_description_generator = ProductDescriptionGenerator()
        self.prompt_scorer = PromptScorer(exemplar_descriptions)
        self.prompt_optimizer = PromptOptimizer()
        self.current_prompts = list(initial_prompts)
        self.product_features = product_features
        self.max_iterations = max_iterations
        self.top_k = top_k
        self.num_variations_per_prompt = num_variations_per_prompt

    def run_optimization(self):
        best_overall_prompt = ""
        highest_overall_score = -1.0

        for iteration in range(self.max_iterations):
            print(f"\n--- Iteration {iteration + 1}/{self.max_iterations} ---")
            prompt_scores = []
            for prompt in self.current_prompts:
                generated_description = self.product_description_generator.generate_description(prompt, self.product_features)
                score = self.prompt_scorer.score_prompt(generated_description)
                prompt_scores.append((prompt, score, generated_description))
                print(f"Prompt: '{prompt[:70]}...', Score: {score:.4f}")

            # Sort prompts by score in descending order
            prompt_scores.sort(key=lambda x: x[1], reverse=True)
            
            if prompt_scores:
                current_best_prompt, current_highest_score, _ = prompt_scores[0]
                print(f"Current best prompt: '{current_best_prompt[:70]}...' with score: {current_highest_score:.4f}")

                if current_highest_score > highest_overall_score:
                    highest_overall_score = current_highest_score
                    best_overall_prompt = current_best_prompt

                # Select top K prompts for variation
                top_prompts = [p[0] for p in prompt_scores[:self.top_k]]
                new_prompts = []
                for p in top_prompts:
                    variations = self.prompt_optimizer.generate_variations(p, self.num_variations_per_prompt)
                    new_prompts.extend(variations)
                
                # Combine top current prompts and new variations, ensuring some diversity
                self.current_prompts = list(set(top_prompts + new_prompts))
                # Keep a reasonable number of prompts to prevent explosion
                random.shuffle(self.current_prompts)
                self.current_prompts = self.current_prompts[:self.top_k * (self.num_variations_per_prompt + 1)] # Limit total prompts
            else:
                print("No prompts to process. Exiting.")
                break

        print(f"\n--- Optimization Complete ---")
        print(f"Best overall prompt found: '{best_overall_prompt}' with score: {highest_overall_score:.4f}")
        return best_overall_prompt, highest_overall_score

if __name__ == "__main__":
    import torch
    exemplar_descriptions = [
        "This ultra-comfortable t-shirt is made from 100% organic cotton, perfect for sensitive skin and everyday wear. Its breathable fabric ensures you stay cool and fresh all day long, while the classic fit provides a timeless style that pairs with anything.",
        "Discover the future of mobile technology with our revolutionary smartphone. Featuring an edge-to-edge OLED display, a lightning-fast A15 Bionic chip, and an advanced quad-camera system for breathtaking photos and videos. Experience unparalleled performance and stunning visuals.",
        "Illuminate your space with our modern LED desk lamp. Designed with a sleek, minimalist aesthetic, it offers adjustable brightness and color temperature to suit any task or mood. Energy-efficient and durable, it's the perfect addition to your home or office."
    ]

    initial_prompts = [
        "Generate a product description for an e-commerce platform.",
        "Write a short marketing blurb for an online store item.",
        "Create a compelling description for a new product."
    ]

    product_features = {
        "name": "Premium Wireless Earbuds",
        "category": "Electronics",
        """key_selling_points""": ["Noise Cancellation", "10-Hour Battery Life", "Comfort Fit", "Rich Bass"],
        "target_audience": "Music lovers, commuters, fitness enthusiasts"
    }

    ape_optimizer = AutomaticPromptEngineer(
        exemplar_descriptions=exemplar_descriptions,
        initial_prompts=initial_prompts,
        product_features=product_features,
        max_iterations=5,
        top_k=2,
        num_variations_per_prompt=3
    )

    optimized_prompt, final_score = ape_optimizer.run_optimization()

    print(f"\nUsing the optimized prompt to generate a final description:")
    final_generator = ProductDescriptionGenerator()
    final_description = final_generator.generate_description(optimized_prompt, product_features)
    print(final_description)