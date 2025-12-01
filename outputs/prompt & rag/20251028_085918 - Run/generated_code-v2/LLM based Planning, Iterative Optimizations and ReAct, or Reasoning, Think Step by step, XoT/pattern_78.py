import random
import re

class MockLLMService:
    def __init__(self, model_name="mock-llm"):
        self.model_name = model_name

    def generate(self, prompt_template, product_data):
        filled_prompt = prompt_template.format(**product_data)
        # Simulate LLM generating a description based on the prompt
        generated_description = f"Discover the amazing {product_data.get('product_name', 'product')}. It features {product_data.get('features', 'advanced capabilities')}. Enjoy the benefits of {product_data.get('benefits', 'unmatched convenience')}. This {product_data.get('target_audience', 'everyone')}-friendly item is a must-have! {filled_prompt}"
        return generated_description

class GrIPS_Optimizer:
    def __init__(self, llm_service, initial_prompt, evaluation_function, keywords_for_addition=None):
        self.llm_service = llm_service
        self.initial_prompt = initial_prompt
        self.evaluation_function = evaluation_function
        self.keywords_for_addition = keywords_for_addition if keywords_for_addition else [
            "innovative", "high-quality", "exclusive", "limited edition", "smart", "eco-friendly"
        ]
        self.operations = [
            self._apply_deletion,
            self._apply_addition,
            self._apply_swapping,
            self._apply_paraphrasing,
        ]

    def _apply_deletion(self, prompt):
        words = prompt.split()
        if len(words) < 5: # Don't delete too much from short prompts
            return prompt
        start_idx = random.randint(0, len(words) - 1)
        end_idx = min(start_idx + random.randint(1, 3), len(words))
        new_words = words[:start_idx] + words[end_idx:]
        return " ".join(new_words)

    def _apply_addition(self, prompt):
        if not self.keywords_for_addition:
            return prompt
        word_to_add = random.choice(self.keywords_for_addition)
        words = prompt.split()
        insert_idx = random.randint(0, len(words))
        new_words = words[:insert_idx] + [word_to_add] + words[insert_idx:]
        return " ".join(new_words)

    def _apply_swapping(self, prompt):
        sentences = re.split(r'([.!?])\s*', prompt)
        sentences = [s.strip() for s in sentences if s.strip() and s not in ['.', '!', '?']]
        if len(sentences) < 2:
            return prompt
        idx1, idx2 = random.sample(range(len(sentences)), 2)
        sentences[idx1], sentences[idx2] = sentences[idx2], sentences[idx1]
        return ". ".join(sentences) + ('.' if prompt.endswith('.') else '') # Re-add ending punctuation if present

    def _apply_paraphrasing(self, prompt):
        # Simplified paraphrasing: attempt to replace a common word with a synonym or rephrase simple structures
        replacements = {
            "great": "excellent", "good": "superior", "amazing": "fantastic",
            "features": "includes", "benefits": "advantages", "discover": "explore"
        }
        words = prompt.split()
        new_words = []
        for word in words:
            cleaned_word = word.lower().strip(".!?,()").replace(':', '')
            if cleaned_word in replacements and random.random() < 0.5: # 50% chance to replace
                new_words.append(word.replace(cleaned_word, replacements[cleaned_word]))
            else:
                new_words.append(word)
        return " ".join(new_words)

    def _generate_description(self, prompt, product_data):
        return self.llm_service.generate(prompt, product_data)

    def optimize_prompt(self, product_data, num_iterations=10, variations_per_iteration=5):
        current_best_prompt = self.initial_prompt
        best_score = -float('inf')

        print(f"Starting optimization with initial prompt: {current_best_prompt}")

        for i in range(num_iterations):
            candidate_prompts = [current_best_prompt] # Always include the current best

            for _ in range(variations_per_iteration - 1):
                temp_prompt = current_best_prompt
                # Apply a random number of random operations (1 to 3)
                num_ops = random.randint(1, min(len(self.operations), 3))
                for _ in range(num_ops):
                    op = random.choice(self.operations)
                    temp_prompt = op(temp_prompt)
                candidate_prompts.append(temp_prompt)

            iteration_best_prompt = current_best_prompt
            iteration_best_score = best_score

            for prompt_candidate in candidate_prompts:
                generated_desc = self._generate_description(prompt_candidate, product_data)
                score = self.evaluation_function(generated_desc, product_data)

                if score > iteration_best_score:
                    iteration_best_score = score
                    iteration_best_prompt = prompt_candidate

            if iteration_best_score > best_score:
                best_score = iteration_best_score
                current_best_prompt = iteration_best_prompt
                print(f"Iteration {i+1}: New best prompt found (Score: {best_score:.2f})\n-> {current_best_prompt}\n")
            else:
                print(f"Iteration {i+1}: No improvement, best score remains {best_score:.2f}\n")

        return current_best_prompt

def evaluate_description(generated_description, product_data):
    score = 0

    # 1. Keyword Presence (simple check)
    seo_keywords = product_data.get('seo_keywords', [])
    for keyword in seo_keywords:
        if keyword.lower() in generated_description.lower():
            score += 10
        else:
            score -= 5 # Penalize missing keywords

    # 2. Length Optimization
    min_length = product_data.get('min_description_length', 100)
    max_length = product_data.get('max_description_length', 250)
    desc_length = len(generated_description)

    if min_length <= desc_length <= max_length:
        score += 20
    elif desc_length < min_length:
        score -= (min_length - desc_length) / 2 # Penalize being too short
    else: # desc_length > max_length
        score -= (desc_length - max_length) / 5 # Penalize being too long

    # 3. Mock Sentiment Analysis (assuming positive is good)
    # In a real scenario, use a sentiment analysis model (e.g., from transformers)
    mock_sentiment_score = random.uniform(0.7, 0.9) # Simulate a positive sentiment
    score += mock_sentiment_score * 10

    # 4. Mock Readability Score
    # In a real scenario, use textstat library
    words = len(generated_description.split())
    sentences = len(re.split(r'[.!?]', generated_description)) - 1 # Approx sentences
    if sentences > 0: # Avoid division by zero
        avg_words_per_sentence = words / sentences
        if 15 <= avg_words_per_sentence <= 25: # Ideal sentence length range
            score += 15
        else:
            score -= abs(avg_words_per_sentence - 20) / 2 # Penalize too short/long sentences
    else:
        score -= 10 # Penalize if no discernible sentences

    return score

if __name__ == "__main__":
    # 1. Product Data
    product_data = {
        "product_name": "Smart Coffee Mug",
        "features": "temperature control, app connectivity, wireless charging",
        "benefits": "enjoy perfect coffee longer, personalized experience, clutter-free desk",
        "target_audience": "tech enthusiasts, busy professionals",
        "seo_keywords": ["smart mug", "coffee warmer", "temperature control mug", "app control"],
        "min_description_length": 120,
        "max_description_length": 280,
    }

    # 2. Instantiate LLMService (Mock for demonstration)
    llm_service = MockLLMService()

    # 3. Initial Prompt
    initial_prompt_template = (
        "Generate a compelling product description for the {product_name}. "
        "Highlight its key features like {features} and the incredible benefits such as {benefits}. "
        "Emphasize why it's perfect for {target_audience}. "
        "Focus on innovation and convenience."
    )

    # 4. Instantiate GrIPS_Optimizer
    optimizer = GrIPS_Optimizer(
        llm_service=llm_service,
        initial_prompt=initial_prompt_template,
        evaluation_function=evaluate_description,
        keywords_for_addition=["intelligent", "seamless", "premium", "ergonomic", "next-gen"]
    )

    # 5. Optimize the Prompt
    optimized_prompt = optimizer.optimize_prompt(
        product_data=product_data,
        num_iterations=5,
        variations_per_iteration=7
    )

    print("\n--- Optimization Complete ---")
    print(f"Final Optimized Prompt:\n{optimized_prompt}\n")

    # Generate final description with the optimized prompt
    final_description = llm_service.generate(optimized_prompt, product_data)
    final_score = evaluate_description(final_description, product_data)
    print(f"Generated Description with Optimized Prompt:\n{final_description}\n")
    print(f"Final Description Score: {final_score:.2f}")
