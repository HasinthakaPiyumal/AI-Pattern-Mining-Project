import random
from collections import defaultdict

class EpsilonGreedyBandit:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self.counts = defaultdict(int)
        self.values = defaultdict(float)
        self.arms = []

    def add_arm(self, arm_id, initial_value=0.0):
        if arm_id not in self.arms:
            self.arms.append(arm_id)
            self.values[arm_id] = initial_value

    def select_arm(self):
        if not self.arms:
            return None
        if random.random() < self.epsilon:
            return random.choice(self.arms)
        else:
            return max(self.arms, key=lambda arm: self.values[arm])

    def update(self, arm_id, reward):
        self.counts[arm_id] += 1
        n = self.counts[arm_id]
        old_value = self.values[arm_id]
        new_value = old_value + (reward - old_value) / n
        self.values[arm_id] = new_value

mock_products = [
    {"id": 1, "name": "Smartwatch Pro", "category": "Electronics", "features": "Heart rate monitor, GPS, long battery life, waterproof"},
    {"id": 2, "name": "Organic Coffee Beans", "category": "Groceries", "features": "Single origin, medium roast, arabica, ethically sourced"},
    {"id": 3, "name": "Wireless Earbuds X", "category": "Electronics", "features": "Noise cancellation, 30-hour playback, comfortable fit, Bluetooth 5.2"}
]

def mock_llm_generate_description(product, prompt_template):
    description = f"Prompt: {prompt_template}\n"
    description += f"Product Name: {product['name']}\n"
    description += f"Category: {product['category']}\n"
    description += f"Features: {product['features']}\n"
    description += "A high-quality product designed for modern living. Get yours today!"
    return description

def simulate_ground_truth(product, generated_description):
    if "high-quality" in generated_description and product["category"] == "Electronics":
        return "Excellent, detailed, persuasive."
    elif "ethically sourced" in generated_description and product["category"] == "Groceries":
        return "Good, highlights key value."
    return "Needs improvement, not engaging."

def simulate_reward(generated_description, ground_truth):
    if "Excellent" in ground_truth:
        return random.uniform(0.8, 1.0)
    elif "Good" in ground_truth:
        return random.uniform(0.5, 0.7)
    else:
        return random.uniform(0.1, 0.4)

def mock_llm_criticize_prompt(generated_description, ground_truth, current_prompt):
    if "Needs improvement" in ground_truth:
        criticism = "The description lacks specific benefits and calls to action. The prompt should encourage more persuasive language."
        suggested_prompt = current_prompt.replace("Get yours today!", "Highlight specific benefits and create urgency.")
    elif "Excellent" in ground_truth:
        criticism = "The description is good but could be more concise. Consider shortening some sentences."
        suggested_prompt = current_prompt.replace("long battery life", "extended battery")
    else:
        criticism = "The description is adequate but could use more engaging adjectives."
        suggested_prompt = current_prompt + " (Use more engaging adjectives)"
    return criticism, suggested_prompt

def mock_llm_generate_new_prompts(criticism, base_prompt):
    new_prompts = [base_prompt]
    if "persuasive language" in criticism:
        new_prompts.append(base_prompt + " (Focus on persuasive marketing language)")
    if "concise" in criticism:
        new_prompts.append(base_prompt.replace("Product Name:", "Name:"))
    if "engaging adjectives" in criticism:
        new_prompts.append(base_prompt + " (Incorporate exciting adjectives)")
    return new_prompts

def optimization_loop(num_iterations=10, products_per_batch=1, criticism_frequency=3):
    initial_prompt = "Generate a product description for an e-commerce item, including its name, category, and features. End with 'Get yours today!'"
    
    bandit = EpsilonGreedyBandit(epsilon=0.2)
    bandit.add_arm(initial_prompt)
    current_best_prompt = initial_prompt

    print("Starting prompt optimization...")
    print(f"Initial Prompt: {initial_best_prompt}")

    for i in range(num_iterations):
        print(f"\n--- Iteration {i+1} ---")
        
        selected_prompt = bandit.select_arm()
        if selected_prompt is None:
            selected_prompt = initial_prompt # Fallback if no arms added yet (shouldn't happen after add_arm)
        current_best_prompt = selected_prompt # Keep track of the selected prompt for generating new ones

        rewards_batch = []
        for _ in range(products_per_batch):
            product = random.choice(mock_products)
            generated_description = mock_llm_generate_description(product, selected_prompt)
            ground_truth = simulate_ground_truth(product, generated_description)
            reward = simulate_reward(generated_description, ground_truth)
            rewards_batch.append(reward)
            print(f"  Product: {product['name']} | Prompt: '{selected_prompt[:50]}...' | Reward: {reward:.2f}")

        avg_reward = sum(rewards_batch) / len(rewards_batch)
        bandit.update(selected_prompt, avg_reward)

        if (i + 1) % criticism_frequency == 0:
            print("  Generating criticism and new prompts...")
            # For criticism, we'll just pick one example from the last batch for simplicity
            # In a real system, this would involve aggregate analysis or specific problematic examples
            product_for_criticism = random.choice(mock_products)
            desc_for_criticism = mock_llm_generate_description(product_for_criticism, selected_prompt)
            gt_for_criticism = simulate_ground_truth(product_for_criticism, desc_for_criticism)
            
            criticism_text, suggested_prompt = mock_llm_criticize_prompt(
                desc_for_criticism, gt_for_criticism, selected_prompt
            )
            print(f"    Criticism: {criticism_text[:70]}...")
            
            new_prompts_generated = mock_llm_generate_new_prompts(criticism_text, selected_prompt)
            for new_p in new_prompts_generated:
                if new_p not in bandit.arms:
                    bandit.add_arm(new_p, initial_value=avg_reward) # Initialize new arms with current avg reward
                    print(f"    Added new prompt: '{new_p[:50]}...'\n")

    print("\nOptimization complete.")
    final_best_prompt = max(bandit.arms, key=lambda arm: bandit.values[arm])
    print(f"Final Best Prompt: {final_best_prompt}")
    print("Performance of prompts:")
    for arm in bandit.arms:
        print(f"  '{arm[:50]}...': Avg Reward = {bandit.values[arm]:.2f}, Pulled = {bandit.counts[arm]} times")

if __name__ == "__main__":
    optimization_loop(num_iterations=20, products_per_batch=2, criticism_frequency=5)