from meta_prompt_optimizer import MetaPromptOptimizer
from product_description_generator import ProductDescriptionGenerator
import random

def simple_feedback_scorer(description: str, product_details: dict) -> float:
    """Simulates a feedback mechanism for the generated description.

    A higher score indicates a better description.
    This is a very simplistic rule-based scorer for demonstration.
    In a real system, this could involve:
    - Keyword density/relevance
    - Readability scores
    - Sentiment analysis
    - A human in the loop
    - Another LLM for evaluation
    - A/B testing results
    """
    score = 0.0
    product_name = product_details["product_name"]
    features = product_details["features"]
    benefits = product_details["benefits"]

    # Check for inclusion of product name and key features/benefits
    if product_name.lower() in description.lower():
        score += 0.2
    for feature in features:
        if feature.lower() in description.lower():
            score += 0.1
    for benefit in benefits:
        if benefit.lower() in description.lower():
            score += 0.05

    # Check for length (avoiding too short or too long)
    word_count = len(description.split())
    if 50 <= word_count <= 150:
        score += 0.3
    elif 30 <= word_count < 50 or 150 < word_count <= 200:
        score += 0.1

    # Check for engaging keywords (simulated)
    engaging_keywords = ["amazing", "innovative", "experience", "unlock", "transform"]
    for keyword in engaging_keywords:
        if keyword.lower() in description.lower():
            score += 0.05

    # Introduce some randomness to simulate real-world variability
    score += random.uniform(-0.1, 0.1)

    return max(0.0, min(1.0, score)) # Ensure score is between 0 and 1

if __name__ == "__main__":
    print("\n--- Smart Product Description Generator (Meta-Prompting Demo) ---\n")

    product_details = {
        "product_name": "Eco-Smart Water Bottle",
        "features": ["double-wall insulation", "BPA-free material", "leak-proof lid", "24-hour cold/12-hour hot"],
        "benefits": ["sustained hydration", "eco-friendly choice", "convenient to carry", "stylish design"]
    }

    meta_optimizer = MetaPromptOptimizer()
    desc_generator = ProductDescriptionGenerator()

    current_prompt = meta_optimizer.generate_initial_prompt(product_details)
    print(f"Initial Prompt generated.\n")

    num_iterations = 5
    best_description = ""
    highest_score = -1.0
    best_prompt = ""

    for i in range(num_iterations):
        print(f"\n--- Iteration {i+1}/{num_iterations} ---")
        print(f"Current Prompt: {current_prompt[:100]}...")

        # 1. Generate description with the current prompt
        generated_description = desc_generator.generate_description(current_prompt, product_details)
        print(f"Generated Description:\n{generated_description}\n")

        # 2. Get feedback on the generated description
        feedback_score = simple_feedback_scorer(generated_description, product_details)
        print(f"Feedback Score: {feedback_score:.2f}\n")

        if feedback_score > highest_score:
            highest_score = feedback_score
            best_description = generated_description
            best_prompt = current_prompt
            print("[INFO] New best description found!")

        # 3. Refine the prompt based on feedback (Meta-Prompting)
        if i < num_iterations - 1: # Don't refine after the last iteration
            current_prompt = meta_optimizer.refine_prompt(current_prompt, feedback_score)
            print(f"Prompt refined for next iteration.\n")

    print("\n--- Optimization Complete ---")
    print(f"Highest Score Achieved: {highest_score:.2f}")
    print(f"Best Prompt:\n{best_prompt}\n")
    print(f"Best Generated Description:\n{best_description}\n")