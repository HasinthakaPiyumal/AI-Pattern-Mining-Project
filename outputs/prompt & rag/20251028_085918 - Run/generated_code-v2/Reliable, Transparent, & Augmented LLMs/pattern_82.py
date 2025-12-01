import random
from collections import Counter

# 1. Mock Data
# Represents a training set of labeled tickets
TRAINING_DATA = [
    {"ticket": "My internet is not working.", "category": "Technical Issue"},
    {"ticket": "I can't log into my account.", "category": "Technical Issue"},
    {"ticket": "How do I update my payment method?", "category": "Billing"},
    {"ticket": "I was charged twice this month.", "category": "Billing"},
    {"ticket": "I want to request a new feature for the mobile app.", "category": "Feature Request"},
    {"ticket": "The app crashes when I open it.", "category": "Technical Issue"},
    {"ticket": "What are your subscription plans?", "category": "Product Inquiry"},
    {"ticket": "I need to change my shipping address.", "category": "Shipping"},
    {"ticket": "When will my order arrive?", "category": "Shipping"},
    {"ticket": "I received a damaged item.", "category": "Product Issue"},
]

# 2. LLM Simulation
def simulate_llm_classification(prompt_text: str) -> str:
    """
    Simulates an LLM classifying a ticket based on a prompt.
    In a real scenario, this would be an API call to an actual LLM.
    For demonstration, it extracts a "category" from the prompt or
    makes a random plausible guess based on keywords.
    """
    ticket_to_classify = prompt_text.split("Ticket: ")[-1].split("\nCategory:")[0].strip()

    if "billing" in ticket_to_classify.lower() or "charged" in ticket_to_classify.lower() or "payment" in ticket_to_classify.lower():
        return random.choice(["Billing", "Billing Issue"])
    elif "internet" in ticket_to_classify.lower() or "log in" in ticket_to_classify.lower() or "app crashes" in ticket_to_classify.lower():
        return random.choice(["Technical Issue", "Tech Support"])
    elif "feature" in ticket_to_classify.lower() or "request" in ticket_to_classify.lower():
        return random.choice(["Feature Request", "Product Enhancement"])
    elif "order" in ticket_to_classify.lower() or "shipping" in ticket_to_classify.lower() or "address" in ticket_to_classify.lower():
        return random.choice(["Shipping", "Delivery Issue"])
    elif "product" in ticket_to_classify.lower() or "subscription" in ticket_to_classify.lower():
        return random.choice(["Product Inquiry", "General Inquiry"])
    elif "damaged" in ticket_to_classify.lower():
        return random.choice(["Product Issue", "Damaged Item"])
    else:
        # Fallback for less clear cases, introduce some noise for demonstration of variance
        all_categories = list(set([d["category"] for d in TRAINING_DATA]))
        return random.choice(all_categories)


# 3. Prompt Generation
def generate_few_shot_prompt(new_ticket: str, exemplars: list) -> str:
    """
    Generates a few-shot prompt for the LLM based on provided exemplars.
    """
    prompt = "Classify the following customer support ticket into one of these categories: "
    # Get all unique categories from the training data for the prompt instruction
    all_categories = sorted(list(set([d["category"] for d in TRAINING_DATA])))
    prompt += ", ".join(all_categories)
    prompt += ".\n\n"

    for exemplar in exemplars:
        prompt += f"Ticket: {exemplar['ticket']}\nCategory: {exemplar['category']}\n\n"

    prompt += f"Ticket: {new_ticket}\nCategory:"
    return prompt

# 4. DENSE Ensembling Logic
def classify_ticket_dense(new_ticket: str, training_data: list, num_ensembles: int = 5, exemplars_per_prompt: int = 3) -> tuple[str, list]:
    """
    Classifies a new customer support ticket using the DENSE ensembling pattern.
    Generates multiple few-shot prompts with distinct exemplar subsets,
    calls a simulated LLM for each, and aggregates the results.
    
    Args:
        new_ticket (str): The customer support ticket to classify.
        training_data (list): A list of dictionaries, where each dict has 'ticket' and 'category'.
        num_ensembles (int): The number of distinct prompts/LLM calls to make.
        exemplars_per_prompt (int): The number of exemplars to include in each few-shot prompt.
        
    Returns:
        tuple[str, list]: A tuple containing the final aggregated classification (str)
                          and a list of all individual LLM classification results (list).
    """
    ensemble_results = []

    for i in range(num_ensembles):
        # Select a distinct subset of exemplars for each prompt
        num_exemplars_to_sample = min(exemplars_per_prompt, len(training_data))
        
        # Randomly sample exemplars for each ensemble iteration.
        # This ensures distinct subsets across the ensembles.
        exemplars = random.sample(training_data, num_exemplars_to_sample)
        
        prompt = generate_few_shot_prompt(new_ticket, exemplars)
        classification = simulate_llm_classification(prompt)
        ensemble_results.append(classification)

    # Aggregate results using majority voting
    final_classification = Counter(ensemble_results).most_common(1)[0][0]
    return final_classification, ensemble_results

# 5. Main Workflow (Example Usage)
def main():
    print("--- DENSE Ensembling for Customer Support Ticket Classification ---")
    print("\nThis script demonstrates how the Demonstration Ensembling (DENSE) pattern")
    print("can reduce variance and improve accuracy in FewShot Prompting for LLMs")
    print("by aggregating classifications from multiple prompts, each with a distinct")
    print("subset of exemplars.\n")

    new_ticket_1 = "My account is locked, I can't access anything after the recent update."
    print(f"Classifying ticket: '{new_ticket_1}'")
    final_category_1, all_results_1 = classify_ticket_dense(new_ticket_1, TRAINING_DATA, num_ensembles=5, exemplars_per_prompt=3)
    print(f"  Individual LLM classifications: {all_results_1}")
    print(f"  Final DENSE classification: {final_category_1}")

    print("\n" + "-" * 70 + "\n")

    new_ticket_2 = "I need to know the detailed features and pricing of your premium subscription plan."
    print(f"Classifying ticket: '{new_ticket_2}'")
    final_category_2, all_results_2 = classify_ticket_dense(new_ticket_2, TRAINING_DATA, num_ensembles=7, exemplars_per_prompt=4)
    print(f"  Individual LLM classifications: {all_results_2}")
    print(f"  Final DENSE classification: {final_category_2}")
    
    print("\n" + "-" * 70 + "\n")

    new_ticket_3 = "The delivery driver left my package at the wrong address, it's urgent!"
    print(f"Classifying ticket: '{new_ticket_3}'")
    final_category_3, all_results_3 = classify_ticket_dense(new_ticket_3, TRAINING_DATA, num_ensembles=5, exemplars_per_prompt=3)
    print(f"  Individual LLM classifications: {all_results_3}")
    print(f"  Final DENSE classification: {final_category_3}")

if __name__ == "__main__":
    main()