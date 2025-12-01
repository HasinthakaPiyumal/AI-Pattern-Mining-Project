import random
from collections import Counter

class MockLLM:
    def __init__(self, categories):
        self.categories = categories

    def generate_response(self, prompt: str) -> str:
        # Simulate LLM behavior to extract a category from the prompt.
        # In a real scenario, this would be an actual LLM API call.
        for category in self.categories:
            if category.lower() in prompt.lower():
                return category
        # If no specific category is found, pick a random one for demonstration
        return random.choice(self.categories)

historical_tickets = [
    {"text": "My internet is not working, it keeps disconnecting.", "category": "Technical Issue"},
    {"text": "I need to update my billing address and payment method.", "category": "Billing Inquiry"},
    {"text": "Can I add a new feature to my account, like multi-user access?", "category": "Feature Request"},
    {"text": "I forgot my password and cannot log in to my account.", "category": "Account Management"},
    {"text": "My service is slow after the last update.", "category": "Technical Issue"},
    {"text": "Where can I find my past invoices?", "category": "Billing Inquiry"},
    {"text": "It would be great if you could integrate with Zapier.", "category": "Feature Request"},
    {"text": "How do I change my profile picture?", "category": "Account Management"},
    {"text": "My device is not connecting to the network.", "category": "Technical Issue"},
    {"text": "I have a question about my last charge.", "category": "Billing Inquiry"},
]

def generate_exemplar_subsets(exemplars: list, num_subsets: int, exemplars_per_subset: int) -> list:
    subsets = []
    for _ in range(num_subsets):
        subset = random.sample(exemplars, min(exemplars_per_subset, len(exemplars)))
        subsets.append(subset)
    return subsets

def create_few_shot_prompt(incoming_ticket_text: str, exemplar_subset: list) -> str:
    prompt_parts = []
    prompt_parts.append("Categorize the following customer support tickets. Choose from: Technical Issue, Billing Inquiry, Feature Request, Account Management.")
    prompt_parts.append("\n--- Examples ---\n")
    for ex in exemplar_subset:
        prompt_parts.append(f"Ticket: {ex['text']}\nCategory: {ex['category']}\n")
    prompt_parts.append("\n--- New Ticket ---\n")
    prompt_parts.append(f"Ticket: {incoming_ticket_text}\nCategory:")
    return "".join(prompt_parts)

def aggregate_categorizations(categorizations: list) -> str:
    if not categorizations:
        return "Uncategorized"
    # Majority voting
    most_common = Counter(categorizations).most_common(1)
    return most_common[0][0] if most_common else "Uncategorized"

def categorize_ticket_dense(ticket_text: str, llm: MockLLM, exemplars: list, num_prompts: int = 5, exemplars_per_prompt: int = 3) -> str:
    all_categorizations = []

    # Generate multiple distinct subsets of exemplars
    exemplar_subsets = generate_exemplar_subsets(exemplars, num_prompts, exemplars_per_prompt)

    for i, subset in enumerate(exemplar_subsets):
        # Create a few-shot prompt for each subset
        prompt = create_few_shot_prompt(ticket_text, subset)
        
        # Get LLM's categorization for this prompt
        llm_response = llm.generate_response(prompt)
        all_categorizations.append(llm_response)

    # Aggregate the categorizations using majority voting
    final_category = aggregate_categorizations(all_categorizations)
    print(f"Individual categorizations for '{ticket_text}': {all_categorizations}")
    return final_category

if __name__ == "__main__":
    # Define the possible categories
    possible_categories = ["Technical Issue", "Billing Inquiry", "Feature Request", "Account Management"]
    
    # Initialize the mock LLM
    mock_llm = MockLLM(possible_categories)

    # Example incoming tickets
    new_tickets = [
        "I can't access my dashboard, it shows an error 500.",
        "When will my next payment be due?",
        "I wish you had an option to export data to CSV.",
        "My account is locked, how can I unlock it?",
        "The application crashes every time I open settings."
    ]

    print("--- DENSE Ticket Categorization Examples ---")
    for ticket in new_tickets:
        final_category = categorize_ticket_dense(ticket, mock_llm, historical_tickets, num_prompts=5, exemplars_per_prompt=3)
        print(f"Ticket: '{ticket}'\nFinal DENSE Category: {final_category}\n")

    print("--- DENSE with more prompts/exemplars (demonstrates robustness) ---")
    # Example demonstrating higher robustness with more prompts/exemplars
    ticket_for_robustness = "I need help resetting my login credentials."
    final_category_robust = categorize_ticket_dense(ticket_for_robust, mock_llm, historical_tickets, num_prompts=10, exemplars_per_prompt=4)
    print(f"Ticket: '{ticket_for_robust}'\nFinal DENSE Category (robust): {final_category_robust}\n")