import random
from collections import Counter

TICKET_DATA = [
    {"ticket_id": "T001", "text": "My internet is not working.", "category": "Technical Support"},
    {"ticket_id": "T002", "text": "I can't log into my account.", "category": "Account Issues"},
    {"ticket_id": "T003", "text": "How do I update my billing information?", "category": "Billing"},
    {"ticket_id": "T004", "text": "I need help with a software bug.", "category": "Technical Support"},
    {"ticket_id": "T005", "text": "My subscription renewal failed.", "category": "Billing"},
    {"ticket_id": "T006", "text": "I forgot my password.", "category": "Account Issues"},
    {"ticket_id": "T007", "text": "The website is down.", "category": "Technical Support"},
    {"ticket_id": "T008", "text": "I want to change my email address.", "category": "Account Issues"},
    {"ticket_id": "T009", "text": "Where is my invoice?", "category": "Billing"},
    {"ticket_id": "T010", "text": "Product feature request.", "category": "Feature Request"},
    {"ticket_id": "T011", "text": "My payment was declined.", "category": "Billing"},
    {"ticket_id": "T012", "text": "I need to reset my password.", "category": "Account Issues"},
    {"ticket_id": "T013", "text": "The application crashes often.", "category": "Technical Support"},
    {"ticket_id": "T014", "text": "I want to upgrade my plan.", "category": "Billing"},
    {"ticket_id": "T015", "text": "Question about data privacy.", "category": "Privacy"},
    {"ticket_id": "T016", "text": "Troubleshooting connection problems.", "category": "Technical Support"},
    {"ticket_id": "T017", "text": "Unable to access premium features.", "category": "Account Issues"},
    {"ticket_id": "T018", "text": "Complaint about a recent charge.", "category": "Billing"},
    {"ticket_id": "T019", "text": "New user onboarding guide request.", "category": "General Inquiry"},
    {"ticket_id": "T020", "text": "The login button is not working.", "category": "Technical Support"},
]

class LLMSimulator:
    def __init__(self, known_categories):
        self.known_categories = known_categories
        self.category_mapping = {
            "internet not working": "Technical Support",
            "log in": "Account Issues",
            "billing information": "Billing",
            "software bug": "Technical Support",
            "subscription renewal": "Billing",
            "forgot password": "Account Issues",
            "website down": "Technical Support",
            "change email": "Account Issues",
            "invoice": "Billing",
            "feature request": "Feature Request",
            "payment declined": "Billing",
            "reset password": "Account Issues",
            "application crashes": "Technical Support",
            "upgrade plan": "Billing",
            "data privacy": "Privacy",
            "connection problems": "Technical Support",
            "access premium features": "Account Issues",
            "recent charge": "Billing",
            "onboarding guide": "General Inquiry",
            "login button not working": "Technical Support",
        }

    def _predict_based_on_keywords(self, text):
        text_lower = text.lower()
        for keyword, category in self.category_mapping.items():
            if keyword in text_lower:
                return category
        return random.choice(self.known_categories)

    def generate_response(self, prompt, temperature=0.5):
        lines = prompt.strip().split('\n')
        if not lines:
            return "Unknown"

        new_ticket_text = ""
        for i in reversed(range(len(lines))):
            if "Ticket:" in lines[i]:
                new_ticket_text = lines[i].replace("Ticket:", "").strip()
                break
        
        if not new_ticket_text:
            for i in reversed(range(len(lines))):
                if lines[i].strip():
                    new_ticket_text = lines[i].strip()
                    break

        if not new_ticket_text:
            return random.choice(self.known_categories)

        if random.random() < temperature:
            return random.choice(self.known_categories)
        else:
            return self._predict_based_on_keywords(new_ticket_text)

def create_few_shot_prompt(exemplars, new_ticket_text, num_exemplars=3):
    if not exemplars:
        return f"Categorize the following customer support ticket:\nTicket: {new_ticket_text}\nCategory:"

    num_exemplars = min(num_exemplars, len(exemplars))

    selected_exemplars = random.sample(exemplars, num_exemplars)

    prompt_parts = ["Categorize the following customer support tickets into one of the predefined categories."]
    prompt_parts.append("Categories: Technical Support, Account Issues, Billing, Feature Request, Privacy, General Inquiry.")
    prompt_parts.append("Here are some examples:")

    for ex in selected_exemplars:
        prompt_parts.append(f"Ticket: {ex['text']}\nCategory: {ex['category']}")

    prompt_parts.append(f"\nNow, categorize the following ticket:\nTicket: {new_ticket_text}\nCategory:")

    return "\n".join(prompt_parts)

class DenseCategorizer:
    def __init__(self, training_data, known_categories, num_ensembles=5, exemplars_per_prompt=3):
        self.training_data = training_data
        self.known_categories = known_categories
        self.num_ensembles = num_ensembles
        self.exemplars_per_prompt = exemplars_per_prompt
        self.llm = LLMSimulator(known_categories)

    def categorize_ticket(self, new_ticket_text):
        if not self.training_data:
            print("Warning: No training data provided. Categorizing with basic LLM.")
            prompt = f"Categorize the following customer support ticket:\nTicket: {new_ticket_text}\nCategory:"
            return self.llm.generate_response(prompt, temperature=0.1), []

        individual_predictions = []

        for _ in range(self.num_ensembles):
            if len(self.training_data) < self.exemplars_per_prompt:
                print("Warning: Not enough training data to sample desired number of exemplars.")
                exemplars_subset = self.training_data
            else:
                exemplars_subset = random.sample(self.training_data, self.exemplars_per_prompt)

            prompt = create_few_shot_prompt(exemplars_subset, new_ticket_text, self.exemplars_per_prompt)

            prediction = self.llm.generate_response(prompt, temperature=0.3)
            individual_predictions.append(prediction)

        if not individual_predictions:
            return "Unknown", []

        category_counts = Counter(individual_predictions)
        final_category = category_counts.most_common(1)[0][0]

        return final_category, individual_predictions

def run_demonstration():
    print("--- DENSE (Demonstration Ensembling) Categorization Demonstration ---")

    known_categories = list(set([ticket['category'] for ticket in TICKET_DATA]))
    print(f"Known Categories: {known_categories}")

    categorizer = DenseCategorizer(
        training_data=TICKET_DATA,
        known_categories=known_categories,
        num_ensembles=5,
        exemplars_per_prompt=3
    )

    test_tickets = [
        "My internet is completely down, I can't connect to anything.",
        "I need to change my credit card details for my subscription.",
        "I forgot my password and can't access my account.",
        "Can I request a new feature for the mobile app?",
        "I have a general question about your services.",
        "The application keeps crashing every time I open it.",
        "Where can I find information about data retention policy?",
        "I want to know if you offer any discounts for students.",
    ]

    print("\n--- Categorizing New Tickets ---")
    for i, ticket_text in enumerate(test_tickets):
        print(f"\nProcessing Ticket {i+1}: \"{ticket_text}\"" )
        final_category, all_predictions = categorizer.categorize_ticket(ticket_text)
        print(f"  Individual LLM Predictions: {all_predictions}")
        print(f"  Final Aggregated Category (DENSE): {final_category}")
        print("-" * 30)

if __name__ == "__main__":
    run_demonstration()
