exemplars = [
    {"email": "My last bill was much higher than expected. Can you check it?", "category": "Billing Inquiry"},
    {"email": "I can't log into my account. I keep getting an error message.", "category": "Technical Support"},
    {"email": "The new feature you released is fantastic! Great work.", "category": "Product Feedback"},
    {"email": "Where is my order? It was supposed to arrive yesterday.", "category": "Order Status"},
    {"email": "I need to update my payment information for my subscription.", "category": "Billing Inquiry"}
]

possible_categories = ["Billing Inquiry", "Technical Support", "Product Feedback", "Order Status"]

def create_few_shot_prompt(instructions: str, exemplars: list, new_email: str) -> str:
    prompt_parts = [instructions, "\n\nExamples:"]
    for ex in exemplars:
        prompt_parts.append(f"Email: {ex['email']}\nCategory: {ex['category']}\n")
    prompt_parts.append(f"New Email: {new_email}\nCategory:")
    return "\n".join(prompt_parts)

def mock_llm_classify(prompt: str, categories: list) -> str:
    # Simulate LLM's classification based on keywords in the new email part of the prompt
    # This is a very basic heuristic to demonstrate the concept
    new_email_start_index = prompt.rfind("New Email: ") + len("New Email: ")
    new_email_end_index = prompt.rfind("\nCategory:")
    new_email_content = prompt[new_email_start_index:new_email_end_index].lower()

    if "bill" in new_email_content or "payment" in new_email_content or "charge" in new_email_content:
        return "Billing Inquiry"
    elif "log in" in new_email_content or "account" in new_email_content or "error" in new_email_content or "technical" in new_email_content:
        return "Technical Support"
    elif "feature" in new_email_content or "feedback" in new_email_content or "great work" in new_email_content:
        return "Product Feedback"
    elif "order" in new_email_content or "where is" in new_email_content or "arrive" in new_email_content or "shipping" in new_email_content:
        return "Order Status"
    else:
        return "Uncategorized"

def classify_customer_email(email_content: str, instructions: str, exemplars: list, categories: list) -> str:
    prompt = create_few_shot_prompt(instructions, exemplars, email_content)
    predicted_category = mock_llm_classify(prompt, categories)
    return predicted_category

if __name__ == "__main__":
    classification_instructions = "Classify the following customer email into one of the predefined categories."

    # Test cases
    email1 = "My internet is not working. I can't connect to any websites."
    email2 = "I received a damaged item in my last delivery."
    email3 = "I want to commend your support team for their quick response."
    email4 = "Can you tell me the status of my recent purchase?"
    email5 = "I was overcharged for my subscription this month."
    email6 = "I have a question about the new product features."

    print(f"Email: '{email1}'\nPredicted Category: {classify_customer_email(email1, classification_instructions, exemplars, possible_categories)}\n")
    print(f"Email: '{email2}'\nPredicted Category: {classify_customer_email(email2, classification_instructions, exemplars, possible_categories)}\n")
    print(f"Email: '{email3}'\nPredicted Category: {classify_customer_email(email3, classification_instructions, exemplars, possible_categories)}\n")
    print(f"Email: '{email4}'\nPredicted Category: {classify_customer_email(email4, classification_instructions, exemplars, possible_categories)}\n")
    print(f"Email: '{email5}'\nPredicted Category: {classify_customer_email(email5, classification_instructions, exemplars, possible_categories)}\n")
    print(f"Email: '{email6}'\nPredicted Category: {classify_customer_email(email6, classification_instructions, exemplars, possible_categories)}\n")