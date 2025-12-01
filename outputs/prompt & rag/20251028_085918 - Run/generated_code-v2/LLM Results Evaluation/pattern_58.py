
DEMONSTRATIONS = [
    {
        "query_keywords": ["order", "status"],
        "ambiguous_question_template": "What about my order?",
        "clarification_prompt_template": "To help me with your order, could you please provide your order number or the email address used for the purchase?",
        "options": ["Provide order number", "Provide email address", "Track delivery", "Cancel order"]
    },
    {
        "query_keywords": ["billing", "charge", "invoice"],
        "ambiguous_question_template": "I have a billing question.",
        "clarification_prompt_template": "Regarding your billing question, are you inquiring about a recent charge, an invoice detail, or a subscription plan?",
        "options": ["Recent charge", "Invoice detail", "Subscription plan", "Payment method update"]
    },
    {
        "query_keywords": ["product", "item", "info"],
        "ambiguous_question_template": "Tell me about a product.",
        "clarification_prompt_template": "Which product are you interested in? Please provide the product name or category.",
        "options": ["Specific product name", "Product category", "Feature comparison", "Pricing"]
    },
    {
        "query_keywords": ["account", "login", "password"],
        "ambiguous_question_template": "I need help with my account.",
        "clarification_prompt_template": "Are you looking to reset your password, update your account information, or something else?",
        "options": ["Reset password", "Update account info", "Account security", "Login issues"]
    }
]

def retrieve_relevant_demonstrations(user_query):
    user_query_lower = user_query.lower()
    relevant_demos = []
    for demo in DEMONSTRATIONS:
        for keyword in demo["query_keywords"]:
            if keyword in user_query_lower:
                relevant_demos.append(demo)
                break
    return relevant_demos

def construct_prompt_with_demonstrations(user_query, demonstrations):
    prompt_parts = ["Customer query:", user_query, "\n"]

    if demonstrations:
        prompt_parts.append("Here are some examples of ambiguous questions and how to clarify them:")
        for i, demo in enumerate(demonstrations):
            prompt_parts.append(f"\nExample {i+1}:")
            prompt_parts.append(f"Question: {demo['ambiguous_question_template']}")
            prompt_parts.append(f"Clarification: {demo['clarification_prompt_template']}")
            if demo["options"]:
                prompt_parts.append(f"Options: {", ".join(demo['options'])}")
        prompt_parts.append("\nBased on these examples and your understanding, how would you respond to the customer query?")
    else:
        prompt_parts.append("How would you respond to the customer query?")

    return "\n".join(prompt_parts)

def simulate_llm_response(full_prompt):
    # This function simulates an LLM response based on the prompt content.
    # In a real application, this would be an actual LLM call.

    if "order" in full_prompt.lower() and "clarification" in full_prompt.lower():
        return "It seems your query is about an order. Could you please provide your order number or the email address used for the purchase?"
    elif "billing" in full_prompt.lower() and "clarification" in full_prompt.lower():
        return "For your billing question, are you inquiring about a recent charge, an invoice detail, or a subscription plan?"
    elif "product" in full_prompt.lower() and "clarification" in full_prompt.lower():
        return "To assist you with a product, please provide the product name or category."
    elif "account" in full_prompt.lower() and "clarification" in full_prompt.lower():
        return "Regarding your account, are you looking to reset your password, update your account information, or something else?"
    else:
        return "I'm not sure how to handle that specific ambiguous query. Can you provide more details?"

def run_chatbot():
    print("Welcome to the Intelligent Customer Support Chatbot! (Type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break

        relevant_demos = retrieve_relevant_demonstrations(user_input)
        prompt_for_llm = construct_prompt_with_demonstrations(user_input, relevant_demos)

        # In a real application, you would send prompt_for_llm to an actual LLM API
        llm_response = simulate_llm_response(prompt_for_llm)

        print(f"Chatbot: {llm_response}")

if __name__ == "__main__":
    run_chatbot()
