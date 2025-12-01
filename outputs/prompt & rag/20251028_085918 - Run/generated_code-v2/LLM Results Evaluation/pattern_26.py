"""AI-powered customer support chatbot demonstrating the 'Ambiguous Demonstrations' pattern."""

def _simulate_llm_response(prompt: str) -> str:
    """Simulates an LLM's response based on the prompt content.
    In a real application, this would be an actual API call to an LLM (e.g., OpenAI, Gemini, HuggingFace inference).
    """
    print(f"\n--- Simulated LLM Prompt ---\n{prompt}\n----------------------------")

    # Simple keyword-based simulation for demonstration
    if "tracking" in prompt.lower() and "status" in prompt.lower() and "package" in prompt.lower():
        return "It seems your query might be about package tracking. Could you please provide your order number so I can check the status?"
    elif "item" in prompt.lower() and "missing" in prompt.lower() and "part" in prompt.lower():
        return "I understand you're missing something from your order. To clarify, are you missing an entire item from your order, or a component/part within a single product? Please provide more details or your order number."
    elif "return" in prompt.lower() and "policy" in prompt.lower():
        return "Our return policy allows returns within 30 days of purchase for most items. Please visit our 'Returns & Refunds' page for detailed instructions and exceptions. Do you have a specific item in mind?"
    else:
        return "I'm designed to help with a wide range of inquiries. To assist you better, could you please rephrase your question or provide more specific details?"


def _get_ambiguous_demonstrations() -> str:
    """Returns a string of predefined ambiguous demonstrations for ICL.
    In a more advanced system, these could be retrieved dynamically.
    """
    demonstrations = [
        {
            "query": "My order is missing a part.",
            "response": "I understand you're missing something from your order. To clarify, are you missing an entire item from your order (e.g., you ordered two shirts but only received one), or a component/part within a single product (e.g., a screw from a furniture assembly kit)? Please provide your order number or more details."
        },
        {
            "query": "What's the status?",
            "response": "The status of what exactly? Are you asking about an order, a return, or something else? Please specify what you'd like to know the status of, and if it's an order or return, please provide the corresponding number."
        },
        {
            "query": "I need help with a product.",
            "response": "I can certainly help! Could you please tell me which product you're referring to and what kind of help you need? For example, are you looking for assembly instructions, troubleshooting tips, or something else?"
        }
    ]

    demo_string = []
    for i, demo in enumerate(demonstrations):
        demo_string.append(f"Customer: {demo['query']}")
        demo_string.append(f"Chatbot: {demo['response']}")
        if i < len(demonstrations) - 1:
            demo_string.append("") # Add a blank line between demonstrations

    return "\n".join(demo_string)


def generate_chatbot_prompt(user_query: str) -> str:
    """Generates the complete prompt for the LLM, including ambiguous demonstrations.
    """
    system_message = (
        "You are an AI-powered customer support chatbot for an e-commerce platform. "
        "Your goal is to assist customers efficiently by understanding their queries, "
        "especially ambiguous ones. If a query is ambiguous, try to ask clarifying questions "
        "or offer multiple interpretations based on the provided examples. "
        "Always be polite and helpful.\n\n"
        "Here are some examples of ambiguous customer queries and how to respond to them "
        "by asking for clarification or offering interpretations:\n\n"
    )

    demonstrations = _get_ambiguous_demonstrations()

    current_query_section = f"\nCustomer: {user_query}\nChatbot:"

    return system_message + demonstrations + current_query_section


def chat_with_bot(query: str) -> str:
    """Main function to interact with the chatbot.
    """
    print(f"User: {query}")
    prompt = generate_chatbot_prompt(query)
    response = _simulate_llm_response(prompt)
    print(f"Chatbot: {response}")
    return response


if __name__ == "__main__":
    print("--- E-commerce Chatbot (Ambiguous Demonstrations) ---\n")

    # Test cases demonstrating ambiguous queries
    chat_with_bot("My order is missing a part.")
    print("\n" + "="*70 + "\n")

    chat_with_bot("What's the status?")
    print("\n" + "="*70 + "\n")

    chat_with_bot("I need help with a product.")
    print("\n" + "="*70 + "\n")

    # Test a less ambiguous query to show general behavior
    chat_with_bot("Where is my package? My order number is #12345.")
    print("\n" + "="*70 + "\n")

    chat_with_bot("What is your return policy?")
    print("\n" + "="*70 + "\n")

    chat_with_bot("Can I get a refund for my recent purchase?")
    print("\n" + "="*70 + "\n")

    print("\n--- Chatbot demonstration finished ---")
