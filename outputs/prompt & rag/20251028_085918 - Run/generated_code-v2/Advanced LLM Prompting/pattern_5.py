def LLM_Simulator(prompt: str) -> str:
    if "order status" in prompt.lower() or "where is my order" in prompt.lower():
        return "Your order #12345 is currently being processed and is expected to ship within 2-3 business days."
    elif "return policy" in prompt.lower() or "how to return" in prompt.lower():
        return "Our return policy allows returns within 30 days of purchase, provided the item is in its original condition. Please visit our returns page for detailed instructions."
    elif "product information" in prompt.lower() or "about product" in prompt.lower():
        return "Please specify which product you are interested in, and I can provide more details."
    else:
        return "I'm an e-commerce customer support assistant. Please ask your question regarding products, orders, shipping, or returns. If I don't know the answer, I will let you know."

def zero_shot_prompt_builder(customer_query: str) -> str:
    instructions = (
        "You are an e-commerce customer support assistant. "
        "Answer questions concisely and professionally. "
        "Focus on providing helpful information related to products, orders, shipping, or returns. "
        "If you don't know the answer to a specific question, state that you cannot assist with that particular query at this moment."
    )
    return f"{instructions}\n\nCustomer Query: {customer_query}"

def customer_support_assistant(customer_query: str) -> str:
    prompt = zero_shot_prompt_builder(customer_query)
    response = LLM_Simulator(prompt)
    return response

if __name__ == "__main__":
    print("Welcome to the ZeroShot E-commerce Customer Support!\n")
    queries = [
        "Where is my order?",
        "How can I return an item?",
        "Tell me about the new smartphone.",
        "What is your shipping policy?",
        "Can I get a discount?",
        "I need help with my account settings."
    ]

    for query in queries:
        print(f"Customer: {query}")
        assistant_response = customer_support_assistant(query)
        print(f"Assistant: {assistant_response}\n")