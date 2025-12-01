def rephrase_query(customer_query: str) -> str:
    if "order status" in customer_query.lower():
        return f"Could you please elaborate on the order status for which you are looking for an update? Specifically, what is the order number or item you are inquiring about? Original query: {customer_query}"
    elif "return policy" in customer_query.lower() or "refund" in customer_query.lower():
        return f"To provide the most accurate information on our return and refund policy, could you specify the item you wish to return or the reason for a refund? Original query: {customer_query}"
    elif "shipping options" in customer_query.lower() or "delivery time" in customer_query.lower():
        return f"Regarding your query about shipping and delivery, are you interested in options for a new purchase, or tracking an existing order? What is your general location? Original query: {customer_query}"
    else:
        return f"I understand you are asking about: '{customer_query}'. To help me understand better, could you please provide more context or details about what you need assistance with?"

def generate_response(rephrased_query: str) -> str:
    if "order number" in rephrased_query.lower() or "item you are inquiring about" in rephrased_query.lower():
        return "Please provide your order number and I can check the status for you. You can find your order number in your confirmation email."
    elif "item you wish to return" in rephrased_query.lower() or "reason for a refund" in rephrased_query.lower():
        return "Our standard return policy allows returns within 30 days of purchase for most items, provided they are in new and unused condition. For detailed information and to initiate a return, please visit our Returns & Refunds page on the website."
    elif "options for a new purchase" in rephrased_query.lower() or "tracking an existing order" in rephrased_query.lower():
        return "We offer various shipping options including standard, expedited, and express delivery. Estimated delivery times vary by location. For new purchases, you can see options at checkout. For existing orders, please provide your tracking number."
    else:
        return "Thank you for clarifying. For further assistance, please feel free to ask another question or contact our live support during business hours."

def customer_support_chatbot(customer_query: str):
    print(f"Original Customer Query: {customer_query}")
    rephrased_query = rephrase_query(customer_query)
    print(f"Internally Rephrased Query: {rephrased_query}")
    final_response = generate_response(rephrased_query)
    print(f"Chatbot Final Response: {final_response}")

if __name__ == "__main__":
    print("--- Chatbot Interaction 1 ---")
    customer_support_chatbot("What is my order status?")
    print("\n--- Chatbot Interaction 2 ---")
    customer_support_chatbot("How can I return an item?")
    print("\n--- Chatbot Interaction 3 ---")
    customer_support_chatbot("Tell me about shipping.")
    print("\n--- Chatbot Interaction 4 ---")
    customer_support_chatbot("I have a general question.")