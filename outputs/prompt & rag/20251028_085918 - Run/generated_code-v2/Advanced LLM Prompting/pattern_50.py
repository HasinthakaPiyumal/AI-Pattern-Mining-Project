few_shot_examples = [
    {"query": "My order #12345 hasn't arrived yet.", "response": "I can help with that! Please provide your full name and shipping address so I can check the status of your order #12345."},
    {"query": "How do I return a product?", "response": "To return a product, please visit our returns page at www.example.com/returns and follow the instructions. You'll typically need your order number and the reason for the return."},
    {"query": "What are the features of the new Model X phone?", "response": "The new Model X phone boasts a 6.7-inch OLED display, a triple-lens camera system, and a 5000mAh battery for extended use."},
    {"query": "I received a damaged item, what should I do?", "response": "We apologize for the inconvenience. Please send us a photo of the damaged item along with your order number to support@example.com, and we'll arrange a replacement."}
]

def simulate_llm_response(prompt: str) -> str:
    if "order status" in prompt.lower() or "not arrived" in prompt.lower() or "order #" in prompt.lower():
        import re
        match = re.search(r"order #(\\d+)", prompt.lower())
        if match:
            order_id = match.group(1)
            return f"Based on the examples, for your order {order_id}, please provide your full name and shipping address so I can check its status." 
        return "Based on the examples, it seems you're asking about an order. Please provide your order number for me to assist you."
    elif "return a product" in prompt.lower() or "how to return" in prompt.lower():
        return "Following the examples, if you need to return a product, please visit our returns page at www.example.com/returns and provide your order details."
    elif "features" in prompt.lower() and "model x" in prompt.lower():
        return "Looking at similar examples, Model X features include a 6.7-inch OLED display, a triple-lens camera system, and a 5000mAh battery. What specific features are you interested in?"
    elif "damaged item" in prompt.lower() or "received a damaged" in prompt.lower():
        return "Apologies for the damaged item. Following the examples, please send a photo of the damaged item with your order number to support@example.com for a replacement."
    return "Thank you for your query. Based on the examples provided, I understand you need assistance. Could you please rephrase or provide more details?"

def construct_few_shot_prompt(user_query: str, examples: list) -> str:
    prefix = "The following are examples of customer support queries and helpful responses:\n\n"
    example_str = ""
    for ex in examples:
        example_str += f"Customer: {ex['query']}\n"
        example_str += f"Chatbot: {ex['response']}\n\n"
    suffix = f"Customer: {user_query}\nChatbot:"
    return prefix + example_str + suffix

def run_chatbot():
    print("Welcome to the Few-Shot Customer Support Chatbot!")
    print("Type 'exit' to end the chat.")

    while True:
        user_input = input("\nCustomer: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break

        full_prompt = construct_few_shot_prompt(user_input, few_shot_examples)
        chatbot_response = simulate_llm_response(full_prompt)
        print(f"Chatbot: {chatbot_response}")

if __name__ == "__main__":
    run_chatbot()