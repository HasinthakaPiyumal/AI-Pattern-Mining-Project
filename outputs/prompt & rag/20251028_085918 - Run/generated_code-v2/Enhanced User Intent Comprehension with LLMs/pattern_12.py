class CustomerSupportAssistant:
    def __init__(self):
        self.knowledge_base = {
            "product_info": {
                "laptop": "Our latest laptop features an Intel i7 processor, 16GB RAM, and a 512GB SSD. It comes with a 1-year warranty.",
                "smartphone": "The new smartphone boasts a 6.7-inch OLED display, a triple camera system, and a 4000mAh battery.",
                "headphone": "These noise-cancelling headphones offer superior sound quality and up to 30 hours of battery life."
            },
            "order_status_info": {
                "12345": {"status": "Shipped", "eta": "2 days"},
                "67890": {"status": "Processing", "eta": "4-5 days"}
            }
        }
        self.intents = {
            "order_status": ["where is my order", "order status", "track my package", "my order is late"],
            "return_item": ["return an item", "how to return", "initiate a return", "send back a product"],
            "product_info": ["tell me about", "product details", "specifications of", "info on"],
            "greeting": ["hello", "hi", "hey", "good morning"],
            "escalate": ["talk to a human", "speak to a representative", "customer service", "urgent help"]
        }

    def _understand_intent(self, query):
        query_lower = query.lower()
        for intent, keywords in self.intents.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return intent
        return "unknown"

    def _get_order_status(self, order_id):
        if order_id in self.knowledge_base["order_status_info"]:
            order = self.knowledge_base["order_status_info"][order_id]
            return f"Your order {order_id} is currently {order['status']} and is expected in {order['eta']}."
        return "I couldn't find details for that order ID. Please double-check it."

    def _get_product_info(self, product_name):
        product_name_lower = product_name.lower()
        for p_name, info in self.knowledge_base["product_info"].items():
            if p_name in product_name_lower:
                return info
        return f"I don't have detailed information for '{product_name}'. Can you be more specific?"

    def handle_query(self, query):
        intent = self._understand_intent(query)

        if intent == "greeting":
            return "Hello! How can I assist you with your e-commerce needs today?"

        elif intent == "order_status":
            import re
            match = re.search(r'\b(?:order|id|number)\s*#?(\d+)\b', query, re.IGNORECASE)
            if match:
                order_id = match.group(1)
                return self._get_order_status(order_id)
            else:
                return "To check your order status, please provide your order ID."

        elif intent == "return_item":
            return "To initiate a return, please visit our 'Returns & Refunds' section on the website or provide your order ID for further assistance."

        elif intent == "product_info":
            import re
            match = re.search(r'(?:about|details about|info on)\s+(.+?)(?:\?|$)', query, re.IGNORECASE)
            if match:
                product_name = match.group(1).strip()
                return self._get_product_info(product_name)
            else:
                return "Which product are you interested in? Please tell me the product name."

        elif intent == "escalate":
            return "I understand. Let me connect you to a human agent who can help you further. Please wait a moment."

        else:
            return "I'm sorry, I didn't quite understand that. Could you please rephrase or be more specific?"

if __name__ == "__main__":
    assistant = CustomerSupportAssistant()
    print("E-commerce Customer Support Assistant (Type 'exit' to quit)")
    while True:
        user_query = input("You: ")
        if user_query.lower() == 'exit':
            break
        response = assistant.handle_query(user_query)
        print(f"Assistant: {response}")
