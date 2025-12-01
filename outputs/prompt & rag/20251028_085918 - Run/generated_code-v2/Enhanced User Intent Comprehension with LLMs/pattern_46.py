class IntentUnderstandingChatbot:
    def __init__(self):
        # Predefined intents and associated keywords for simulation
        self.intents = {
            "order_status": ["order status", "where is my order", "track my package"],
            "return_item": ["return item", "return a product", "send back", "refund"],
            "shipping_address_update": ["change shipping", "update address", "new address"],
            "product_info": ["product details", "tell me about", "specifications"],
            "contact_support": ["speak to human", "customer service", "talk to agent"]
        }

        # Responses or actions associated with each intent
        self.responses = {
            "order_status": "To check your order status, please provide your order number.",
            "return_item": "You can initiate a return by visiting our Returns page and entering your order details.",
            "shipping_address_update": "Please log in to your account and navigate to 'My Orders' to update your shipping address for future orders.",
            "product_info": "Could you please specify which product you are interested in?",
            "contact_support": "Connecting you to a customer service representative now. Please wait.",
            "ambiguous": "I'm not quite sure what you mean. Could you please rephrase or be more specific? For example, are you asking about an order, a return, or something else?",
            "unknown_intent": "I apologize, I don't understand your request. Please ask me about order status, returns, or shipping updates."
        }

    def _simulate_llm_intent_recognition(self, query: str) -> dict:
        """Simulates an LLM recognizing intent from a user query."""
        query_lower = query.lower()
        matched_intents = []

        for intent_name, keywords in self.intents.items():
            for keyword in keywords:
                if keyword in query_lower:
                    matched_intents.append(intent_name)
                    break # Match found for this intent, move to next intent
        
        if not matched_intents:
            return {"intent": "unknown_intent", "confidence": 0.0}
        elif len(set(matched_intents)) > 1: # Check for multiple unique intents identified
            return {"intent": "ambiguous", "confidence": 0.3}
        else:
            # Simulate a higher confidence for a clear match
            return {"intent": matched_intents[0], "confidence": 0.9}

    def process_query(self, user_query: str) -> str:
        """Processes a user query to understand intent and generate a response."""
        recognition_result = self._simulate_llm_intent_recognition(user_query)
        intent = recognition_result["intent"]
        confidence = recognition_result["confidence"]

        if confidence < 0.5 and intent != "ambiguous": # Low confidence on a specific intent
             return self.responses["ambiguous"]
        else:
            return self.responses[intent]

# --- Example Usage ---
if __name__ == "__main__":
    chatbot = IntentUnderstandingChatbot()

    print("\n--- Chatbot Ready ---\n")

    # Test clear intent
    query1 = "What's the status of my order?"
    print(f"User: {query1}")
    print(f"Chatbot: {chatbot.process_query(query1)}\n")

    # Test another clear intent
    query2 = "I want to return a purchase."
    print(f"User: {query2}")
    print(f"Chatbot: {chatbot.process_query(query2)}\n")

    # Test ambiguous query
    query3 = "I need help with an item."
    print(f"User: {query3}")
    print(f"Chatbot: {chatbot.process_query(query3)}\n")

    # Test vague query potentially leading to ambiguity (if keywords overlap more)
    query4 = "Can I change something on my order?"
    print(f"User: {query4}")
    print(f"Chatbot: {chatbot.process_query(query4)}\n")

    # Test unknown intent
    query5 = "Tell me a joke."
    print(f"User: {query5}")
    print(f"Chatbot: {chatbot.process_query(query5)}\n")

    # Test an update shipping query
    query6 = "I need to update my shipping address."
    print(f"User: {query6}")
    print(f"Chatbot: {chatbot.process_query(query6)}\n")

    # Test a query that might trigger contact support
    query7 = "I need to talk to someone."
    print(f"User: {query7}")
    print(f"Chatbot: {chatbot.process_query(query7)}\n")