from transformers import pipeline

class SmartCustomerSupportAssistant:
    def __init__(self, model_name="distilbert-base-uncased-distilled-squad"):
        self.nlp_pipeline = pipeline("question-answering", model=model_name)
        self.intents = [
            "Order Status", 
            "Return/Exchange", 
            "Product Information", 
            "Technical Support", 
            "Billing Inquiry", 
            "Escalate to Human"
        ]
        # Simulate a simple user data store for personalization
        self.user_data_store = {
            "user_123": {"orders": ["#ORD-001", "#ORD-005"], "preferences": {"language": "en"}},
            "user_456": {"orders": ["#ORD-002"], "preferences": {"language": "es"}},
        }

    def _classify_intent(self, query: str) -> str:
        # A more sophisticated intent classification would use a fine-tuned model
        # or more advanced prompt engineering. For this example, we'll use keyword matching 
        # and a general QA pipeline for a simplistic approach to demonstrate.
        query_lower = query.lower()
        if "order" in query_lower or "shipment" in query_lower or "delivery" in query_lower:
            return "Order Status"
        elif "return" in query_lower or "exchange" in query_lower or "faulty" in query_lower:
            return "Return/Exchange"
        elif "product" in query_lower or "specifications" in query_lower or "details" in query_lower:
            return "Product Information"
        elif "technical" in query_lower or "bug" in query_lower or "issue" in query_lower:
            return "Technical Support"
        elif "bill" in query_lower or "invoice" in query_lower or "charge" in query_lower:
            return "Billing Inquiry"
        elif "speak to human" in query_lower or "agent" in query_lower or "escalate" in query_lower:
            return "Escalate to Human"
        else:
            # Fallback to a generic intent or try to extract info with QA
            return "General Inquiry"

    def _generate_response(self, intent: str, query: str, user_id: str = None) -> str:
        user_context = self.user_data_store.get(user_id, {})
        
        if intent == "Order Status":
            orders = user_context.get("orders", [])
            if orders:
                return f"To check your order status, please provide an order number. Your recent orders include {', '.join(orders)}."
            else:
                return "Please provide your order number to check its status."
        elif intent == "Return/Exchange":
            return "For returns or exchanges, please visit our returns portal at [link_to_portal] or provide your order number."
        elif intent == "Product Information":
            try:
                # Use the QA pipeline to extract information if possible
                context = "Our products include a wide range of electronics, clothing, and home goods. Each product page has detailed specifications. What product are you interested in?"
                answer = self.nlp_pipeline(question=query, context=context)['answer']
                return f"Based on your query: {answer}. Can you specify which product you're asking about?"
            except Exception:
                return "I can help with product information. Could you please specify which product you're interested in?"
        elif intent == "Technical Support":
            return "For technical support, please describe your issue in more detail, and we will connect you with a specialist."
        elif intent == "Billing Inquiry":
            return "For billing inquiries, please have your invoice number ready, and we can assist you. Would you like to check a recent charge?"
        elif intent == "Escalate to Human":
            return "I understand. I'm connecting you to a human agent now. Please wait a moment."
        elif intent == "General Inquiry":
            try:
                context = "Welcome to our customer support. We can assist with order status, returns, product information, technical issues, and billing. How can I help you today?"
                answer = self.nlp_pipeline(question=query, context=context)['answer']
                if answer:
                    return f"I'm not entirely sure, but based on your query, it seems to be about {answer}. Can you elaborate?"
                else:
                    return "I'm not entirely sure how to help with that. Can you rephrase or provide more details?"
            except Exception:
                return "I'm not entirely sure how to help with that. Can you rephrase or provide more details?"
        else:
            return "I'm sorry, I couldn't understand your request. Could you please try again?"

    def process_query(self, query: str, user_id: str = None) -> str:
        intent = self._classify_intent(query)
        response = self._generate_response(intent, query, user_id)
        return response

if __name__ == "__main__":
    assistant = SmartCustomerSupportAssistant()

    print("--- Test Queries ---")
    
    # Test 1: Order Status
    user_query_1 = "Where is my stuff?"
    print(f"User (user_123): {user_query_1}")
    print(f"Assistant: {assistant.process_query(user_query_1, user_id='user_123')}")
    print("\n")

    # Test 2: Product Information
    user_query_2 = "Tell me about the new smart speaker."
    print(f"User: {user_query_2}")
    print(f"Assistant: {assistant.process_query(user_query_2)}")
    print("\n")

    # Test 3: Return/Exchange
    user_query_3 = "I want to return an item."
    print(f"User: {user_query_3}")
    print(f"Assistant: {assistant.process_query(user_query_3)}")
    print("\n")

    # Test 4: Escalate to Human
    user_query_4 = "I need to speak to someone right now."
    print(f"User: {user_query_4}")
    print(f"Assistant: {assistant.process_query(user_query_4)}")
    print("\n")

    # Test 5: Vague query (General Inquiry)
    user_query_5 = "What can you do?"
    print(f"User: {user_query_5}")
    print(f"Assistant: {assistant.process_query(user_query_5)}")
    print("\n")

    # Test 6: Billing Inquiry
    user_query_6 = "My last bill seems off."
    print(f"User: {user_query_6}")
    print(f"Assistant: {assistant.process_query(user_query_6)}")
    print("\n")