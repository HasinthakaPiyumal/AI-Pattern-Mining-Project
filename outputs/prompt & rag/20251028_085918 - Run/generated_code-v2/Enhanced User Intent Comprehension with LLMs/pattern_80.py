
class SmartCustomerSupportAssistant:
    def __init__(self):
        self.intents = {
            "order_status": ["order", "status", "where is my", "track my", "delivery"],
            "return_item": ["return", "item", "product", "send back", "refund"],
            "technical_issue": ["technical", "issue", "problem", "broken", "not working"],
            "billing_inquiry": ["bill", "charge", "invoice", "payment", "statement"],
            "contact_support": ["speak to", "human", "agent", "customer service", "talk to"]
        }
        self.responses = {
            "order_status": "Please provide your order number to check the status of your delivery.",
            "return_item": "To initiate a return, please visit our 'Returns' page or provide your order details.",
            "technical_issue": "Could you please describe your technical issue in more detail, including the product name and model?",
            "billing_inquiry": "For billing inquiries, please provide your account details or invoice number.",
            "contact_support": "Connecting you to a customer service agent now. Please wait while we find an available representative.",
            "ambiguous": "I'm not sure I understand. Could you please rephrase your query or provide more details?",
            "no_intent": "I can help with questions about orders, returns, technical issues, and billing. What can I assist you with today?"
        }

    def _classify_intent(self, query: str) -> str:
        query_lower = query.lower()
        best_intent = "no_intent"
        max_keyword_matches = 0

        for intent, keywords in self.intents.items():
            current_matches = sum(1 for keyword in keywords if keyword in query_lower)
            if current_matches > max_keyword_matches:
                max_keyword_matches = current_matches
                best_intent = intent

        # Simple ambiguity check: if very few keywords match, it might be ambiguous
        if max_keyword_matches == 0:
            return "no_intent"
        elif max_keyword_matches <= 1 and len(query_lower.split()) > 3: # If short query with one match, assume direct intent. Long query with one match, maybe ambiguous.
            # A more sophisticated model would handle true ambiguity better
            return best_intent # For this simplified example, we'll pick the best single match

        return best_intent

    def handle_query(self, query: str) -> str:
        intent = self._classify_intent(query)
        print(f"[DEBUG] Detected intent: {intent}") # For demonstration
        return self.responses.get(intent, self.responses["no_intent"])

# --- Example Usage ---
if __name__ == "__main__":
    assistant = SmartCustomerSupportAssistant()

    print("Hello! I am your Smart Customer Support Assistant. How can I help you today?")
    print("Type 'exit' to quit.")

    while True:
        user_query = input("\nYou: ")
        if user_query.lower() == 'exit':
            print("Assistant: Goodbye!")
            break

        response = assistant.handle_query(user_query)
        print(f"Assistant: {response}")

        # Simulate routing for specific intents (conceptual)
        if assistant._classify_intent(user_query) == "contact_support":
            print("Assistant: Routing you to a live agent. Please hold...")

