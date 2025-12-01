class SmartCustomerSupportAssistant:
    def __init__(self):
        self.personalized_rules = {
            "where is my stuff": "track_order",
            "my package is late": "track_order",
            "how to send something back": "initiate_return",
            "i want to return an item": "initiate_return",
            "change address for delivery": "change_shipping_address",
            "update my delivery location": "change_shipping_address"
        }

        self.intent_patterns = {
            "track order": "track_order",
            "order status": "track_order",
            "return": "initiate_return",
            "send back": "initiate_return",
            "shipping address": "change_shipping_address",
            "delivery address": "change_shipping_address",
            "product inquiry": "product_inquiry",
            "item details": "product_inquiry",
            "billing issue": "billing_issue",
            "payment problem": "billing_issue"
        }

    def _classify_intent(self, query):
        query = query.lower()

        for phrase, intent in self.personalized_rules.items():
            if phrase in query:
                return intent

        for keyword, intent in self.intent_patterns.items():
            if keyword in query:
                return intent

        return "general_inquiry"

    def handle_query(self, query):
        intent = self._classify_intent(query)
        response = ""

        if intent == "track_order":
            response = "To track your order, please visit our 'Order Tracking' page and enter your order number."
        elif intent == "initiate_return":
            response = "You can initiate a return by visiting our 'Returns & Refunds' section. Please ensure you have your order details handy."
        elif intent == "change_shipping_address":
            response = "To change your shipping address for an existing order, please contact our support team immediately or update it in your account settings if the order hasn't shipped."
        elif intent == "product_inquiry":
            response = "Please provide more details about the product you are interested in, and I can connect you with a specialist or provide information."
        elif intent == "billing_issue":
            response = "For billing issues, please go to your account's 'Billing History' or contact our finance department directly for assistance."
        else: # general_inquiry
            response = "I'm not entirely sure what you're asking. Could you please rephrase your request or provide more details?"
        
        return response

if __name__ == '__main__':
    assistant = SmartCustomerSupportAssistant()

    print("\n--- Smart Customer Support Assistant --- ")
    print("Type 'exit' to quit.")

    while True:
        user_query = input("\nUser: ")
        if user_query.lower() == 'exit':
            print("Assistant: Goodbye!")
            break
        
        assistant_response = assistant.handle_query(user_query)
        print(f"Assistant: {assistant_response}")