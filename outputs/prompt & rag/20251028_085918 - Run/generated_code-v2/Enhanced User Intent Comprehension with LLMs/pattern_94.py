from transformers import pipeline

class SmartCustomerSupportAssistant:
    def __init__(self):
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        self.candidate_labels = [
            "order_status",
            "returns_policy",
            "product_information",
            "technical_support",
            "account_management"
        ]
        self.intent_actions = {
            "order_status": self._get_order_status,
            "returns_policy": self._get_returns_policy,
            "product_information": self._get_product_information,
            "technical_support": self._get_technical_support,
            "account_management": self._get_account_management,
        }
        self.response_templates = {
            "order_status": "I can help you check your order status. Please provide your order number.",
            "returns_policy": "Our returns policy allows returns within 30 days of purchase. You can find more details on our website.",
            "product_information": "To provide accurate product information, could you please specify the product you are interested in?",
            "technical_support": "For technical support, please describe your issue in detail, and we'll connect you with a specialist.",
            "account_management": "Regarding account management, what specific task would you like to perform, such as updating details or changing a password?",
            "default": "I'm not entirely sure how to help with that. Could you please rephrase your query or ask something else?"
        }

    def _get_order_status(self, query):
        return "Simulating fetching order status..."

    def _get_returns_policy(self, query):
        return "Simulating providing returns policy..."

    def _get_product_information(self, query):
        return "Simulating retrieving product information..."

    def _get_technical_support(self, query):
        return "Simulating connecting to technical support..."
        
    def _get_account_management(self, query):
        return "Simulating account management actions..."

    def predict_intent(self, query):
        results = self.classifier(query, self.candidate_labels, multi_label=False)
        # The `scores` are sorted in descending order, so the first label is the most probable.
        predicted_intent = results["labels"][0]
        confidence = results["scores"][0]
        return predicted_intent, confidence

    def execute_action(self, intent, query):
        action_function = self.intent_actions.get(intent)
        if action_function:
            return action_function(query)
        return None

    def generate_response(self, intent, action_result):
        if action_result:
            return self.response_templates.get(intent, self.response_templates["default"])
        return self.response_templates["default"]

    def process_query(self, query):
        print(f"User query: '{query}'")
        predicted_intent, confidence = self.predict_intent(query)
        print(f"Predicted intent: '{predicted_intent}' with confidence: {confidence:.2f}")

        if confidence > 0.7:  # Threshold for confidence
            action_result = self.execute_action(predicted_intent, query)
            response = self.generate_response(predicted_intent, action_result)
        else:
            response = self.response_templates["default"]
        
        return response

def main():
    assistant = SmartCustomerSupportAssistant()
    print("Smart Customer Support Assistant. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        
        response = assistant.process_query(user_input)
        print(f"Assistant: {response}")

if __name__ == "__main__":
    main()