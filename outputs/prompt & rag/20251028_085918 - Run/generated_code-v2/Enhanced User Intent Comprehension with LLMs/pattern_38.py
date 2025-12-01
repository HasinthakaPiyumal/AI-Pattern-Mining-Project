import random

class MockIntentClassifier:
    def __init__(self):
        self.intents = {
            "order_status": ["where is my order", "track my package", "what's my order status", "delivery update"],
            "return_request": ["i want to return something", "how to return an item", "start a return", "return policy"],
            "product_inquiry": ["tell me about this product", "product details", "specifications", "is this in stock"],
            "shipping_info": ["shipping cost", "delivery options", "how long does shipping take"],
            "account_management": ["change my password", "update my address", "my account details"],
            "general_query": ["hello", "hi", "how are you", "help", "support"]
        }
        self.confidence_threshold = 0.6

    def classify_intent(self, query):
        query = query.lower()
        for intent, phrases in self.intents.items():
            for phrase in phrases:
                if phrase in query:
                    # Simulate confidence based on match strength
                    if len(query) < len(phrase) + 5:
                        return intent, 0.95  # High confidence for close match
                    else:
                        return intent, 0.75  # Medium confidence
        return "general_query", 0.4 # Low confidence for unknown or general queries

class MockToolIntegration:
    def execute_tool(self, intent, entities=None, user_id=None):
        if intent == "order_status":
            return f"Checking order status for user {user_id}. Your order is currently out for delivery."
        elif intent == "return_request":
            return f"Initiating a return for user {user_id}. Please check your email for instructions."
        elif intent == "product_inquiry":
            return "Please specify the product you are interested in. I can provide details."
        elif intent == "shipping_info":
            return "Standard shipping usually takes 3-5 business days."
        elif intent == "account_management":
            return "I can help you with account settings. What specifically would you like to do?"
        else:
            return "I'm not sure how to handle that request right now. Can you rephrase?"

class DialogueManager:
    def __init__(self, intent_classifier, tool_integration):
        self.intent_classifier = intent_classifier
        self.tool_integration = tool_integration
        self.user_contexts = {}
        self.clarification_questions = {
            "order_status": "Can you provide your order number or the email associated with the order?",
            "return_request": "What item are you looking to return, and what is the reason?",
            "product_inquiry": "Which product are you interested in? Please be more specific.",
            "general_query": "I'm sorry, I didn't quite understand. Could you please elaborate or ask in a different way?"
        }

    def get_user_context(self, user_id):
        if user_id not in self.user_contexts:
            self.user_contexts[user_id] = {"history": [], "current_intent": None, "last_query": None}
        return self.user_contexts[user_id]

    def update_user_context(self, user_id, key, value):
        self.user_contexts[user_id][key] = value

    def generate_response(self, user_id, query):
        user_context = self.get_user_context(user_id)
        user_context["history"].append(query)
        user_context["last_query"] = query

        intent, confidence = self.intent_classifier.classify_intent(query)

        if confidence < self.intent_classifier.confidence_threshold:
            # Low confidence, try to clarify or use previous intent if relevant
            if user_context["current_intent"] and user_context["current_intent"] != "general_query":
                # Try to re-confirm previous intent or ask a general clarification
                return self.clarification_questions.get(user_context["current_intent"], self.clarification_questions["general_query"])
            else:
                return self.clarification_questions["general_query"]
        
        user_context["current_intent"] = intent
        response = self.tool_integration.execute_tool(intent, user_id=user_id)
        
        # Simple personalization based on hypothetical user history
        if "thank you" in query.lower() and len(user_context["history"]) > 2:
            response += " Is there anything else I can help you with today?"

        return response

class ChatbotUI:
    def __init__(self, dialogue_manager):
        self.dialogue_manager = dialogue_manager
        self.user_id = "user123" # For simplicity, a fixed user_id for the prototype

    def start_chat(self):
        print("Welcome to the E-commerce Customer Support Chatbot! Type 'exit' to end the conversation.")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                print("Chatbot: Goodbye!")
                break
            
            response = self.dialogue_manager.generate_response(self.user_id, user_input)
            print(f"Chatbot: {response}")

if __name__ == "__main__":
    intent_classifier = MockIntentClassifier()
    tool_integration = MockToolIntegration()
    dialogue_manager = DialogueManager(intent_classifier, tool_integration)
    chatbot_ui = ChatbotUI(dialogue_manager)
    chatbot_ui.start_chat()