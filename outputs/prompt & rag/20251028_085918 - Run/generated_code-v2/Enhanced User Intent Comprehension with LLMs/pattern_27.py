class IntentRecognizer:
    def __init__(self):
        self.known_intents = {
            "track_order": ["where is my order", "order status", "when will my delivery arrive", "my order is late"],
            "initiate_return": ["i want to return something", "how do i return an item", "start a return"],
            "product_inquiry": ["tell me about product X", "specifications of item Y", "product details"],
            "customer_service": ["speak to an agent", "i need help", "talk to someone"],
        }
        self.intent_map = {}
        for intent, phrases in self.known_intents.items():
            for phrase in phrases:
                self.intent_map[phrase.lower()] = intent

    def predict_intent(self, query: str) -> dict:
        query_lower = query.lower()
        
        for phrase, intent in self.intent_map.items():
            if phrase in query_lower:
                return {"intent": intent, "confidence": 0.95, "is_ambiguous": False}
        
        potential_intents = []
        if "order" in query_lower:
            potential_intents.append("track_order")
        if "return" in query_lower:
            potential_intents.append("initiate_return")

        if len(potential_intents) == 1:
            return {"intent": potential_intents[0], "confidence": 0.7, "is_ambiguous": False}
        elif len(potential_intents) > 1:
            return {"intent": "unknown", "confidence": 0.4, "is_ambiguous": True, "potential_intents": potential_intents}

        return {"intent": "unknown", "confidence": 0.1, "is_ambiguous": True}

    def add_training_data(self, query: str, intent: str):
        self.intent_map[query.lower()] = intent
        print(f"[PERSONALIZED LEARNING] Added \'{query}\' mapped to \'{intent}\'")

class ActionExecutor:
    def __init__(self):
        pass

    def execute_action(self, intent: str, details: dict = None) -> str:
        if details is None:
            details = {}

        if intent == "track_order":
            order_id = details.get("order_id", "your most recent order")
            return f"[ACTION] Contacting order management system to track {order_id}. Please wait a moment..."
        elif intent == "initiate_return":
            item = details.get("item", "the item in question")
            return f"[ACTION] Initiating the return process for {item}. Please provide more details about the item and reason for return."
        elif intent == "product_inquiry":
            product_name = details.get("product_name", "the requested product")
            return f"[ACTION] Fetching details for {product_name}. What specific information are you looking for?"
        elif intent == "customer_service":
            return "[ACTION] Connecting you to a customer service representative. Please hold."
        else:
            return "[ACTION] I\'m sorry, I don\'t know how to handle that request currently."

class CustomerSupportAssistant:
    def __init__(self):
        self.intent_recognizer = IntentRecognizer()
        self.action_executor = ActionExecutor()

    def process_query(self, query: str) -> str:
        intent_result = self.intent_recognizer.predict_intent(query)
        
        if intent_result["is_ambiguous"]:
            if intent_result["intent"] == "unknown" and "potential_intents" in intent_result:
                response = "I\'m not sure if you want to " + " or ".join(intent_result["potential_intents"]).replace("_", " ") + ". Could you please clarify?"
            else:
                response = "I\'m having a bit of trouble understanding your request. Could you rephrase it or be more specific?"
        else:
            intent = intent_result["intent"]
            response = self.action_executor.execute_action(intent)
        
        return response

    def provide_personalized_feedback(self, query: str, correct_intent: str):
        self.intent_recognizer.add_training_data(query, correct_intent)
        return f"Thank you for the feedback! I\'ve noted that \'{query}\' should map to \'{correct_intent}\' for future reference."

if __name__ == "__main__":
    assistant = CustomerSupportAssistant()
    print("Welcome to the E-commerce Customer Support Assistant! Type 'exit' to quit.")
    print("You can also type 'feedback:<query>:<intent>' to help me learn.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Assistant: Goodbye!")
            break
        elif user_input.lower().startswith('feedback:'):
            parts = user_input.split(':', 2)
            if len(parts) == 3:
                query_for_feedback = parts[1]
                correct_intent = parts[2]
                print("Assistant: " + assistant.provide_personalized_feedback(query_for_feedback, correct_intent))
            else:
                print("Assistant: Invalid feedback format. Use 'feedback:<query>:<intent>'")
        else:
            response = assistant.process_query(user_input)
            print(f"Assistant: {response}")