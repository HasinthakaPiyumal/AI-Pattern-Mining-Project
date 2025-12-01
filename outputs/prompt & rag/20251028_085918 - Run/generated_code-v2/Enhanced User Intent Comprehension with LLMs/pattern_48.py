class CustomerSupportChatbot:
    def __init__(self):
        self.intents = {
            "order_status": ["order status", "where is my order", "track my package"],
            "account_info": ["my account", "account details", "update personal info"],
            "technical_support": ["technical issue", "troubleshoot", "fix problem"],
            "password_reset": ["reset password", "forgot password"],
            "greeting": ["hello", "hi", "hey"],
            "goodbye": ["bye", "goodbye", "see you"]
        }
        self.tools = {
            "order_status": self._order_lookup,
            "account_info": self._account_update,
            "technical_support": self._technical_troubleshooting,
            "password_reset": self._password_reset
        }
        self.user_context = {}
        self.user_profiles = {}

    def _recognize_intent(self, query):
        query_lower = query.lower()
        for intent, keywords in self.intents.items():
            for keyword in keywords:
                if keyword in query_lower:
                    # Simple entity extraction (e.g., order ID, if present)
                    entities = {}
                    if intent == "order_status":
                        import re
                        match = re.search(r'order\s*#?(\d+)', query_lower)
                        if match: 
                            entities["order_id"] = match.group(1)
                    elif intent == "account_info":
                        if "email" in query_lower: 
                            entities["entity_type"] = "email"
                        elif "phone" in query_lower: 
                            entities["entity_type"] = "phone"
                    return intent, entities, 0.9 # High confidence for direct match
        return "unknown", {}, 0.1 # Low confidence for unknown intent

    def _order_lookup(self, entities):
        order_id = entities.get("order_id")
        if order_id:
            return f"Looking up status for order {order_id}. It is currently out for delivery."
        return "Please provide an order ID for me to look up."

    def _account_update(self, entities):
        entity_type = entities.get("entity_type")
        if entity_type:
            return f"Please confirm the details you wish to update for your {entity_type} in your account."
        return "What specific account information would you like to update?"

    def _technical_troubleshooting(self, entities):
        return "Could you please describe your technical issue in more detail? I can guide you through some troubleshooting steps."

    def _password_reset(self, entities):
        return "I can help you reset your password. Please visit our website and click on the 'Forgot Password' link, or I can send you a reset link to your registered email."

    def _invoke_tool(self, intent, entities):
        tool_function = self.tools.get(intent)
        if tool_function:
            return tool_function(entities)
        return "I'm sorry, I don't have a specific tool for that right now."

    def _manage_dialogue(self, user_id, intent, entities, confidence):
        if intent == "unknown" and confidence < 0.5:
            return "I'm not sure I understand. Could you please rephrase your request?"

        if intent == "order_status" and not entities.get("order_id"):
            return "I can help with order status. What is your order ID?"
        
        if intent == "account_info" and not entities.get("entity_type"):
            return "What specific account information are you trying to update? (e.g., email, phone, address)"

        if intent in self.tools:
            response = self._invoke_tool(intent, entities)
            self._update_personalization(user_id, intent, response) # Log successful interaction
            return response
        elif intent == "greeting":
            return self._get_personalized_response(user_id, intent, "Hello! How can I assist you today?")
        elif intent == "goodbye":
            return self._get_personalized_response(user_id, intent, "Goodbye! Have a great day!")
        
        return "I'm sorry, I cannot fulfill that request at the moment."

    def _update_personalization(self, user_id, intent, response):
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {"history": []}
        self.user_profiles[user_id]["history"].append({"intent": intent, "response": response})

    def _get_personalized_response(self, user_id, intent, default_response):
        # For this simplified version, personalization is just a log and returning default
        # In a real system, this could modify phrasing based on past interactions
        return default_response

    def process_query(self, user_id, query):
        intent, entities, confidence = self._recognize_intent(query)
        response = self._manage_dialogue(user_id, intent, entities, confidence)
        return response

if __name__ == "__main__":
    chatbot = CustomerSupportChatbot()
    print("Welcome to Customer Support! Type 'bye' to exit.")
    user_id = "user123" # Simulate a user ID

    while True:
        user_input = input(f"You ({user_id}): ")
        if user_input.lower() == 'bye':
            print("Chatbot: Goodbye!")
            break

        response = chatbot.process_query(user_id, user_input)
        print(f"Chatbot: {response}")