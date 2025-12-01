class Chatbot:
    def __init__(self):
        self.intent_keywords = {
            "order_status": ["order", "status", "where is my", "tracking"],
            "product_inquiry": ["product", "info", "details", "specifications", "about"],
            "return_request": ["return", "item", "exchange", "send back"],
            "payment_issue": ["payment", "charge", "bill", "invoice", "paid"],
            "technical_support": ["technical", "issue", "error", "broken", "help with"],
            "greeting": ["hello", "hi", "hey"],
            "goodbye": ["bye", "goodbye", "see you"]
        }
        self.knowledge_base = {
            "order_status": "Please provide your order number so I can check its status for you.",
            "product_inquiry": "What product are you interested in? I can provide details like features, price, and availability.",
            "return_request": "To initiate a return, please visit our returns page on the website or provide your order number and the reason for return.",
            "payment_issue": "Could you please specify the payment issue? For security reasons, I can\'t access your payment details directly, but I can guide you.",
            "technical_support": "I understand you\'re facing a technical issue. Please describe it in more detail, and I\'ll try to assist or connect you with a specialist.",
            "greeting": "Hello! How can I assist you today?",
            "goodbye": "Goodbye! Have a great day!",
            "ambiguous": "I\'m not quite sure I understand. Could you please rephrase your request or provide more details?",
            "clarification_order_num": "Could you please provide your order number?",
            "clarification_product_name": "Which product are you asking about?"
        }
        self.user_data = {}
        self.conversation_history = {}
        self.current_user_id = "guest"
        self.current_state = {}

    def _identify_intent(self, query):
        detected_intents = []
        for intent, keywords in self.intent_keywords.items():
            if any(keyword in query.lower() for keyword in keywords):
                detected_intents.append(intent)

        if len(detected_intents) == 1:
            return detected_intents[0]
        elif len(detected_intents) > 1:
            return "ambiguous"
        else:
            return "unknown"

    def _get_personalized_response(self, intent):
        user_info = self.user_data.get(self.current_user_id, {})
        if intent == "greeting" and user_info.get("name"):
            return f"Welcome back, {user_info['name']}! How can I help you today?"
        return self.knowledge_base.get(intent, self.knowledge_base["ambiguous"])

    def _handle_clarification(self, intent, query):
        if intent == "ambiguous":
            return self.knowledge_base["ambiguous"]
        if intent == "order_status" and not any(char.isdigit() for char in query):
            return self.knowledge_base["clarification_order_num"]
        if intent == "product_inquiry" and not any(word in query.lower() for word in ["phone", "laptop", "shoe", "item"]):
            return self.knowledge_base["clarification_product_name"]
        return None

    def _human_handoff(self, reason, history):
        print(f"Chatbot: Transferring to a human agent. Reason: {reason}")
        print(f"Chatbot: Conversation history: {history}")
        return "I\'m connecting you with a human agent who can provide further assistance."

    def process_query(self, query, user_id="guest"):
        self.current_user_id = user_id
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
        self.conversation_history[user_id].append(f"User: {query}")

        if user_id not in self.current_state:
            self.current_state[user_id] = {"awaiting_input_for_intent": None}

        if self.current_state[user_id]["awaiting_input_for_intent"]:
            intent_to_clarify = self.current_state[user_id]["awaiting_input_for_intent"]
            if intent_to_clarify == "order_status" and any(char.isdigit() for char in query):
                self.current_state[user_id]["awaiting_input_for_intent"] = None
                response = f"Thanks for the order number '{query}'. Let me check the status for you... (Simulated lookup: Your order is on its way!)"
                self.conversation_history[user_id].append(f"Chatbot: {response}")
                return response
            elif intent_to_clarify == "product_inquiry":
                self.current_state[user_id]["awaiting_input_for_intent"] = None
                response = f"You\'re asking about '{query}'. (Simulated: This product has great features and is available!)"
                self.conversation_history[user_id].append(f"Chatbot: {response}")
                return response

        intent = self._identify_intent(query)
        response = ""

        if intent == "ambiguous":
            response = self._handle_clarification(intent, query)
            self.conversation_history[user_id].append(f"Chatbot: {response}")
            return response
        elif intent == "unknown":
            response = self._human_handoff("Couldn\'t determine intent", self.conversation_history[user_id])
            self.conversation_history[user_id].append(f"Chatbot: {response}")
            return response
        elif intent == "order_status" and not any(char.isdigit() for char in query):
            self.current_state[user_id]["awaiting_input_for_intent"] = "order_status"
            response = self._handle_clarification(intent, query)
            self.conversation_history[user_id].append(f"Chatbot: {response}")
            return response
        elif intent == "product_inquiry" and not any(word in query.lower() for word in ["phone", "laptop", "shoe", "item"]):
             self.current_state[user_id]["awaiting_input_for_intent"] = "product_inquiry"
             response = self._handle_clarification(intent, query)
             self.conversation_history[user_id].append(f"Chatbot: {response}")
             return response
        else:
            response = self._get_personalized_response(intent)

        self.conversation_history[user_id].append(f"Chatbot: {response}")
        return response

    def set_user_info(self, user_id, name=None):
        if user_id not in self.user_data:
            self.user_data[user_id] = {}
        if name:
            self.user_data[user_id]["name"] = name

if __name__ == "__main__":
    chatbot = Chatbot()

    chatbot.set_user_info("user123", name="Alice")

    print("Chatbot: Hello! How can I assist you today?")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break

        response = chatbot.process_query(user_input, user_id="user123")
        print(f"Chatbot: {response}")