class NLUModule:
    def __init__(self):
        self.intent_keywords = {
            "Order Status": ["order status", "where is my order", "track my order", "delivery time"],
            "Return/Refund": ["return", "refund", "item back", "send back", "wrong item"],
            "Technical Support": ["technical issue", "not working", "bug", "error", "help with product"],
            "Product Inquiry": ["about product", "product details", "specifications", "features"],
            "General Help": ["help", "support", "question", "assist"]
        }

    def process_query(self, query):
        lower_query = query.lower()
        detected_intents = []
        for intent, keywords in self.intent_keywords.items():
            for keyword in keywords:
                if keyword in lower_query:
                    detected_intents.append(intent)
                    break
        
        # Simple entity extraction placeholder (e.g., assuming order IDs are numbers)
        entities = {}
        if "order" in lower_query:
            import re
            order_id_match = re.search(r"order #?(\d+)", lower_query)
            if order_id_match:
                entities["order_id"] = order_id_match.group(1)
        
        return list(set(detected_intents)), entities

class DialogueManager:
    def __init__(self):
        self.state_tracker = {}
        self.clarification_questions = {
            "Order Status": "Are you looking for the status of a specific order, or general shipping information?",
            "Return/Refund": "Are you trying to return an item, or inquire about a refund for a past return?",
            "Technical Support": "Could you please specify which product you're having an issue with?",
            "Product Inquiry": "Which product are you interested in, and what specific details would you like to know?"
        }

    def get_state(self, user_id):
        return self.state_tracker.get(user_id, {"current_intent": None, "awaiting_clarification": False, "last_query": None})

    def set_state(self, user_id, state):
        self.state_tracker[user_id] = state

    def generate_clarification(self, detected_intents):
        if not detected_intents:
            return "I'm sorry, I couldn't understand your request. Could you please rephrase it or provide more details?", None
        elif len(detected_intents) > 1:
            options = ", ".join(detected_intents)
            return f"It seems like you might be asking about multiple things: {options}. Could you please clarify what you need help with?", "awaiting_clarification"
        else:
            intent = detected_intents[0]
            return self.clarification_questions.get(intent, f"Could you provide more details about your request related to {intent}?"), "awaiting_clarification"

    def generate_response(self, intent, action_result=None):
        if intent == "Order Status":
            if action_result and "order_id" in action_result:
                if action_result.get("status"):
                    return f"Your order {action_result['order_id']} is currently {action_result['status']}."
                else:
                    return f"I couldn't find details for order {action_result['order_id']}. Please double-check the ID."
            return "Please provide your order number so I can check its status."
        elif intent == "Return/Refund":
            if action_result and action_result.get("success"):
                return "Your return/refund request has been initiated. You will receive an email with further instructions."
            return "To initiate a return or refund, please provide the item details and reason."
        elif intent == "Technical Support":
            return "Please describe your technical issue in more detail, and I can try to help or connect you to an agent."
        elif intent == "Product Inquiry":
            return "I can provide information about products. What product are you interested in and what would you like to know?"
        elif intent == "General Help":
            return "I'm here to help! What can I assist you with today?"
        return "I'm not sure how to respond to that. Can I help with something else?"

class ActionExecutor:
    def __init__(self):
        self.action_dictionary = {
            "check_order_status": self._check_order_status,
            "initiate_return": self._initiate_return,
            "escalate_to_human": self._escalate_to_human,
            "provide_product_info": self._provide_product_info
        }

    def _check_order_status(self, entities):
        order_id = entities.get("order_id")
        if order_id:
            # Simulate API call to check order status
            if order_id == "12345":
                return {"order_id": order_id, "status": "shipped"}
            elif order_id == "67890":
                return {"order_id": order_id, "status": "processing"}
            return {"order_id": order_id, "status": None}
        return {"status": "Order ID missing"}

    def _initiate_return(self, entities):
        # Simulate API call to initiate return
        item_id = entities.get("item_id")
        reason = entities.get("reason")
        if item_id and reason:
            return {"success": True, "message": f"Return initiated for item {item_id} due to {reason}."}
        return {"success": False, "message": "Item ID or reason missing for return."}

    def _escalate_to_human(self, entities):
        return {"success": True, "message": "Connecting you to a human agent. Please wait.", "escalated": True}
    
    def _provide_product_info(self, entities):
        product_name = entities.get("product_name")
        if product_name:
            return {"success": True, "message": f"Searching for information about {product_name}."}
        return {"success": False, "message": "Please specify the product you are interested in."}

    def execute_action(self, intent, entities):
        if intent == "Order Status" and "order_id" in entities:
            return self.action_dictionary["check_order_status"](entities)
        elif intent == "Return/Refund":
            return self.action_dictionary["initiate_return"](entities)
        elif intent == "Technical Support":
            return self.action_dictionary["escalate_to_human"](entities)
        elif intent == "Product Inquiry":
            return self.action_dictionary["provide_product_info"](entities)
        return {"success": False, "message": "No specific action mapped for this intent and entities."}

class PersonalizationModule:
    def __init__(self):
        self.user_profiles = {}

    def get_profile(self, user_id):
        return self.user_profiles.get(user_id, {"history": []})

    def update_profile(self, user_id, new_data):
        profile = self.user_profiles.get(user_id, {"history": []})
        if "history" in new_data:
            profile["history"].append(new_data["history"])
        # Placeholder for more sophisticated updates
        self.user_profiles[user_id] = profile

class Chatbot:
    def __init__(self):
        self.nlu = NLUModule()
        self.dialogue_manager = DialogueManager()
        self.action_executor = ActionExecutor()
        self.personalization = PersonalizationModule()

    def converse(self, user_id, query):
        current_state = self.dialogue_manager.get_state(user_id)
        response = ""
        action_result = None

        if current_state["awaiting_clarification"]:
            # User is responding to a clarification question
            detected_intents, entities = self.nlu.process_query(query)
            if detected_intents:
                chosen_intent = detected_intents[0] # Assuming first detected is the clarification
                current_state["current_intent"] = chosen_intent
                current_state["awaiting_clarification"] = False
                self.dialogue_manager.set_state(user_id, current_state)
                action_result = self.action_executor.execute_action(chosen_intent, entities)
                response = self.dialogue_manager.generate_response(chosen_intent, action_result)
            else:
                response = "I'm still not clear. Could you please be more specific?"
        else:
            detected_intents, entities = self.nlu.process_query(query)
            
            if len(detected_intents) == 1:
                chosen_intent = detected_intents[0]
                current_state["current_intent"] = chosen_intent
                current_state["awaiting_clarification"] = False
                self.dialogue_manager.set_state(user_id, current_state)
                action_result = self.action_executor.execute_action(chosen_intent, entities)
                response = self.dialogue_manager.generate_response(chosen_intent, action_result)
            elif len(detected_intents) > 1 or not detected_intents:
                response, new_state_flag = self.dialogue_manager.generate_clarification(detected_intents)
                if new_state_flag == "awaiting_clarification":
                    current_state["awaiting_clarification"] = True
                    current_state["last_query"] = query
                    self.dialogue_manager.set_state(user_id, current_state)
            else:
                response = "I'm sorry, I couldn't understand your request. Can I help with something else?"
        
        # Personalization (simple history logging)
        self.personalization.update_profile(user_id, {"history": {"query": query, "response": response, "intent": current_state["current_intent"]}})
        
        return response

if __name__ == "__main__":
    chatbot = Chatbot()
    user_id = "user123"

    print("Chatbot: Hello! How can I assist you today?")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Chatbot: Goodbye!")
            break
        
        bot_response = chatbot.converse(user_id, user_input)
        print(f"Chatbot: {bot_response}")
