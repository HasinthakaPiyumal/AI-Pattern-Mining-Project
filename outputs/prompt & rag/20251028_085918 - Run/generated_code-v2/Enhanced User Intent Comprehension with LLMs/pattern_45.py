import re

class NLUModule:
    def __init__(self):
        self.intents = {
            "track_order": {
                "keywords": ["track", "where is", "order status", "my package"],
                "entities": {"order_id": r"\b\d{9,12}\b"}
            },
            "initiate_return": {
                "keywords": ["return", "send back", "refund"],
                "entities": {"order_id": r"\b\d{9,12}\b", "reason": r"for (.*?)(?:\.|,|for|$)"}
            },
            "account_query": {
                "keywords": ["account", "my details", "login", "password"],
                "entities": {}
            },
            "product_info": {
                "keywords": ["product", "about", "specifications", "details"],
                "entities": {"product_name": r"product (.*?)(?:\.|,|for|$)", "product_id": r"\b[A-Z0-9]{3,10}\b"}
            },
            "shipping_info": {
                "keywords": ["shipping", "delivery time", "cost"],
                "entities": {}
            },
            "greeting": {
                "keywords": ["hello", "hi", "hey"],
                "entities": {}
            },
            "farewell": {
                "keywords": ["bye", "goodbye", "see you"],
                "entities": {}
            }
        }

    def predict_intent(self, text):
        text_lower = text.lower()
        best_intent = "unknown"
        max_confidence = 0.0

        for intent, data in self.intents.items():
            for keyword in data["keywords"]:
                if keyword in text_lower:
                    # Simple confidence based on keyword count, higher for more specific matches
                    confidence = 0.6 + (text_lower.count(keyword) * 0.1)
                    if confidence > max_confidence:
                        max_confidence = min(confidence, 0.95) # Cap confidence
                        best_intent = intent
                    break # Found a keyword for this intent, move to next intent

        if best_intent == "unknown" and len(text.split()) > 1:
            max_confidence = 0.5 # Default low confidence for general unknown intent
        elif best_intent != "unknown":
            # If a clear intent is found, boost confidence slightly
            max_confidence = max(0.7, max_confidence)

        return {"intent": best_intent, "confidence": max_confidence}

    def extract_entities(self, text, intent):
        entities = {}
        if intent in self.intents:
            for entity_name, pattern in self.intents[intent]["entities"].items():
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    if entity_name == "reason":
                        # For reason, often the group 1 is the actual reason
                        entities[entity_name] = match.group(1).strip().replace("for ", "")
                    else:
                        entities[entity_name] = match.group(0).strip()
        return entities

class DialogueManagementModule:
    def __init__(self):
        self.context = {}
        self.clarification_questions = {
            "track_order": "What is your order number?",
            "initiate_return": "What is the order number you wish to return, and what is the reason for the return?",
            "account_query": "Could you please specify what you need help with regarding your account?",
            "product_info": "Which product are you interested in? Please provide a name or ID if you have it."
        }

    def get_context(self):
        return self.context

    def update_context(self, key, value):
        self.context[key] = value

    def clear_context(self):
        self.context = {}

    def manage_dialogue(self, nlu_result):
        intent = nlu_result["intent"]
        confidence = nlu_result["confidence"]
        entities = nlu_result["entities"]
        response = ""
        action_required = False
        required_entities_present = True

        if intent == "unknown" or confidence < 0.7:
            response = "I'm not sure I understand. Could you please rephrase or provide more details?"
        else:
            # Check for missing crucial entities for action-oriented intents
            if intent == "track_order" and "order_id" not in entities:
                response = self.clarification_questions["track_order"]
                required_entities_present = False
            elif intent == "initiate_return" and ("order_id" not in entities or "reason" not in entities):
                response = self.clarification_questions["initiate_return"]
                required_entities_present = False
            elif intent == "product_info" and "product_name" not in entities and "product_id" not in entities:
                response = self.clarification_questions["product_info"]
                required_entities_present = False

            if required_entities_present:
                action_required = True

        return {"response": response, "action_required": action_required, "intent": intent, "entities": entities}

class ToolIntegrationModule:
    def __init__(self):
        pass

    def _simulate_api_call(self, tool_name, params):
        print(f"[SIMULATING API CALL to {tool_name} with params: {params}]")
        import time
        time.sleep(0.5) # Simulate network latency
        return True # Assume success for simulation

    def track_order_api(self, order_id):
        if self._simulate_api_call("Order Tracking API", {"order_id": order_id}):
            return f"Your order {order_id} is currently out for delivery and expected within 2 days."
        return "Sorry, I couldn't track your order at the moment. Please try again later."

    def initiate_return_api(self, order_id, reason):
        if self._simulate_api_call("Return Initiation API", {"order_id": order_id, "reason": reason}):
            return f"Return for order {order_id} due to '{reason}' has been initiated. You will receive an email with instructions shortly."
        return "Sorry, I couldn't initiate the return. Please check your order details and try again."

    def get_account_details_api(self, user_id):
        if self._simulate_api_call("Account Details API", {"user_id": user_id}):
            return "I've retrieved your account details. What specific information are you looking for? (e.g., address, payment methods)"
        return "Sorry, I couldn't access your account details. Please ensure you are logged in correctly."

    def get_product_info_api(self, product_identifier):
        if self._simulate_api_call("Product Info API", {"product_identifier": product_identifier}):
            return f"The {product_identifier} is a popular item with excellent reviews. It features X, Y, and Z. Would you like to know more?"
        return f"I couldn't find information for {product_identifier}. Please check the product name or ID."

    def route_and_execute(self, intent, entities):
        if intent == "track_order" and "order_id" in entities:
            return self.track_order_api(entities["order_id"])
        elif intent == "initiate_return" and "order_id" in entities and "reason" in entities:
            return self.initiate_return_api(entities["order_id"], entities["reason"])
        elif intent == "account_query": # For account query, we might just provide general info or prompt for more specifics
            # In a real system, you'd use user_id from session or authentication
            return self.get_account_details_api("current_user_id_mock") # Mock user ID
        elif intent == "product_info" and ("product_name" in entities or "product_id" in entities):
            product_identifier = entities.get("product_name") or entities.get("product_id")
            return self.get_product_info_api(product_identifier)
        elif intent == "greeting":
            return "Hello! How can I assist you today?"
        elif intent == "farewell":
            return "Goodbye! Have a great day!"
        else:
            return "I can't perform this action right now. Can I help with something else?"

class SmartCustomerSupportAssistant:
    def __init__(self):
        self.nlu = NLUModule()
        self.dialogue_manager = DialogueManagementModule()
        self.tool_integrator = ToolIntegrationModule()

    def process_query(self, query):
        # NLU Phase
        nlu_intent_result = self.nlu.predict_intent(query)
        intent = nlu_intent_result["intent"]
        confidence = nlu_intent_result["confidence"]
        entities = self.nlu.extract_entities(query, intent)
        
        # Combine NLU results
        nlu_full_result = {
            "intent": intent,
            "confidence": confidence,
            "entities": entities
        }

        # Dialogue Management Phase
        dialogue_response = self.dialogue_manager.manage_dialogue(nlu_full_result)

        if dialogue_response["action_required"]:
            # Tool Integration Phase
            tool_output = self.tool_integrator.route_and_execute(dialogue_response["intent"], dialogue_response["entities"])
            return tool_output
        else:
            return dialogue_response["response"]

    def run(self):
        print("\nWelcome to the Smart Customer Support Assistant! Type 'quit' to exit.\n")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'quit':
                print("Assistant: Goodbye!\n")
                break
            
            response = self.process_query(user_input)
            print(f"Assistant: {response}")

if __name__ == "__main__":
    assistant = SmartCustomerSupportAssistant()
    assistant.run()