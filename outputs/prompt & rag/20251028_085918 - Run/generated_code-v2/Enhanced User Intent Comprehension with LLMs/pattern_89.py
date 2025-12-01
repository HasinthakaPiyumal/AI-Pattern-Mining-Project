
import re

class NLUModule:
    def __init__(self):
        self.intent_patterns = {
            "order_tracking": r"(?:track|where is|locate).*order(?: number)?\s*(\d+)?",
            "product_return": r"(?:return|refund|exchange).*product(?: item)?(?: order)?\s*(\d+)?",
            "account_management": r"(?:change|update).*password|address|account info|email",
            "technical_support": r"(?:problem|issue|not working|troubleshoot).*(\w+)?"
        }
        self.general_keywords = ["help", "support", "question", "can you assist"]

    def process_query(self, query: str) -> dict:
        query_lower = query.lower()
        recognized_intent = "unknown"
        entities = {}
        confidence = 0.0

        for intent, pattern in self.intent_patterns.items():
            match = re.search(pattern, query_lower)
            if match:
                recognized_intent = intent
                confidence = 0.9 # High confidence if a specific pattern matches
                if match.group(1): # Check if a group was captured (e.g., order number)
                    if intent == "order_tracking" or intent == "product_return":
                        entities["order_number"] = match.group(1)
                    elif intent == "technical_support":
                        entities["product_name"] = match.group(1)
                break

        if recognized_intent == "unknown":
            for keyword in self.general_keywords:
                if keyword in query_lower:
                    recognized_intent = "ambiguous"
                    confidence = 0.6 # Moderate confidence for general help
                    break

        if recognized_intent == "unknown" and not entities: # Still unknown and no entities
            confidence = 0.1

        return {"intent": recognized_intent, "entities": entities, "confidence": confidence}

class DialogueManager:
    def __init__(self):
        self.conversation_history = {}

    def update_context(self, user_id: str, turn: dict):
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
        self.conversation_history[user_id].append(turn)

    def get_context(self, user_id: str) -> list:
        return self.conversation_history.get(user_id, [])

    def needs_clarification(self, nlu_result: dict) -> bool:
        intent = nlu_result["intent"]
        confidence = nlu_result["confidence"]
        entities = nlu_result["entities"]

        if confidence < 0.7 and intent not in ["unknown", "ambiguous"]:
            return True
        if intent == "ambiguous":
            return True
        if intent == "order_tracking" and "order_number" not in entities:
            return True
        if intent == "product_return" and "order_number" not in entities:
            return True
        if intent == "technical_support" and "product_name" not in entities:
            return True
        return False

class ResponseGenerator:
    def __init__(self):
        self.responses = {
            "order_tracking": "Please provide your order number to track your package.",
            "product_return": "To process a return, I need your order number and the reason for return.",
            "account_management": "What specifically would you like to do with your account? (e.g., change password, update address)",
            "technical_support": "Could you describe your technical issue in more detail and mention the product name?",
            "ambiguous": "I'm not quite sure I understand. Could you please rephrase your request or provide more details?",
            "unknown": "I apologize, I'm currently unable to assist with that request. Please try asking something else or visit our FAQ page."
        }

    def generate_response(self, intent: str, entities: dict) -> str:
        if intent == "order_tracking" and "order_number" in entities:
            return f"Checking status for order {entities['order_number']}. Please wait a moment... (This is a simulation)"
        elif intent == "product_return" and "order_number" in entities:
            return f"Initiating return for order {entities['order_number']}. What is the reason for the return?"
        elif intent == "technical_support" and "product_name" in entities:
            return f"Please tell me more about the issue with your {entities['product_name']}."
        
        return self.responses.get(intent, self.responses["unknown"])

    def generate_clarifying_question(self, nlu_result: dict) -> str:
        intent = nlu_result["intent"]
        entities = nlu_result["entities"]

        if intent == "ambiguous":
            return "It seems your request is a bit unclear. Are you looking to inquire about an order, return a product, manage your account, or something else?"
        elif intent == "order_tracking" and "order_number" not in entities:
            return "To track your order, I need your order number. Could you please provide it?"
        elif intent == "product_return" and "order_number" not in entities:
            return "To help with your return, please provide your order number and the reason for the return."
        elif intent == "technical_support" and "product_name" not in entities:
            return "Which product are you having a technical issue with?"
        elif intent == "account_management": # If intent is recognized but details are missing
            return "Can you specify what you'd like to do with your account? (e.g., change password, update address, view orders)"
        
        return self.responses.get(intent, self.responses["unknown"])

class SmartCustomerSupportChatbot:
    def __init__(self):
        self.nlu_module = NLUModule()
        self.dialogue_manager = DialogueManager()
        self.response_generator = ResponseGenerator()
        self.user_id_counter = 0 # For simple demo, assign new user ID each run
        print("Smart Customer Support Chatbot initialized. Type 'exit' to quit.")

    def _get_or_create_user_id(self) -> str:
        # In a real app, this would come from session or login
        self.user_id_counter += 1
        return f"user_{self.user_id_counter}"

    def process_message(self, user_id: str, message: str) -> str:
        nlu_result = self.nlu_module.process_query(message)

        if self.dialogue_manager.needs_clarification(nlu_result):
            response = self.response_generator.generate_clarifying_question(nlu_result)
        else:
            response = self.response_generator.generate_response(nlu_result["intent"], nlu_result["entities"])
        
        turn = {
            "query": message,
            "nlu_result": nlu_result,
            "response": response
        }
        self.dialogue_manager.update_context(user_id, turn)
        return response

    def run(self):
        current_user_id = self._get_or_create_user_id()
        print(f"Chatbot: Hello! How can I assist you today? (Your session ID: {current_user_id})")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                print("Chatbot: Goodbye!")
                break
            response = self.process_message(current_user_id, user_input)
            print(f"Chatbot: {response}")

if __name__ == "__main__":
    chatbot = SmartCustomerSupportChatbot()
    chatbot.run()
