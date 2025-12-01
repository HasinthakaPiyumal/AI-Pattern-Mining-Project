import re

class IntentRecognizer:
    def __init__(self):
        self.patterns = {
            "greeting": r"^(hi|hello|hey|greetings|good morning|good afternoon|good evening).*",
            "thank_you": r"^(thank you|thanks|i appreciate it|much obliged).*",
            "check_order_status": r"(where is my order|order status|track my order|my stuff|delivery status|when will my order arrive).*",
            "reset_password": r"(reset my password|forgot password|change password|password help).*",
            "update_shipping": r"(update my address|change shipping address|my address is wrong|shipping details).*",
            "escalate_to_agent": r"(talk to a human|speak to an agent|connect me with support|i need more help).*"
        }

    def recognize_intent(self, text):
        text = text.lower()
        for intent, pattern in self.patterns.items():
            if re.match(pattern, text):
                return intent
        return "unknown"

class ToolExecutor:
    def check_order_status(self, order_id=None):
        if order_id:
            return f"Simulating: Checking status for order ID {order_id}."
        else:
            return "Please provide an order ID to check its status."

    def reset_password(self):
        return "Simulating: Initiating password reset process. A link has been sent to your registered email."

    def update_shipping(self, address=None):
        if address:
            return f"Simulating: Updating shipping address to {address}."
        else:
            return "Please provide the new shipping address."

    def escalate_to_agent(self):
        return "Simulating: Connecting you with a human agent. Please wait while we transfer you."

class PersonalizationModule:
    def __init__(self):
        self.user_preferences = {}

    def get_preference(self, user_id, key, default=None):
        return self.user_preferences.get(user_id, {}).get(key, default)

    def set_preference(self, user_id, key, value):
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {}
        self.user_preferences[user_id][key] = value

class DialogueManager:
    def __init__(self):
        self.intent_recognizer = IntentRecognizer()
        self.tool_executor = ToolExecutor()
        self.personalization_module = PersonalizationModule()
        self.conversation_state = {
            "current_intent": None,
            "awaiting_param": None,
            "user_id": "test_user"  # Simplified for demonstration
        }

    def process_input(self, user_input):
        user_id = self.conversation_state["user_id"]

        if self.conversation_state["awaiting_param"]:
            param = user_input.strip()
            intent_to_fulfill = self.conversation_state["current_intent"]
            self.conversation_state["awaiting_param"] = None
            self.conversation_state["current_intent"] = None

            if intent_to_fulfill == "check_order_status":
                return self.tool_executor.check_order_status(order_id=param)
            elif intent_to_fulfill == "update_shipping":
                return self.tool_executor.update_shipping(address=param)

        intent = self.intent_recognizer.recognize_intent(user_input)
        self.personalization_module.set_preference(user_id, "last_intent", intent)

        response = ""
        if intent == "greeting":
            response = "Hello! How can I assist you today?"
        elif intent == "thank_you":
            response = "You're welcome! Is there anything else I can help with?"
        elif intent == "check_order_status":
            self.conversation_state["current_intent"] = "check_order_status"
            self.conversation_state["awaiting_param"] = "order_id"
            response = "Please provide your order ID."
        elif intent == "reset_password":
            response = self.tool_executor.reset_password()
        elif intent == "update_shipping":
            self.conversation_state["current_intent"] = "update_shipping"
            self.conversation_state["awaiting_param"] = "address"
            response = "Please provide your new shipping address."
        elif intent == "escalate_to_agent":
            response = self.tool_executor.escalate_to_agent()
        elif intent == "unknown":
            last_intent_pref = self.personalization_module.get_preference(user_id, "last_intent")
            if last_intent_pref and last_intent_pref != "unknown":
                response = f"I'm not sure I understand. Were you trying to {last_intent_pref.replace('_', ' ')}? Can you rephrase or be more specific?"
            else:
                response = "I'm sorry, I didn't understand that. Can you please rephrase your request or choose from options like 'check order status', 'reset password', 'update shipping', or 'talk to an agent'?"
        
        return response

def run_chatbot():
    print("Welcome to the E-commerce Customer Support Chatbot! Type 'exit' to end the conversation.")
    dialogue_manager = DialogueManager()

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break

        response = dialogue_manager.process_input(user_input)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    run_chatbot()