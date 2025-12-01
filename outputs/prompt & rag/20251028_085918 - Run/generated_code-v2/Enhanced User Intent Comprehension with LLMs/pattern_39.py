
import re

class IntentClassifier:
    def __init__(self):
        # Simulate a trained foundation model with instruction tuning
        # In a real-world scenario, this would be a loaded and fine-tuned LLM
        self.known_intents = {
            "order_status": [
                "where is my order", "track my package", "order status",
                "delivery status", "when will my order arrive"
            ],
            "initiate_return": [
                "i want to return", "how to return", "return an item",
                "start a return", "item not as described"
            ],
            "product_information": [
                "tell me about", "product details", "specifications of",
                "more info on", "what is this product"
            ],
            "technical_support": [
                "technical issue", "not working", "bug", "error",
                "troubleshoot"
            ],
            "account_inquiry": [
                "my account", "change password", "update profile",
                "billing issue", "login problem"
            ]
        }
        self.intent_keywords = {
            intent: [keyword.lower() for keyword in keywords] 
            for intent, keywords in self.known_intents.items()
        }

    def predict_intent(self, query: str) -> tuple[str | None, float]:
        query_lower = query.lower()
        matched_intents = []
        for intent, keywords in self.intent_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    matched_intents.append(intent)
                    break # Only count once per intent

        if len(matched_intents) == 1:
            return matched_intents[0], 1.0 # High confidence if a single clear intent
        elif len(matched_intents) > 1:
            # Ambiguous: multiple potential intents detected
            return "ambiguous", matched_intents # Return the list of ambiguous intents
        else:
            return None, 0.0 # No clear intent found

class SmartCustomerSupportAgent:
    def __init__(self):
        self.classifier = IntentClassifier()
        self.user_context = {}
        print("Hello! How can I assist you today?")

    def process_query(self, query: str) -> str:
        query_lower = query.lower()

        # Handle greetings and farewells
        if any(greeting in query_lower for greeting in ["hello", "hi", "hey"]):
            return "Hello! How can I help you today?"
        if any(farewell in query_lower for farewell in ["bye", "goodbye", "see you"]):
            return "Goodbye! Have a great day."

        intent, confidence_or_intents = self.classifier.predict_intent(query)

        if intent == "ambiguous":
            return self._ask_clarification(confidence_or_intents)
        elif intent:
            # Here, we could potentially update user_context based on the intent
            # For personalization, we'd store user preferences/history tied to an ID.
            return self._execute_action(intent, query)
        else:
            return "I'm sorry, I couldn't quite understand your request. Could you please rephrase it or provide more details?"

    def _execute_action(self, intent: str, query: str) -> str:
        # In a real system, this would trigger calls to various e-commerce APIs
        if intent == "order_status":
            order_id_match = re.search(r'order #?(\d+)', query, re.IGNORECASE)
            if order_id_match:
                order_id = order_id_match.group(1)
                return f"Checking the status for order {order_id}. Please wait a moment... (Simulated response: Your order #{order_id} is currently in transit and expected by tomorrow.)"
            else:
                return "Please provide your order number so I can check its status for you."
        elif intent == "initiate_return":
            return "To initiate a return, please visit our returns portal at example.com/returns or provide your order number."
        elif intent == "product_information":
            product_match = re.search(r'(tell me about|more info on|what is this product)\s+(.+)', query, re.IGNORECASE)
            if product_match:
                product_name = product_match.group(2).strip()
                return f"Retrieving information for {product_name}. (Simulated response: {product_name} is a highly-rated item with features X, Y, Z.)"
            else:
                return "What product are you interested in? Please provide the product name or ID."
        elif intent == "technical_support":
            return "I can connect you with a technical support agent, or you can find troubleshooting guides in our help center."
        elif intent == "account_inquiry":
            return "For account-related issues, please verify your identity or visit your account settings page."
        else:
            return "An unexpected action was requested. Please try again."

    def _ask_clarification(self, ambiguous_intents: list[str]) -> str:
        intent_names = {
            "order_status": "order status",
            "initiate_return": "returning an item",
            "product_information": "product information",
            "technical_support": "technical support",
            "account_inquiry": "account inquiry"
        }
        # Generate a question to clarify between the detected ambiguous intents
        options = [intent_names.get(i, i.replace('_', ' ')) for i in ambiguous_intents]
        return f"It seems you might be asking about {', '.join(options[:-1])} or {options[-1]}? Could you please clarify?"

# --- Main Interaction Loop ---
if __name__ == "__main__":
    agent = SmartCustomerSupportAgent()
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Agent: Goodbye!")
            break
        response = agent.process_query(user_input)
        print(f"Agent: {response}")
