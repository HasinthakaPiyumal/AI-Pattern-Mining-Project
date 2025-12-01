import re

class IntentRecognizer:
    def __init__(self):
        self.intents = {
            "track_order": ["track", "order", "where's my", "delivery", "late"],
            "return_request": ["return", "refund", "send back", "damaged"],
            "product_info": ["product", "details", "about", "specifications"],
            "payment_issue": ["payment", "charged", "billing", "invoice"],
            "general_greeting": ["hello", "hi", "hey", "how are you"],
            "general_farewell": ["bye", "goodbye", "see you"],
            "unknown": []
        }

    def recognize_intent(self, query: str) -> dict:
        query_lower = query.lower()
        recognized_intent = "unknown"
        entities = {}

        for intent, keywords in self.intents.items():
            for keyword in keywords:
                if keyword in query_lower:
                    recognized_intent = intent
                    break
            if recognized_intent != "unknown":
                break

        if recognized_intent == "track_order":
            order_id_match = re.search(r"(?:order number|order id|id is|#|num)\s*([a-zA-Z0-9-]+)", query_lower)
            if order_id_match:
                entities["order_id"] = order_id_match.group(1)
            else:
                potential_ids = re.findall(r"\b[a-zA-Z0-9]{5,15}\b", query_lower)
                if potential_ids:
                    entities["order_id"] = potential_ids[0]

        return {"intent": recognized_intent, "entities": entities}


class CustomerSupportAgent:
    def __init__(self):
        self.intent_recognizer = IntentRecognizer()
        self.conversation_state = {
            "awaiting_order_id": False,
            "last_intent": None
        }

    def _simulate_order_tracking(self, order_id: str) -> str:
        if not order_id:
            return "I need an order number to track your order. Could you please provide it?"
        
        mock_orders = {
            "ABC12345": "Your order ABC12345 is currently in transit and expected to arrive by EOD tomorrow.",
            "XYZ98765": "Order XYZ98765 was delivered on 2023-10-26. If you haven't received it, please contact us directly.",
            "OPQ67890": "Order OPQ67890 is being processed and will be shipped within 2 business days."
        }

        response = mock_orders.get(order_id.upper(), f"I couldn't find any information for order ID {order_id}. Please double-check it.")
        return response

    def handle_query(self, query: str) -> str:
        response = ""
        current_intent_data = self.intent_recognizer.recognize_intent(query)
        intent = current_intent_data["intent"]
        entities = current_intent_data["entities"]

        if self.conversation_state["awaiting_order_id"]:
            order_id = entities.get("order_id")
            if order_id:
                response = self._simulate_order_tracking(order_id)
                self.conversation_state["awaiting_order_id"] = False
                self.conversation_state["last_intent"] = None
            else:
                response = "I'm still waiting for your order number. Could you please provide it?"
            return response

        self.conversation_state["last_intent"] = intent

        if intent == "general_greeting":
            response = "Hello! How can I assist you with your e-commerce query today?"
        elif intent == "track_order":
            order_id = entities.get("order_id")
            if order_id:
                response = self._simulate_order_tracking(order_id)
            else:
                self.conversation_state["awaiting_order_id"] = True
                response = "To track your order, I'll need your order number. Could you please provide it?"
        elif intent == "return_request":
            response = "I can help you with returns. Please visit our returns page at [Link to Returns Page] for more information or provide your order number to initiate a return."
        elif intent == "product_info":
            response = "I can provide product information. Which product are you interested in?"
        elif intent == "payment_issue":
            response = "For payment issues, please contact our billing department directly at billing@example.com or call us at 1-800-PAYMENT. You can also check our FAQ for common billing questions."
        elif intent == "general_farewell":
            response = "Goodbye! Have a great day and thank you for shopping with us!"
        else:
            response = "I'm sorry, I didn't quite understand that. Could you please rephrase your request or ask about something else like order tracking, returns, or product information?"
        
        return response


def main():
    print("Welcome to the E-commerce Customer Support Assistant! Type 'exit' to quit.")
    agent = CustomerSupportAgent()

    while True:
        user_query = input("You: ")
        if user_query.lower() == 'exit':
            print("Assistant: Goodbye!\n")
            break
        
        response = agent.handle_query(user_query)
        print(f"Assistant: {response}\n")

if __name__ == "__main__":
    main()