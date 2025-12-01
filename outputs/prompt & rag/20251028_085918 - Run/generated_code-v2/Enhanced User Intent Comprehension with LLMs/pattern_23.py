import re

class IntentRecognizer:
    def __init__(self):
        self.intent_patterns = {
            "check_order_status": [
                r"order status (\w+)",
                r"where is my order (\w+)",
                r"tracking for (\w+)",
            ],
            "product_inquiry": [
                r"tell me about (.*) product",
                r"details on (.*)",
                r"info about (.*)",
            ],
            "initiate_return": [
                r"return order (\w+) for (.*)",
                r"i want to return (.*) from order (\w+)",
                r"start a return for (.*)",
            ],
        }

    def recognize_intent(self, text):
        text_lower = text.lower()
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    entities = [arg for arg in match.groups() if arg]
                    return {"intent": intent, "entities": entities, "ambiguous": False}
        return {"intent": None, "entities": [], "ambiguous": True}

class ECommerceTools:
    def get_order_status(self, order_id):
        if order_id == "12345":
            return f"Order {order_id} is currently 'Shipped' and expected on [Date]."
        elif order_id == "67890":
            return f"Order {order_id} is currently 'Processing'."
        else:
            return f"Order {order_id} not found."

    def search_products(self, query):
        if "laptop" in query.lower():
            return f"Found several laptops: 'ProBook X', 'UltraLap 5000'."
        elif "headphone" in query.lower():
            return f"Check out 'NoiseCanceller 3000' and 'SportBuds Pro'."
        else:
            return f"No products found matching '{query}'."

    def initiate_return(self, order_id, product_name, reason):
        if order_id == "12345" and product_name.lower() == "laptop":
            return f"Return for '{product_name}' from order {order_id} initiated. Reason: {reason}. You will receive a return label shortly."
        else:
            return f"Could not initiate return for '{product_name}' from order {order_id}. Please check details."

class DialogueManager:
    def __init__(self):
        self.intent_recognizer = IntentRecognizer()
        self.ecommerce_tools = ECommerceTools()
        self.context = {}

    def process_query(self, user_query):
        intent_data = self.intent_recognizer.recognize_intent(user_query)
        intent = intent_data["intent"]
        entities = intent_data["entities"]
        ambiguous = intent_data["ambiguous"]

        if ambiguous or intent is None:
            return "I'm not sure what you mean. Could you please rephrase or be more specific?"

        if intent == "check_order_status":
            if not entities:
                self.context["pending_intent"] = "check_order_status"
                return "Could you please provide the order ID?"
            order_id = entities[0]
            self.context = {}
            return self.ecommerce_tools.get_order_status(order_id)

        elif intent == "product_inquiry":
            if not entities:
                self.context["pending_intent"] = "product_inquiry"
                return "What product are you interested in?"
            product_query = entities[0]
            self.context = {}
            return self.ecommerce_tools.search_products(product_query)

        elif intent == "initiate_return":
            if len(entities) < 3:
                self.context["pending_intent"] = "initiate_return"
                if not entities:
                    return "To initiate a return, I need the order ID, product name, and reason for return."
                elif len(entities) == 1:
                    self.context["order_id"] = entities[0]
                    return "Okay, you want to return from order {}. What product would you like to return and why?".format(entities[0])
                elif len(entities) == 2:
                    self.context["product_name"] = entities[0]
                    self.context["order_id"] = entities[1]
                    return "And what is the reason for returning the {} from order {}?".format(entities[0], entities[1])
            
            order_id = None
            product_name = None
            reason = None

            # Attempt to extract entities based on common patterns for return
            # Example: 