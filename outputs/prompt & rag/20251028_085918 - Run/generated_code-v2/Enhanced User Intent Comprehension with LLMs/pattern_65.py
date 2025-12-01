import re

class VirtualAssistant:
    def __init__(self):
        # In a real application, a foundation model (e.g., from transformers library) would be loaded here.
        # For demonstration, we will simulate its understanding capabilities with keywords.
        print("Virtual Assistant initialized. Ready to understand customer intent.")
        self.known_intents = {
            "order_status": ["order", "late", "where is", "delivery", "track"],
            "product_info": ["product", "about", "details", "specifications", "info"],
            "return_policy": ["return", "refund", "policy", "exchange"],
            "escalate": ["complain", "speak to human", "agent", "support"]
        }
        self.order_data = {
            "ORD123": {"status": "shipped", "estimated_delivery": "2 days", "items": ["Laptop X"]},
            "ORD456": {"status": "processing", "items": ["Mouse Y", "Keyboard Z"]}
        }

    def _get_intent(self, query):
        query_lower = query.lower()
        for intent, keywords in self.known_intents.items():
            if any(keyword in query_lower for keyword in keywords):
                return intent
        return "general_inquiry"

    def _extract_entities(self, query, intent):
        entities = {}
        if intent == "order_status":
            order_match = re.search(r'(ORD\d{3})', query, re.IGNORECASE)
            if order_match:
                entities["order_number"] = order_match.group(1).upper()
        elif intent == "product_info":
            # Simple product name extraction - in a real system, this would be more robust
            product_keywords = ["laptop", "mouse", "keyboard"]
            for keyword in product_keywords:
                if keyword in query.lower():
                    entities["product_name"] = keyword.capitalize()
                    break
        return entities

    def _clarify_query(self, query, detected_intent, entities):
        if detected_intent == "order_status" and "order_number" not in entities:
            return "It seems you're asking about an order. Could you please provide your order number?"
        if detected_intent == "product_info" and "product_name" not in entities:
            return "Please tell me which product you are interested in. For example, 'tell me about the laptop'."
        return None

    def _handle_intent(self, intent, entities):
        if intent == "order_status":
            order_number = entities.get("order_number")
            if order_number and order_number in self.order_data:
                status = self.order_data[order_number]["status"]
                delivery_info = self.order_data[order_number].get("estimated_delivery", "N/A")
                items = ", ".join(self.order_data[order_number].get("items", []))
                return f"Your order {order_number} is currently {status}. Estimated delivery: {delivery_info}. Items: {items}."
            elif order_number:
                return f"I couldn't find details for order {order_number}. Please double-check the number."
            else:
                return "I need an order number to check the status. Can you provide it?"
        elif intent == "product_info":
            product_name = entities.get("product_name")
            if product_name:
                # Placeholder for actual product database lookup
                return f"Details for {product_name}: It's a high-quality {product_name} with excellent features. Would you like to know more about its specifications?"
            else:
                return "Please tell me which product you are interested in, and I can provide details."
        elif intent == "return_policy":
            return "Our return policy allows returns within 30 days of purchase for most items, provided they are in their original condition. Please refer to our website for full details or ask if you have a specific item in mind."
        elif intent == "escalate":
            return "I understand your frustration. I will connect you to a human agent who can assist you further. Please wait while I transfer you."
        else: # general_inquiry
            return "I'm sorry, I'm not entirely sure how to help with that. Could you rephrase your question or provide more details?"

    def process_query(self, user_query):
        print(f"User: {user_query}")

        intent = self._get_intent(user_query)
        entities = self._extract_entities(user_query, intent)
        
        clarification_response = self._clarify_query(user_query, intent, entities)
        if clarification_response:
            return clarification_response

        response = self._handle_intent(intent, entities)
        return response

# Example Usage:
# if __name__ == "__main__":
#     assistant = VirtualAssistant()
#     print(f"Assistant: {assistant.process_query('My order is late')}\n")
#     print(f"Assistant: {assistant.process_query('Where is my order ORD123?')}\n")
#     print(f"Assistant: {assistant.process_query('Tell me about your laptops')}\n")
#     print(f"Assistant: {assistant.process_query('What is the refund policy?')}\n")
#     print(f"Assistant: {assistant.process_query('I want to complain about a product')}\n")
#     print(f"Assistant: {assistant.process_query('I need help with something else')}\n")
