class FoundationModel: 
    """Simulates a fine-tuned foundation model for intent understanding."""
    def __init__(self):
        # A simple mapping for demonstration of intent understanding
        self.intent_map = {
            "problem with order": {"intent": "order_issue", "confidence": 0.9},
            "track my order": {"intent": "track_order", "confidence": 0.95},
            "where is my package": {"intent": "track_order", "confidence": 0.9},
            "return an item": {"intent": "returns", "confidence": 0.85},
            "change my address": {"intent": "account_update", "confidence": 0.8},
            "billing question": {"intent": "billing_issue", "confidence": 0.8},
            "payment issue": {"intent": "billing_issue", "confidence": 0.85},
            "technical support": {"intent": "tech_support", "confidence": 0.9},
            "help me": {"intent": "ambiguous", "confidence": 0.5},
            "something wrong": {"intent": "ambiguous", "confidence": 0.6},
            "how to": {"intent": "faq", "confidence": 0.7},
            "general query": {"intent": "general_inquiry", "confidence": 0.75},
        }

    def predict_intent(self, query: str) -> dict:
        """Predicts the intent and confidence based on the query."""
        query_lower = query.lower()
        for phrase, data in self.intent_map.items():
            if phrase in query_lower:
                return data
        return {"intent": "unknown", "confidence": 0.4}

class ToolManager:
    """Manages various tools like order tracking and FAQ lookup."""
    def track_order(self, order_id: str) -> str:
        """Simulates tracking an order."""
        if order_id == "12345":
            return f"Order {order_id}: Shipped and expected to arrive on 2023-10-27."
        elif order_id == "67890":
            return f"Order {order_id}: Processing. Will ship within 2 business days."
        else:
            return f"Order {order_id}: Not found. Please check the order ID."

    def search_faq(self, query: str) -> str:
        """Simulates searching the FAQ database."""
        if "return policy" in query.lower():
            return "Our return policy allows returns within 30 days of purchase with the original receipt and packaging."
        elif "shipping cost" in query.lower():
            return "Standard shipping is $5.99. Free shipping on orders over $50."
        else:
            return "I couldn't find a direct answer in the FAQ. Please provide more details."

class IntentRouter:
    """Routes intents to appropriate actions or departments."""
    def __init__(self, tool_manager):
        self.tool_manager = tool_manager

    def route_intent(self, intent_data: dict, query: str, user_profile: dict) -> str:
        intent = intent_data["intent"]
        confidence = intent_data["confidence"]

        if confidence < 0.6:
            return "I'm not sure I fully understand. Could you please rephrase or provide more details?"

        if intent == "track_order":
            # Simple regex-like extraction for demonstration
            import re
            match = re.search(r'order\s*ID\s*(\d+)|order\s*number\s*(\d+)|#(\d+)|\b(\d{5})\b', query, re.IGNORECASE)
            order_id = match.group(1) or match.group(2) or match.group(3) or match.group(4) if match else None

            if order_id:
                return self.tool_manager.track_order(order_id)
            else:
                return "To track your order, please provide your order ID."
        elif intent == "returns":
            return self.tool_manager.search_faq("return policy") + " Would you like to initiate a return?"
        elif intent == "billing_issue":
            return "Please provide your account details so I can connect you with our billing department."
        elif intent == "tech_support":
            return "Connecting you to a technical support specialist. Please describe your issue in detail."
        elif intent == "faq" or intent == "general_inquiry":
            return self.tool_manager.search_faq(query)
        elif intent == "account_update":
            return "Please log in to your account to update your personal information or address."
        elif intent == "order_issue":
            return "I understand you have an issue with your order. Can you tell me more about it (e.g., wrong item, damaged, not received)?"
        elif intent == "ambiguous":
            return "It seems your request is a bit general. Could you tell me if this is about an order, a product, billing, or something else?"
        else:
            return "I'm sorry, I cannot handle that request at the moment. Please try again later or contact a human agent."

class CustomerSupportAssistant:
    """A smart customer support assistant for an E-commerce platform."""
    def __init__(self):
        self.foundation_model = FoundationModel()
        self.tool_manager = ToolManager()
        self.intent_router = IntentRouter(self.tool_manager)
        self.user_profiles = {}

    def _get_or_create_user_profile(self, user_id: str) -> dict:
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                "past_interactions": [],
                "preferred_contact": None,
                "recent_order_ids": [],
            }
        return self.user_profiles[user_id]

    def handle_query(self, user_id: str, query: str) -> str:
        user_profile = self._get_or_create_user_profile(user_id)
        user_profile["past_interactions"].append({"query": query, "timestamp": "now"}) # Simplified timestamp

        intent_data = self.foundation_model.predict_intent(query)
        print(f"[DEBUG] Predicted intent for '{query}': {intent_data}")

        response = self.intent_router.route_intent(intent_data, query, user_profile)

        # Simulate personalization based on interaction history (simplified)
        if "order_id" in response and user_profile["recent_order_ids"] and user_profile["recent_order_ids"][-1] not in response:
             # If a new order is discussed, add it to recent_order_ids
            import re
            match = re.search(r'Order\s*(\d+)', response, re.IGNORECASE)
            if match and match.group(1) not in user_profile["recent_order_ids"]:
                user_profile["recent_order_ids"].append(match.group(1))

        return response

# Example Usage:
if __name__ == "__main__":
    assistant = CustomerSupportAssistant()

    print("--- User 1 (New User) ---")
    print("User 1: ", "I need help with something.")
    response = assistant.handle_query("user123", "I need help with something.")
    print("Assistant: ", response)

    print("\nUser 1: ", "I have a problem with my order ID 12345.")
    response = assistant.handle_query("user123", "I have a problem with my order ID 12345.")
    print("Assistant: ", response)
    print(f"[DEBUG] User1 Profile: {assistant.user_profiles['user123']}")

    print("\nUser 1: ", "Where is my package? My order number is 12345.")
    response = assistant.handle_query("user123", "Where is my package? My order number is 12345.")
    print("Assistant: ", response)

    print("\nUser 1: ", "How can I return an item?")
    response = assistant.handle_query("user123", "How can I return an item?")
    print("Assistant: ", response)

    print("\n--- User 2 (Existing User) ---")
    print("User 2: ", "My payment didn't go through.")
    response = assistant.handle_query("user456", "My payment didn't go through.")
    print("Assistant: ", response)
    print(f"[DEBUG] User2 Profile: {assistant.user_profiles['user456']}")

    print("\nUser 2: ", "Can you help me with a billing question?")
    response = assistant.handle_query("user456", "Can you help me with a billing question?")
    print("Assistant: ", response)

    print("\nUser 1: ", "I have a technical issue.")
    response = assistant.handle_query("user123", "I have a technical issue.")
    print("Assistant: ", response)

    print("\nUser 1: ", "What about order 67890?")
    response = assistant.handle_query("user123", "What about order 67890?")
    print("Assistant: ", response)
    print(f"[DEBUG] User1 Profile: {assistant.user_profiles['user123']}")

    print("\nUser 1: ", "What is the shipping cost?")
    response = assistant.handle_query("user123", "What is the shipping cost?")
    print("Assistant: ", response)