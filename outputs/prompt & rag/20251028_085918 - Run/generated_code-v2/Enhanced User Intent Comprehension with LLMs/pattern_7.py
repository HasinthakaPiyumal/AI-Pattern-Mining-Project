import random

class CustomerSupportChatbot:
    def __init__(self):
        # In a real-world application, a fine-tuned transformer model (e.g., from Hugging Face's `transformers` library)
        # would be loaded here for robust intent classification. For this demonstration, we use rule-based intent detection.
        self.known_intents = [
            "order_status", "product_information", "returns_exchange",
            "shipping_inquiry", "greeting", "complaint", "technical_support",
            "account_management"
        ]
        self.user_sessions = {} # Stores user-specific data for personalization and context

    def _classify_intent(self, text):
        """Mock intent classification based on keywords. Replaced by an ML model in production."""
        text_lower = text.lower()

        if "order" in text_lower or "where is my stuff" in text_lower or "status of my purchase" in text_lower:
            return {"intent": "order_status", "confidence": 0.95}
        elif "product" in text_lower or "item details" in text_lower or "about this" in text_lower:
            return {"intent": "product_information", "confidence": 0.9}
        elif "return" in text_lower or "exchange" in text_lower or "send back" in text_lower:
            return {"intent": "returns_exchange", "confidence": 0.88}
        elif "shipping" in text_lower or "delivery" in text_lower or "how long" in text_lower:
            return {"intent": "shipping_inquiry", "confidence": 0.85}
        elif any(greeting in text_lower for greeting in ["hello", "hi", "hey", "good morning"]):
            return {"intent": "greeting", "confidence": 0.99}
        elif "problem" in text_lower or "issue" in text_lower or "bug" in text_lower or "not working" in text_lower:
            return {"intent": "technical_support", "confidence": 0.75}
        elif "account" in text_lower or "login" in text_lower or "password" in text_lower or "my details" in text_lower:
            return {"intent": "account_management", "confidence": 0.8}
        else:
            # Simulate low confidence for unhandled or ambiguous queries
            return {"intent": "unknown", "confidence": random.uniform(0.3, 0.6)} # Assign a random low confidence

    def _handle_greeting(self, user_id):
        user_data = self.user_sessions.get(user_id, {})
        name = user_data.get('name', 'there') # Example of personalization
        return f"Hello {name}! How can I assist you with your shopping today?"

    def _handle_order_status(self, user_id):
        return "To check your order status, please provide your order number. (e.g., 'My order is ABC123DEF')"

    def _handle_product_information(self, user_id):
        return "I can help with product information. Which product are you interested in? (e.g., 'Tell me about the new smartphone')"

    def _handle_returns_exchange(self, user_id):
        return "For returns or exchanges, please visit our 'Returns' page or provide your order number. (e.g., 'I want to return order GHI456JKL')"

    def _handle_shipping_inquiry(self, user_id):
        return "Please tell me more about your shipping inquiry. Are you asking about delivery times, costs, or tracking?"

    def _handle_technical_support(self, user_id):
        return "Could you please describe your technical issue in more detail? I can connect you with a specialist if needed."

    def _handle_account_management(self, user_id):
        return "For account management, please visit your account settings page. If you're having trouble logging in, I can guide you through password reset."

    def _ask_clarification(self):
        clarification_phrases = [
            "I'm not quite sure what you mean. Could you please rephrase?",
            "Could you provide more details about your request?",
            "To help me understand better, could you clarify what you're looking for?",
            "Are you looking for information about an order, a product, or something else?",
            "What exactly would you like assistance with?"
        ]
        return random.choice(clarification_phrases)

    def process_message(self, user_id, message):
        """Processes a user message, classifies intent, and generates a response."""
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {'past_intents': [], 'preferences': {}}

        intent_prediction = self._classify_intent(message)
        intent = intent_prediction['intent']
        confidence = intent_prediction['confidence']

        # Update user's intent history for potential personalization
        self.user_sessions[user_id]['past_intents'].append(intent)

        response = ""
        # If confidence is low, ask for clarification
        if confidence < 0.7:  # Threshold for low confidence
            response = self._ask_clarification()
        else:
            # Route to appropriate handler based on classified intent
            if intent == "greeting":
                response = self._handle_greeting(user_id)
            elif intent == "order_status":
                response = self._handle_order_status(user_id)
            elif intent == "product_information":
                response = self._handle_product_information(user_id)
            elif intent == "returns_exchange":
                response = self._handle_returns_exchange(user_id)
            elif intent == "shipping_inquiry":
                response = self._handle_shipping_inquiry(user_id)
            elif intent == "technical_support":
                response = self._handle_technical_support(user_id)
            elif intent == "account_management":
                response = self._handle_account_management(user_id)
            else:
                # Fallback for other potential "unknown" but higher confidence intents
                response = self._ask_clarification() # Still clarify if intent is not explicitly handled

        return response