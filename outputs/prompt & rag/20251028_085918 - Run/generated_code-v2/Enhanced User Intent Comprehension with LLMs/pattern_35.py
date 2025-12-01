import random
import time


class IntentRecognizer:
    def __init__(self):
        self.intents = {
            "order status": ["where is my order", "track my package", "what is the status of my delivery"],
            "return product": ["i want to return an item", "how do i send back my purchase", "return policy"],
            "payment issue": ["payment failed", "charge error", "billing problem"],
            "product inquiry": ["tell me about a product", "product details", "specifications"],
            "account management": ["change my address", "update my profile", "account settings"],
            "greeting": ["hello", "hi", "hey"]
        }
        self.confidence_threshold = 0.7

    def predict_intent(self, text):
        text_lower = text.lower()
        for intent, phrases in self.intents.items():
            for phrase in phrases:
                if phrase in text_lower:
                    return intent, 0.95  # High confidence for direct matches
        
        # Simulate a less confident prediction for unseen phrases
        if "order" in text_lower or "delivery" in text_lower:
            return "order status", 0.65
        if "return" in text_lower or "item back" in text_lower:
            return "return product", 0.60
        if "product" in text_lower or "item" in text_lower:
            return "product inquiry", 0.75
        
        return "unknown", 0.40 # Low confidence for completely unknown


class EntityExtractor:
    def __init__(self):
        pass

    def extract_entities(self, text):
        entities = {}
        text_lower = text.lower()

        # Simulate order number extraction
        if "order number" in text_lower:
            # Simple regex-like extraction for demonstration
            import re
            match = re.search(r'(?:order number|order id|order #)\s*(\d+)', text_lower)
            if match:
                entities["order_number"] = match.group(1)
        
        # Simulate product name extraction
        product_keywords = ["shoe", "shirt", "laptop", "book"]
        for keyword in product_keywords:
            if keyword in text_lower:
                entities["product_name"] = keyword
                break

        return entities


class DialogueManager:
    def __init__(self):
        self.conversation_history = {}
        self.user_profiles = {}

    def track_context(self, user_id, query, intent, entities):
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
        self.conversation_history[user_id].append({"query": query, "intent": intent, "entities": entities, "timestamp": time.time()})

    def get_user_profile(self, user_id):
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {"name": f"Customer {user_id}", "past_orders": [], "preferences": {}}
        return self.user_profiles[user_id]

    def update_user_profile(self, user_id, data):
        if user_id in self.user_profiles:
            self.user_profiles[user_id].update(data)

    def resolve_ambiguity(self, intent, intent_confidence, entities):
        if intent == "order status" and "order_number" not in entities:
            return "To check your order status, please provide your order number."
        elif intent == "return product" and "product_name" not in entities:
            return "Which product would you like to return?"
        elif intent_confidence < 0.7: # General low confidence threshold
            return "I'm not entirely sure I understood. Could you please rephrase or be more specific?"
        return None


class ResponseGenerator:
    def __init__(self):
        self.predefined_responses = {
            "greeting": "Hello! How can I help you today?",
            "default": "I'm sorry, I couldn't understand your request. Can you please tell me more?",
            "human_handoff": "I'm transferring you to a human agent who can help you further. Please wait a moment."
        }

    def _get_order_details(self, order_id):
        # Simulate an API call to an e-commerce backend
        if order_id == "12345":
            return "Your order 12345 is currently out for delivery and expected by end of day."
        else:
            return f"I couldn't find details for order number {order_id}. Please double-check the number."

    def _initiate_return(self, product_name):
        # Simulate an API call to an e-commerce backend
        if product_name:
            return f"Initiating return process for {product_name}. Please check your email for instructions."
        else:
            return "I need to know which product you want to return. Can you please specify?"

    def generate_response(self, user_id, intent, entities, user_profile, conversation_history):
        if intent == "greeting":
            return f"Hello {user_profile['name']}! {self.predefined_responses['greeting']}"

        if intent == "order status":
            order_number = entities.get("order_number")
            if order_number:
                return self._get_order_details(order_number)
            else:
                return "Please provide your order number to check the status."

        elif intent == "return product":
            product_name = entities.get("product_name")
            if product_name:
                return self._initiate_return(product_name)
            else:
                return "To initiate a return, please tell me which product you'd like to return."
        
        elif intent == "product inquiry":
            product_name = entities.get("product_name")
            if product_name:
                return f"For {product_name}, we have various models available. What specific details are you looking for?"
            else:
                return "What product are you interested in?"

        elif intent == "account management":
            return f"I can help you with account management, {user_profile['name']}. What would you like to update?"

        return self.predefined_responses.get(intent, self.predefined_responses["default"])

    def handoff_to_human(self, conversation_history):
        # In a real scenario, this would trigger an alert or ticket in a CRM system
        # For this demo, we just print the history that would be passed.
        print("--- Handoff to Human Agent ---")
        print("Conversation History:", conversation_history)
        print("-----------------------------")
        return self.predefined_responses["human_handoff"]


class Chatbot:
    def __init__(self):
        self.intent_recognizer = IntentRecognizer()
        self.entity_extractor = EntityExtractor()
        self.dialogue_manager = DialogueManager()
        self.response_generator = ResponseGenerator()
        self.user_id_counter = 0

    def get_user_id(self):
        # In a real system, this would come from user authentication/session management
        self.user_id_counter += 1
        return f"user_{self.user_id_counter}"

    def process_query(self, user_id, query):
        intent, confidence = self.intent_recognizer.predict_intent(query)
        entities = self.entity_extractor.extract_entities(query)
        
        self.dialogue_manager.track_context(user_id, query, intent, entities)
        user_profile = self.dialogue_manager.get_user_profile(user_id)

        ambiguity_response = self.dialogue_manager.resolve_ambiguity(intent, confidence, entities)
        if ambiguity_response:
            return ambiguity_response

        if intent == "unknown" and confidence < self.intent_recognizer.confidence_threshold:
            return self.response_generator.handoff_to_human(self.dialogue_manager.conversation_history.get(user_id, []))
        
        response = self.response_generator.generate_response(user_id, intent, entities, user_profile, self.dialogue_manager.conversation_history.get(user_id, []))
        return response


if __name__ == "__main__":
    chatbot = Chatbot()
    print("Welcome to the E-commerce Customer Support Chatbot! Type 'exit' to end the conversation.")

    current_user_id = chatbot.get_user_id()

    while True:
        user_input = input(f"You ({current_user_id}): ")
        if user_input.lower() == 'exit':
            break

        response = chatbot.process_query(current_user_id, user_input)
        print(f"Chatbot: {response}")

