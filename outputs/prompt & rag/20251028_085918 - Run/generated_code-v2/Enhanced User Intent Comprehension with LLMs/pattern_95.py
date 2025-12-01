import re
from collections import defaultdict

class IntentRecognizer:
    def __init__(self):
        self.intents = {
            "track_order": ["track order", "where is my order", "order status"],
            "product_inquiry": ["product information", "details about", "tell me about", "what is"],
            "refund_request": ["return item", "refund my purchase", "want a refund"],
            "greeting": ["hello", "hi", "hey", "good morning"],
            "escalate": ["talk to human", "speak to agent", "customer service"],
        }

    def predict_intent(self, text):
        text_lower = text.lower()
        for intent, keywords in self.intents.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return intent, 0.9  # Simulated high confidence
        return "unknown", 0.5

class EntityExtractor:
    def __init__(self):
        self.order_number_pattern = re.compile(r"\b(?:order|id)?\s*#?(\d{5,})\b")

    def extract_entities(self, text):
        entities = {}
        text_lower = text.lower()

        # Order Number
        order_match = self.order_number_pattern.search(text_lower)
        if order_match:
            entities["order_number"] = order_match.group(1)

        # Product Name (simple keyword detection for demonstration)
        if "bluetooth speaker" in text_lower:
            entities["product_name"] = "Bluetooth Speaker"
        elif "headphones" in text_lower:
            entities["product_name"] = "Headphones"
        elif "t-shirt" in text_lower:
            entities["product_name"] = "T-Shirt"

        # Refund reason (very basic)
        if "damaged" in text_lower or "broken" in text_lower:
            entities["reason"] = "damaged/broken"
        elif "wrong item" in text_lower:
            entities["reason"] = "wrong item received"

        return entities

class ToolLayer:
    def track_order(self, order_number):
        if order_number == "12345":
            return f"Order {order_number} is currently in transit and expected to arrive by {{DATE}}."
        return f"Could not find details for order {order_number}. Please check the number and try again."

    def get_product_info(self, product_name):
        if product_name == "Bluetooth Speaker":
            return "The Bluetooth Speaker features crisp sound, 10-hour battery life, and waterproof design."
        elif product_name == "Headphones":
            return "Our Headphones offer noise cancellation and superior comfort for long listening sessions."
        return f"Sorry, I don't have detailed information for '{product_name}'."

    def initiate_refund(self, order_number, reason):
        if order_number and reason:
            return f"Refund for order {order_number} for reason '{reason}' has been initiated. You will receive an email confirmation shortly."
        return "I need an order number and a reason to initiate a refund."

    def escalate_to_human(self, issue_description):
        return f"I'm escalating your issue: '{issue_description}' to a human agent. Please provide your contact details (email/phone) and an agent will reach out within 24 hours."

class DialogueManager:
    def __init__(self):
        self.intent_recognizer = IntentRecognizer()
        self.entity_extractor = EntityExtractor()
        self.tool_layer = ToolLayer()
        self.conversation_context = defaultdict(lambda: {"current_intent": None, "entities": {}})

    def process_query(self, user_id, user_query):
        context = self.conversation_context[user_id]
        
        intent, _ = self.intent_recognizer.predict_intent(user_query)
        entities = self.entity_extractor.extract_entities(user_query)

        response = ""

        # Update context with new information
        context["current_intent"] = intent
        context["entities"].update(entities)

        if intent == "greeting":
            response = "Hello! How can I assist you today?"
        elif intent == "track_order":
            order_number = context["entities"].get("order_number")
            if not order_number:
                response = "Please provide your order number so I can track it for you."
            else:
                response = self.tool_layer.track_order(order_number)
        elif intent == "product_inquiry":
            product_name = context["entities"].get("product_name")
            if not product_name:
                response = "Which product are you interested in?"
            else:
                response = self.tool_layer.get_product_info(product_name)
        elif intent == "refund_request":
            order_number = context["entities"].get("order_number")
            reason = context["entities"].get("reason")
            if not order_number:
                response = "To process a refund, I need your order number. Could you please provide it?"
            elif not reason:
                response = "What is the reason for the refund? (e.g., damaged, wrong item)"
            else:
                response = self.tool_layer.initiate_refund(order_number, reason)
        elif intent == "escalate":
            issue_description = user_query # For simplicity, use whole query as description
            response = self.tool_layer.escalate_to_human(issue_description)
        else:
            response = "I'm sorry, I didn't quite understand that. Can you please rephrase or ask something else?"
            # If unknown, clear context to avoid carrying irrelevant info for next query
            self.conversation_context[user_id] = defaultdict(lambda: {"current_intent": None, "entities": {}})

        return response

class Chatbot:
    def __init__(self):
        self.dialogue_manager = DialogueManager()
        self.user_id = "test_user_123" # Simulate a single user session

    def run(self):
        print("E-commerce Chatbot: Hello! How can I help you today? Type 'exit' to quit.")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                print("E-commerce Chatbot: Goodbye!")
                break
            
            response = self.dialogue_manager.process_query(self.user_id, user_input)
            print(f"E-commerce Chatbot: {response}")

if __name__ == "__main__":
    bot = Chatbot()
    bot.run()
