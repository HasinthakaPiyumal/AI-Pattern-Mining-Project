import random

class NLUModule:
    def __init__(self):
        self.intents = {
            "check_order_status": ["where is my order", "order status", "track my package"],
            "product_inquiry": ["tell me about product X", "product details", "specifications of Y"],
            "return_request": ["i want to return an item", "how to return", "initiate return"],
            "shipping_info": ["shipping cost", "delivery options", "how long for delivery"],
            "escalate_to_human": ["talk to a representative", "speak to a human", "can i talk to someone"]
        }

    def recognize_intent(self, text):
        text_lower = text.lower()
        detected_intents = {}
        for intent, phrases in self.intents.items():
            for phrase in phrases:
                if phrase in text_lower:
                    detected_intents[intent] = detected_intents.get(intent, 0) + 0.9 / len(phrases) # Simulate confidence

        if not detected_intents:
            return {"unknown": 1.0}

        # Simple confidence normalization and selection for demonstration
        total_confidence = sum(detected_intents.values())
        if total_confidence > 0:
            for intent in detected_intents:
                detected_intents[intent] /= total_confidence

        # Sort by confidence and return top few or single best
        sorted_intents = sorted(detected_intents.items(), key=lambda item: item[1], reverse=True)
        
        if sorted_intents[0][1] < 0.6 and len(sorted_intents) > 1: # Simulate low confidence and multiple possibilities
            return {"ambiguous": sorted_intents[:2]} # Return top 2 ambiguous intents
        else:
            return {sorted_intents[0][0]: sorted_intents[0][1]}

class KnowledgeBase:
    def get_order_status(self, order_id):
        statuses = {"12345": "Shipped on Jan 10th, expected delivery Jan 15th.",
                    "67890": "Processing, will ship within 2 business days.",
                    "11223": "Delivered on Jan 5th."}
        return statuses.get(order_id, "Order ID not found.")

    def get_product_info(self, product_name):
        products = {"laptop": "High-performance laptop with 16GB RAM and 512GB SSD. Price: $1200.",
                    "mouse": "Wireless ergonomic mouse. Price: $30.",
                    "keyboard": "Mechanical RGB gaming keyboard. Price: $80."}
        return products.get(product_name.lower(), "Product information not available.")

    def get_shipping_info(self):
        return "Standard shipping takes 3-5 business days. Express shipping is available for an extra charge and takes 1-2 business days."

class PersonalizationModule:
    def __init__(self):
        self.user_profiles = {}

    def get_style(self, user_id):
        # Simulate different styles based on user_id or history
        return self.user_profiles.get(user_id, {"formality": "neutral", "verbosity": "medium"})

    def update_style(self, user_id, new_style):
        self.user_profiles[user_id] = new_style

class DialogueManager:
    def __init__(self):
        self.nlu = NLUModule()
        self.kb = KnowledgeBase()
        self.personalization = PersonalizationModule()
        self.context = {}

    def process_query(self, user_id, query):
        self.context[user_id] = self.context.get(user_id, {}) # Initialize user context
        intent_recognition_result = self.nlu.recognize_intent(query)
        user_style = self.personalization.get_style(user_id)

        response = "I'm sorry, I couldn't understand that. Could you please rephrase?"

        if "ambiguous" in intent_recognition_result:
            ambiguous_intents = intent_recognition_result["ambiguous"]
            intent_names = [intent[0].replace('_', ' ') for intent in ambiguous_intents]
            response = f"I'm not sure if you mean {' or '.join(intent_names)}. Could you clarify?"

        elif "unknown" in intent_recognition_result:
            response = "I'm still learning and don't understand that request. Could you try asking something else, or would you like to speak to a human?"

        else:
            intent = next(iter(intent_recognition_result))
            confidence = intent_recognition_result[intent]

            if intent == "check_order_status":
                order_id = self._extract_order_id(query)
                if order_id:
                    status = self.kb.get_order_status(order_id)
                    response = f"The status for order {order_id} is: {status}"
                else:
                    response = "Please provide your order ID so I can check its status."
            elif intent == "product_inquiry":
                product_name = self._extract_product_name(query)
                if product_name:
                    info = self.kb.get_product_info(product_name)
                    response = f"Here is some information about {product_name}: {info}"
                else:
                    response = "Which product are you interested in?"
            elif intent == "return_request":
                response = "To initiate a return, please visit our returns page at www.example.com/returns and follow the instructions. Do you need a link to the page?"
            elif intent == "shipping_info":
                response = self.kb.get_shipping_info()
            elif intent == "escalate_to_human":
                response = "Connecting you to a human agent now. Please wait a moment."

        return self._apply_personalization(response, user_style)

    def _extract_order_id(self, query):
        # Simple regex for a 5-digit number for demo purposes
        import re
        match = re.search(r'\b(\d{5})\b', query)
        return match.group(1) if match else None

    def _extract_product_name(self, query):
        # Simple keyword extraction for demo
        for product in ["laptop", "mouse", "keyboard"]:
            if product in query.lower():
                return product
        return None

    def _apply_personalization(self, response, style):
        if style["formality"] == "formal":
            response = f"Dear customer, {response}"
        elif style["formality"] == "informal":
            response = f"Hey there! {response}"

        if style["verbosity"] == "concise" and len(response) > 80:
            response = response[:80] + "..."
        return response

class ChatbotInterface:
    def __init__(self):
        self.dialogue_manager = DialogueManager()
        self.user_id = "user_123" # Fixed user ID for simple demo

    def run(self):
        print("Welcome to our E-commerce Customer Support Chatbot! Type 'exit' to quit.")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                print("Thank you for chatting with us. Goodbye!")
                break
            
            response = self.dialogue_manager.process_query(self.user_id, user_input)
            print(f"Bot: {response}")

if __name__ == "__main__":
    bot = ChatbotInterface()
    bot.run()