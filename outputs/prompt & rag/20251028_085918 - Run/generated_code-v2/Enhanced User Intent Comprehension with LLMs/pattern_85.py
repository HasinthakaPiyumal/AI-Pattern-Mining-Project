import re

class NLU:
    def __init__(self):
        self.intents = {
            "order_status": ["where is my order", "track my package", "order status", "when will my order arrive"],
            "product_inquiry": ["tell me about", "product details", "specifications of", "what is the price"],
            "return_policy": ["return an item", "how to return", "refund policy", "can I return"],
            "technical_support": ["technical issue", "help with a problem", "not working", "troubleshooting"],
            "billing_issue": ["billing problem", "payment issue", "charged incorrectly", "invoice question"],
            "general_greeting": ["hello", "hi", "hey", "good morning", "good afternoon"],
            "unknown": []
        }
        self.order_number_pattern = re.compile(r"\b(?:order|id)?\s*#?(\d{6,})\b", re.IGNORECASE)
        self.product_name_pattern = re.compile(r"(?:product|item)\s*(.+?)(?:\s|$)", re.IGNORECASE)

    def predict_intent(self, query):
        query_lower = query.lower()
        for intent, keywords in self.intents.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return intent, 0.9 # Simulate high confidence
        return "unknown", 0.5 # Simulate low confidence

    def extract_entities(self, query):
        entities = {}
        order_match = self.order_number_pattern.search(query)
        if order_match:
            entities["order_number"] = order_match.group(1)

        product_match = self.product_name_pattern.search(query)
        if product_match:
            entities["product_name"] = product_match.group(1).strip()
            
        return entities

class KnowledgeBase:
    def __init__(self):
        self.answers = {
            "order_status": "To check your order status, please provide your order number. If you have it, I can look it up for you.",
            "product_inquiry": "Please tell me the name of the product you are interested in, and I will try to provide more details.",
            "return_policy": "Our return policy allows returns within 30 days of purchase, provided the item is in its original condition. For more details, please visit our returns page on the website.",
            "technical_support": "For technical support, please describe your issue in detail, and I can connect you with a specialist or provide troubleshooting steps.",
            "billing_issue": "If you have a billing issue, please provide your account details or order number so I can assist you further.",
            "general_greeting": "Hello! How can I assist you today?",
            "unknown": "I'm sorry, I don't quite understand. Could you please rephrase your question or provide more details?"
        }

    def get_answer(self, intent):
        return self.answers.get(intent, self.answers["unknown"])

class DialogueManager:
    def __init__(self, nlu_module, kb_module):
        self.nlu = nlu_module
        self.kb = kb_module
        self.context = {}
        self.clarification_needed = False

    def process_query(self, query):
        self.context["last_query"] = query
        intent, confidence = self.nlu.predict_intent(query)
        entities = self.nlu.extract_entities(query)
        self.context["last_intent"] = intent
        self.context["last_entities"] = entities

        if confidence < 0.7 or intent == "unknown": # Low confidence or unknown intent
            self.clarification_needed = True
            return self.kb.get_answer("unknown")
        else:
            self.clarification_needed = False
            if intent == "order_status" and "order_number" in entities:
                order_number = entities["order_number"]
                # Simulate checking order status from a backend system
                return f"Checking status for order {order_number}. It is currently out for delivery and expected by tomorrow."
            elif intent == "product_inquiry" and "product_name" in entities:
                product_name = entities["product_name"]
                # Simulate fetching product details
                return f"The {product_name} is a high-quality item with excellent reviews. It costs $99.99 and is currently in stock."
            else:
                return self.kb.get_answer(intent)

class ChatbotUI:
    def __init__(self, dialogue_manager):
        self.dm = dialogue_manager

    def run(self):
        print("Welcome to our E-commerce Support Chatbot! Type 'exit' to quit.")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                print("Chatbot: Goodbye!")
                break
            response = self.dm.process_query(user_input)
            print(f"Chatbot: {response}")

if __name__ == "__main__":
    nlu_module = NLU()
    kb_module = KnowledgeBase()
    dialogue_manager = DialogueManager(nlu_module, kb_module)
    chatbot_ui = ChatbotUI(dialogue_manager)
    chatbot_ui.run()
