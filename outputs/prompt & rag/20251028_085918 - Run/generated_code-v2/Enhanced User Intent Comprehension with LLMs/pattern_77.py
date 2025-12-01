import random
import json

class NLUModule:
    def __init__(self):
        self.intents = {
            "check_order_status": ["order status", "where is my order", "track my package"],
            "initiate_return": ["return an item", "how to return", "start a return"],
            "product_inquiry": ["tell me about", "product details", "specifications"],
            "technical_support": ["technical issue", "troubleshoot", "doesn't work"],
            "billing_query": ["billing issue", "charge", "invoice"],
            "account_management": ["change my address", "update profile", "login issue"]
        }
        self.entities = {
            "order_id": ["#\d{6,}", "order number \d+"],
            "product_name": ["laptop", "headphones", "smartphone", "keyboard"],
            "customer_id": ["customer \d+"]
        }

    def predict_intent_entities(self, query):
        query = query.lower()
        recognized_intent = "unknown"
        confidence = 0.5
        extracted_entities = {}

        # Mock intent recognition
        for intent, keywords in self.intents.items():
            for keyword in keywords:
                if keyword in query:
                    recognized_intent = intent
                    confidence = 0.8 + random.random() * 0.1 # Simulate higher confidence
                    break
            if recognized_intent != "unknown":
                break

        # Mock entity extraction
        for entity_type, patterns in self.entities.items():
            for pattern in patterns:
                import re
                match = re.search(pattern, query)
                if match:
                    extracted_entities[entity_type] = match.group(0)

        if recognized_intent == "unknown":
            confidence = 0.3 + random.random() * 0.1 # Simulate lower confidence

        return {"intent": recognized_intent, "entities": extracted_entities, "confidence": confidence}

class DialogueManager:
    def __init__(self, threshold=0.7):
        self.confidence_threshold = threshold
        self.conversation_state = {}

    def manage_dialogue(self, nlu_output, customer_id):
        self.conversation_state[customer_id] = self.conversation_state.get(customer_id, {})

        if nlu_output["confidence"] < self.confidence_threshold or nlu_output["intent"] == "unknown":
            if nlu_output["intent"] == "unknown":
                return "I'm not sure I understood your request. Could you please rephrase or be more specific?"
            else:
                return f"I'm not entirely sure about your intent. Are you trying to {nlu_output['intent'].replace('_', ' ')}?"
        return None # No clarification needed

class ActionInformationRetrieval:
    def __init__(self):
        self.knowledge_base = {
            "product_inquiry": {
                "laptop": "Our latest laptop features an Intel i7 processor, 16GB RAM, and a 512GB SSD. It's great for both work and gaming.",
                "headphones": "These noise-cancelling headphones offer up to 30 hours of battery life and superior sound quality."
            },
            "faq": {
                "return_policy": "You can return most items within 30 days of purchase for a full refund. Please ensure the item is in its original packaging.",
                "shipping_times": "Standard shipping usually takes 3-5 business days. Expedited options are available at checkout."
            }
        }
        self.mock_backend_api = {
            "check_order_status": lambda order_id: f"Order {order_id} is currently in transit and expected to arrive by {{DATE}}.",
            "initiate_return": lambda order_id: f"Return initiated for order {order_id}. You will receive an email with instructions shortly."
        }

    def execute_action(self, intent, entities):
        if intent == "check_order_status" and "order_id" in entities:
            return self.mock_backend_api["check_order_status"](entities["order_id"])
        elif intent == "initiate_return" and "order_id" in entities:
            return self.mock_backend_api["initiate_return"](entities["order_id"])
        elif intent == "product_inquiry" and "product_name" in entities:
            product_info = self.knowledge_base["product_inquiry"].get(entities["product_name"], "I couldn't find details for that specific product.")
            return product_info
        elif intent == "technical_support":
            return "Please wait while I connect you to a technical support agent."
        elif intent == "billing_query":
            return "I'm routing you to our billing department. A representative will be with you shortly."
        elif intent == "account_management":
            return "For account management, please visit your profile settings on our website or app."
        
        # Fallback for general inquiries or unknown intents that might be covered by FAQ
        if intent == "unknown" and any(kw in entities.values() for kw in self.knowledge_base["faq"].keys()):
             for kw, answer in self.knowledge_base["faq"].items():
                 if kw in entities.values(): # Simplified check for demo
                     return answer

        return "I can help with that, but I need more specific information. Can you tell me more?"

class PersonalizationModule:
    def __init__(self):
        self.customer_data = {
            "cust123": {"name": "Alice", "email": "alice@example.com", "past_purchases": ["laptop", "mouse"], "preferences": {"category": "electronics"}},
            "cust456": {"name": "Bob", "email": "bob@example.com", "past_purchases": ["headphones"], "preferences": {"category": "audio"}}
        }

    def get_customer_data(self, customer_id):
        return self.customer_data.get(customer_id, {})

    def personalize_response(self, customer_id, response, intent=None):
        customer_info = self.get_customer_data(customer_id)
        if not customer_info:
            return response

        personalized_response = response
        if customer_info.get("name") and "{{NAME}}" in personalized_response:
            personalized_response = personalized_response.replace("{{NAME}}", customer_info["name"])

        if intent == "product_inquiry" and customer_info.get("preferences", {}).get("category") == "electronics":
            personalized_response += " By the way, we also have new accessories in the electronics category you might like!"

        return f"Hello {customer_info['name']}, {personalized_response}" if customer_info.get("name") else personalized_response


class SmartCustomerSupportAssistant:
    def __init__(self):
        self.nlu = NLUModule()
        self.dialogue_manager = DialogueManager()
        self.action_retrieval = ActionInformationRetrieval()
        self.personalization = PersonalizationModule()

    def process_query(self, customer_id, query):
        print(f"\nCustomer {customer_id} Query: {query}")

        # 1. NLU Module
        nlu_output = self.nlu.predict_intent_entities(query)
        print(f"NLU Output: {json.dumps(nlu_output, indent=2)}")

        # 2. Dialogue Management & Clarification
        clarification_needed = self.dialogue_manager.manage_dialogue(nlu_output, customer_id)
        if clarification_needed:
            return self.personalization.personalize_response(customer_id, clarification_needed)

        # 3. Action & Information Retrieval
        response = self.action_retrieval.execute_action(nlu_output["intent"], nlu_output["entities"])

        # 4. Personalization Module
        final_response = self.personalization.personalize_response(customer_id, response, intent=nlu_output["intent"])

        return final_response


if __name__ == "__main__":
    assistant = SmartCustomerSupportAssistant()

    # --- Test Cases ---

    # Test Case 1: Clear Order Status Inquiry
    print(assistant.process_query("cust123", "What is the status of my order #123456?"))

    # Test Case 2: Ambiguous Product Inquiry (leading to clarification in a real scenario, but here, it might just pick 'product_inquiry')
    print(assistant.process_query("cust456", "Tell me about your best headphones."))

    # Test Case 3: Initiate Return
    print(assistant.process_query("cust123", "I want to return an item, my order number is 987654."))

    # Test Case 4: Technical Support Request
    print(assistant.process_query("cust456", "My new laptop isn't turning on, I need technical support."))

    # Test Case 5: Billing Query
    print(assistant.process_query("cust123", "I have a question about my last bill."))

    # Test Case 6: Vague Query (should trigger clarification)
    print(assistant.process_query("cust456", "I need help with something."))

    # Test Case 7: Product Inquiry with specific product
    print(assistant.process_query("cust123", "What are the specifications of the laptop?"))

    # Test Case 8: Unknown Order ID (simulated entity extraction failure)
    print(assistant.process_query("cust123", "Where is my order?"))

    # Test Case 9: Account Management
    print(assistant.process_query("cust456", "How do I change my shipping address?"))

    # Test Case 10: New Customer, general query
    print(assistant.process_query("cust789", "What is your return policy?"))
