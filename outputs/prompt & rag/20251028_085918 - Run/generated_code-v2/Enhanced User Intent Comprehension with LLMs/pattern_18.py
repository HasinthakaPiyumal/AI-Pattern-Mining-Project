import time
import re
from transformers import pipeline

class IntentRecognizer:
    def __init__(self):
        # For demonstration, we'll use a simple text classification pipeline for sentiment
        # and layer keyword matching to simulate actual e-commerce intent recognition.
        # In a real-world scenario, this would be a model fine-tuned on e-commerce intents.
        try:
            self.nlp_sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst2-english", return_all_scores=True)
        except Exception as e:
            print(f"Warning: Could not load sentiment analysis model. Intent recognition will rely solely on keywords. Error: {e}")
            self.nlp_sentiment = None

        self.ecommerce_intents = [
            "order_status", "product_information", "return_item",
            "shipping_inquiry", "payment_issue", "greeting", "escalate_to_human", "unknown"
        ]
        self.keyword_to_intent = {
            "order": "order_status", "track": "order_status", "status": "order_status",
            "product": "product_information", "item": "product_information", "details": "product_information",
            "return": "return_item", "refund": "return_item",
            "ship": "shipping_inquiry", "delivery": "shipping_inquiry",
            "payment": "payment_issue", "bill": "payment_issue",
            "hello": "greeting", "hi": "greeting", "hey": "greeting", "good morning": "greeting",
            "help": "escalate_to_human", "human": "escalate_to_human", "agent": "escalate_to_human"
        }

    def _extract_entities(self, text, intent):
        entities = {}
        text_lower = text.lower()

        if intent == "order_status" or intent == "return_item":
            order_match = re.search(r'(?:order number|order id|order|#)?\s*(\d{6,})', text_lower)
            if order_match:
                entities["order_id"] = order_match.group(1)

        if intent == "product_information" or intent == "return_item":
            product_match = re.search(r'(?:product|item)\s*(?:named|called)?\s*"([^"]+)"|product\s+([A-Z][a-zA-Z0-9\s]+)|looking for (\w+\s?\w+)', text)
            if product_match:
                if product_match.group(1): entities["product_name"] = product_match.group(1)
                elif product_match.group(2): entities["product_name"] = product_match.group(2)
                elif product_match.group(3): entities["product_name"] = product_match.group(3)

        return entities

    def recognize_intent(self, text):
        detected_intent = "unknown"
        confidence = 0.5
        entities = self._extract_entities(text, detected_intent) # Extract entities initially without strong intent

        # Prioritize keyword matching for intent
        for keyword, intent in self.keyword_to_intent.items():
            if keyword in text.lower():
                detected_intent = intent
                confidence = 0.9 # High confidence for direct keyword match
                break

        # If a generic NLP sentiment model is available and no strong keyword match
        if self.nlp_sentiment and detected_intent == "unknown":
            nlp_results = self.nlp_sentiment(text)
            if nlp_results and nlp_results[0]:
                # Simple mapping: positive sentiment -> product info, negative -> order/return
                # This is a placeholder for a true intent classifier
                sentiment_label = max(nlp_results[0], key=lambda x: x["score"])
                if sentiment_label["label"] == "POSITIVE" and sentiment_label["score"] > 0.8:
                    detected_intent = "product_information"
                    confidence = sentiment_label["score"]
                elif sentiment_label["label"] == "NEGATIVE" and sentiment_label["score"] > 0.7:
                    detected_intent = "order_status" # Could be return, payment, etc. - requires clarification
                    confidence = sentiment_label["score"]
        
        # Re-extract entities with the potentially refined intent
        entities.update(self._extract_entities(text, detected_intent))

        return detected_intent, confidence, entities

class DialogueManager:
    def __init__(self):
        self.conversation_states = {}
        self.required_entities = {
            "order_status": ["order_id"],
            "return_item": ["order_id", "product_name"],
            "product_information": ["product_name"]
        }

    def get_user_state(self, user_id):
        return self.conversation_states.get(user_id, {"history": [], "current_intent": None, "missing_entities": []})

    def update_user_state(self, user_id, state):
        self.conversation_states[user_id] = state

    def manage_dialogue(self, user_id, user_input, intent, confidence, entities, conversation_history):
        current_state = self.get_user_state(user_id)
        response = ""
        action_required = None
        missing_info = []

        current_state["history"].append({"input": user_input, "intent": intent, "entities": entities})
        
        # If a previous intent was being pursued and this input contributes to it
        if current_state["current_intent"] and intent == "unknown" and current_state["missing_entities"]:
            # Try to fill missing entities for the current intent
            for entity_key in current_state["missing_entities"]:
                if entity_key == "order_id":
                    order_match = re.search(r'\b(\d{6,})\b', user_input)
                    if order_match: entities["order_id"] = order_match.group(1)
                elif entity_key == "product_name":
                    product_match = re.search(r'\b(?:product|item|about)\s*([a-zA-Z0-9\s]+)', user_input)
                    if product_match: entities["product_name"] = product_match.group(1).strip()
            intent = current_state["current_intent"] # Keep the old intent
            current_state["missing_entities"] = [] # Reset, will be re-evaluated

        current_state["current_intent"] = intent # Set or update current intent

        if intent == "greeting":
            response = "Hello! How can I assist you with your shopping today?"
            current_state["current_intent"] = None
        elif intent == "escalate_to_human":
            response = "I understand you need more help. I'm connecting you to a human agent now. Please wait a moment."
            action_required = "escalate"
            current_state["current_intent"] = None
        elif intent == "unknown" and confidence < 0.6:
            response = "I'm sorry, I didn't quite understand that. Could you please rephrase or tell me what you'd like to do?"
            current_state["current_intent"] = None
        else:
            if intent in self.required_entities:
                for entity_key in self.required_entities[intent]:
                    if entity_key not in entities or not entities[entity_key]:
                        missing_info.append(entity_key)

            if missing_info:
                if "order_id" in missing_info:
                    response = "To help you with your order, could you please provide your order number?"
                elif "product_name" in missing_info:
                    response = "What product are you interested in or referring to?"
                else:
                    response = "I need a bit more information to help with that. Could you clarify?"

                current_state["missing_entities"] = missing_info
                action_required = None
            else:
                if intent == "order_status":
                    response = f"Searching for your order {entities.get("order_id", "")}... "
                    action_required = "get_order_status"
                elif intent == "product_information":
                    response = f"Retrieving information for {entities.get("product_name", "the product")}. "
                    action_required = "get_product_info"
                elif intent == "return_item":
                    response = f"Initiating return process for {entities.get("product_name", "an item")} in order {entities.get("order_id", "")}. "
                    action_required = "initiate_return"
                elif intent == "shipping_inquiry":
                    response = "I can help with shipping inquiries. What specifically would you like to know about shipping?"
                    action_required = None
                elif intent == "payment_issue":
                    response = "I can help with payment issues. Please describe your payment issue in more detail."
                    action_required = None
                else:
                    response = "I'm not sure how to help with that. Could you rephrase or ask for something else?"
                    current_state["current_intent"] = None
            current_state["missing_entities"] = missing_info

        self.update_user_state(user_id, current_state)
        return response, action_required, missing_info

class ToolExecutor:
    def __init__(self):
        pass

    def get_order_status(self, order_id):
        time.sleep(1)
        if order_id == "123456":
            return f"Order {order_id} is currently 'Shipped' and expected to arrive by November 15, 2023."
        elif order_id == "789012":
            return f"Order {order_id} is 'Processing' and is expected to ship within 2 business days."
        else:
            return f"Could not find an order with ID {order_id}. Please check the number and try again."

    def get_product_info(self, product_name):
        time.sleep(1)
        product_name_lower = product_name.lower() if product_name else ""

        if "laptop" in product_name_lower:
            return f"The '{product_name}' is a high-performance gaming laptop with 16GB RAM and 1TB SSD storage. Price: $1500."
        elif "headphone" in product_name_lower:
            return f"The '{product_name}' are noise-cancelling wireless headphones with 30 hours battery life. Price: $250."
        elif "mouse" in product_name_lower:
            return f"The '{product_name}' is an ergonomic wireless mouse with 16000 DPI. Price: $75."
        else:
            return f"Sorry, I don't have detailed information on '{product_name}' at the moment. Please try a different product."

    def initiate_return(self, order_id, product_name):
        time.sleep(1)
        if order_id == "123456" and "laptop" in (product_name or "").lower():
            return f"Return for product '{product_name}' in order {order_id} has been successfully initiated. You will receive a return label via email shortly."
        else:
            return f"Unable to initiate return for product '{product_name}' in order {order_id}. Please ensure the order ID and product name are correct, or contact a human agent."

    def escalate(self):
        time.sleep(0.5)
        return "An agent will be with you shortly. Please explain your issue to them."

    def execute_tool(self, action, entities):
        if action == "get_order_status":
            return self.get_order_status(entities.get("order_id"))
        elif action == "get_product_info":
            return self.get_product_info(entities.get("product_name"))
        elif action == "initiate_return":
            return self.initiate_return(entities.get("order_id"), entities.get("product_name"))
        elif action == "escalate":
            return self.escalate()
        else:
            return "I apologize, I'm unable to perform that action at this time."

class CustomerProfileManager:
    def __init__(self):
        self.user_profiles = {}

    def get_history(self, user_id):
        return self.user_profiles.get(user_id, {"history": []})["history"]

    def update_history(self, user_id, interaction_data):
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {"history": [], "preferences": {}}
        self.user_profiles[user_id]["history"].append(interaction_data)
        if len(self.user_profiles[user_id]["history"]) > 20:
            self.user_profiles[user_id]["history"] = self.user_profiles[user_id]["history"][-20:]

    def get_user_preferences(self, user_id):
        return self.user_profiles.get(user_id, {}).get("preferences", {"preferred_language": "en"})

    def set_user_preference(self, user_id, key, value):
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {"history": [], "preferences": {}}
        if "preferences" not in self.user_profiles[user_id]:
            self.user_profiles[user_id]["preferences"] = {}
        self.user_profiles[user_id]["preferences"][key] = value
        return True

def main():
    print("Welcome to the Smart E-commerce Customer Support!\n")
    print("How can I help you today? (Type 'exit' to quit)")

    intent_recognizer = IntentRecognizer()
    dialogue_manager = DialogueManager()
    tool_executor = ToolExecutor()
    customer_profile_manager = CustomerProfileManager()

    user_id = "user_123" # For demonstration, a fixed user ID

    while True:
        user_input = input("> ")
        if user_input.lower() == 'exit':
            print("Thank you for contacting support. Goodbye!")
            break

        intent, confidence, entities = intent_recognizer.recognize_intent(user_input)
        # print(f"[DEBUG] Detected Intent: {intent} (Confidence: {confidence:.2f}), Entities: {entities}")

        response, action_required, missing_info = dialogue_manager.manage_dialogue(
            user_id, user_input, intent, confidence, entities, customer_profile_manager.get_history(user_id)
        )

        final_response_parts = [response]

        if action_required and not missing_info:
            tool_response = tool_executor.execute_tool(action_required, entities)
            final_response_parts.append(tool_response)
            customer_profile_manager.update_history(user_id, {"input": user_input, "intent": intent, "tool_response": tool_response})
        else:
            customer_profile_manager.update_history(user_id, {"input": user_input, "intent": intent, "agent_response": response})

        print(f"Agent: {' '.join(final_response_parts).strip()}")
        print("\n")

if __name__ == "__main__":
    main()