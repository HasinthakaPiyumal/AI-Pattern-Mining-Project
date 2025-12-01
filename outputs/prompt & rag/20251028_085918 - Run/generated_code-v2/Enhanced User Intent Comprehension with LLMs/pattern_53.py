from collections import defaultdict

class MockEcommerceAPI:
    def get_order_status(self, order_id):
        if order_id == "12345":
            return {"status": "Shipped", "estimated_delivery": "Tomorrow"}
        return {"status": "Not Found"}

    def return_item(self, order_id, product_name):
        if order_id == "12345" and product_name == "laptop":
            return {"success": True, "message": f"Return initiated for {product_name} from order {order_id}."}
        return {"success": False, "message": "Could not initiate return."}

    def get_product_info(self, product_name):
        if product_name == "smartphone":
            return {"name": "Smartphone X", "price": "$699", "features": "128GB, 6.1-inch display"}
        return {"name": "Not Found"}

    def update_shipping_address(self, order_id, new_address):
        if order_id == "12345":
            return {"success": True, "message": f"Shipping address for order {order_id} updated to {new_address}."}
        return {"success": False, "message": "Could not update shipping address."}

class IntentClassifier:
    def classify_intent(self, text):
        text_lower = text.lower()
        if "hello" in text_lower or "hi" in text_lower:
            return "GREETING"
        if "bye" in text_lower or "goodbye" in text_lower:
            return "FAREWELL"
        if "order status" in text_lower or "my order" in text_lower:
            return "ORDER_STATUS"
        if "return" in text_lower or "send back" in text_lower:
            return "RETURN_ITEM"
        if "product info" in text_lower or "about product" in text_lower or "details of" in text_lower:
            return "PRODUCT_INFO"
        if "update address" in text_lower or "change shipping" in text_lower:
            return "UPDATE_ADDRESS"
        return "AMBIGUOUS"

    def extract_entities(self, text, intent):
        entities = {}
        text_lower = text.lower()

        if intent == "ORDER_STATUS" or intent == "RETURN_ITEM" or intent == "UPDATE_ADDRESS":
            import re
            order_id_match = re.search(r"order\s*id\s*(\d+)", text_lower)
            if order_id_match:
                entities["order_id"] = order_id_match.group(1)
            else:
                order_id_match = re.search(r"#(\d+)", text_lower)
                if order_id_match:
                    entities["order_id"] = order_id_match.group(1)

        if intent == "RETURN_ITEM" or intent == "PRODUCT_INFO":
            if "laptop" in text_lower:
                entities["product_name"] = "laptop"
            elif "smartphone" in text_lower:
                entities["product_name"] = "smartphone"

        if intent == "UPDATE_ADDRESS":
            address_match = re.search(r"to\s+((?:\w+\s*){2,})", text_lower) # Simple regex for address
            if address_match:
                entities["new_address"] = address_match.group(1).strip()

        return entities

class DialogueManager:
    def __init__(self, api, classifier, personalization):
        self.api = api
        self.classifier = classifier
        self.personalization = personalization
        self.conversation_history = defaultdict(list)
        self.current_state = defaultdict(dict)

    def generate_response(self, user_id, intent, entities, tool_output=None):
        response = "I'm sorry, I couldn't understand that completely."
        user_prefs = self.personalization.get(user_id, {})
        
        if intent == "GREETING":
            response = "Hello! How can I assist you today?"
        elif intent == "FAREWELL":
            response = "Goodbye! Have a great day."
        elif intent == "ORDER_STATUS":
            order_id = entities.get("order_id")
            if order_id:
                result = self.api.get_order_status(order_id)
                if result["status"] != "Not Found":
                    response = f"Your order {order_id} is {result['status']}. Estimated delivery: {result['estimated_delivery']}."
                else:
                    response = f"I couldn't find any information for order ID {order_id}. Please double-check the ID."
            else:
                response = "Could you please provide the order ID?"
                self.current_state[user_id]["awaiting_info"] = "order_id"
                self.current_state[user_id]["prev_intent"] = intent
        elif intent == "RETURN_ITEM":
            order_id = entities.get("order_id")
            product_name = entities.get("product_name")
            if order_id and product_name:
                result = self.api.return_item(order_id, product_name)
                response = result["message"]
            else:
                missing_info = []
                if not order_id: missing_info.append("order ID")
                if not product_name: missing_info.append("product name")
                response = f"To process a return, I need the {', and '.join(missing_info)}. Can you provide them?"
                self.current_state[user_id]["awaiting_info"] = missing_info
                self.current_state[user_id]["prev_intent"] = intent
        elif intent == "PRODUCT_INFO":
            product_name = entities.get("product_name")
            if product_name:
                result = self.api.get_product_info(product_name)
                if result["name"] != "Not Found":
                    response = f"Product: {result['name']}. Price: {result['price']}. Features: {result['features']}."
                else:
                    response = f"I couldn't find information for {product_name}."
            else:
                response = "Which product are you interested in?"
                self.current_state[user_id]["awaiting_info"] = "product_name"
                self.current_state[user_id]["prev_intent"] = intent
        elif intent == "UPDATE_ADDRESS":
            order_id = entities.get("order_id")
            new_address = entities.get("new_address")
            if order_id and new_address:
                result = self.api.update_shipping_address(order_id, new_address)
                response = result["message"]
            else:
                missing_info = []
                if not order_id: missing_info.append("order ID")
                if not new_address: missing_info.append("new address")
                response = f"To update the shipping address, I need the {', and '.join(missing_info)}. Can you provide them?"
                self.current_state[user_id]["awaiting_info"] = missing_info
                self.current_state[user_id]["prev_intent"] = intent
        elif intent == "AMBIGUOUS":
            if user_id in self.current_state and "awaiting_info" in self.current_state[user_id]:
                prev_intent = self.current_state[user_id].get("prev_intent")
                awaiting = self.current_state[user_id]["awaiting_info"]
                if awaiting == "order_id":
                    new_order_id = self.classifier.extract_entities(tool_output, prev_intent).get("order_id")
                    if new_order_id:
                        entities["order_id"] = new_order_id
                        del self.current_state[user_id]["awaiting_info"]
                        return self.generate_response(user_id, prev_intent, entities, tool_output)
                elif awaiting == "product_name":
                    new_product_name = self.classifier.extract_entities(tool_output, prev_intent).get("product_name")
                    if new_product_name:
                        entities["product_name"] = new_product_name
                        del self.current_state[user_id]["awaiting_info"]
                        return self.generate_response(user_id, prev_intent, entities, tool_output)
                elif isinstance(awaiting, list) and "order ID" in awaiting and "new address" in awaiting: # For update_address
                    extracted = self.classifier.extract_entities(tool_output, prev_intent)
                    if "order_id" in extracted: entities["order_id"] = extracted["order_id"]
                    if "new_address" in extracted: entities["new_address"] = extracted["new_address"]
                    if all(item in entities for item in ["order_id", "new_address"]):
                        del self.current_state[user_id]["awaiting_info"]
                        return self.generate_response(user_id, prev_intent, entities, tool_output)
                elif isinstance(awaiting, list) and "order ID" in awaiting and "product name" in awaiting: # For return_item
                    extracted = self.classifier.extract_entities(tool_output, prev_intent)
                    if "order_id" in extracted: entities["order_id"] = extracted["order_id"]
                    if "product_name" in extracted: entities["product_name"] = extracted["product_name"]
                    if all(item in entities for item in ["order_id", "product_name"]):
                        del self.current_state[user_id]["awaiting_info"]
                        return self.generate_response(user_id, prev_intent, entities, tool_output)


            response = "I'm not sure I understand. Can you please rephrase or provide more details?"
        
        self.conversation_history[user_id].append((intent, entities, response))
        return response

class PersonalizationModule:
    def __init__(self):
        self.user_data = {}

    def get(self, user_id, default_value=None):
        return self.user_data.get(user_id, default_value)

    def update(self, user_id, key, value):
        if user_id not in self.user_data:
            self.user_data[user_id] = {}
        self.user_data[user_id][key] = value

class SmartCustomerSupportAgent:
    def __init__(self):
        self.api = MockEcommerceAPI()
        self.classifier = IntentClassifier()
        self.personalization = PersonalizationModule()
        self.dialogue_manager = DialogueManager(self.api, self.classifier, self.personalization)

    def process_query(self, user_id, query):
        print(f"User {user_id}: {query}")

        current_intent = self.classifier.classify_intent(query)
        current_entities = self.classifier.extract_entities(query, current_intent)

        response = self.dialogue_manager.generate_response(user_id, current_intent, current_entities, query)
        print(f"Agent: {response}")
        return response


if __name__ == "__main__":
    agent = SmartCustomerSupportAgent()
    user_id_1 = "user123"

    agent.process_query(user_id_1, "Hi there!")
    agent.process_query(user_id_1, "What's the status of my order ID 12345?")
    agent.process_query(user_id_1, "I want to return a laptop from order #12345.")
    agent.process_query(user_id_1, "Tell me about a smartphone.")
    agent.process_query(user_id_1, "Can you update the shipping address for order 12345 to 123 Main Street?")
    agent.process_query(user_id_1, "How about order 99999?")
    agent.process_query(user_id_1, "I need to return something, but what was the product?")
    agent.process_query(user_id_1, "It was a laptop.")
    agent.process_query(user_id_1, "I want to update an address, what's my order id?")
    agent.process_query(user_id_1, "My order is #12345, to 456 Oak Avenue")
    agent.process_query(user_id_1, "Goodbye")

    print("\n--- Another User ---")
    user_id_2 = "user456"
    agent.process_query(user_id_2, "Hello!")
    agent.process_query(user_id_2, "status of my order please")
    agent.process_query(user_id_2, "It's #98765")
    agent.process_query(user_id_2, "What is a smartphone?")
