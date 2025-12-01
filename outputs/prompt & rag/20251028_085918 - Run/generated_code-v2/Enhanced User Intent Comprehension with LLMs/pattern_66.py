class EcommerceAssistant:
    def __init__(self):
        self.products = {
            "laptop": {"price": "$1200", "category": "electronics"},
            "smartphone": {"price": "$800", "category": "electronics"},
            "headset": {"price": "$150", "category": "accessories"},
            "keyboard": {"price": "$75", "category": "accessories"},
        }
        self.orders = {
            "12345": {"item": "laptop", "status": "shipped"},
            "67890": {"item": "smartphone", "status": "processing"},
        }
        self.current_intent = None
        self.current_entities = {}

    def _nlu_process(self, query):
        query = query.lower()
        intent = "unknown"
        entities = {}

        if "search" in query or "find" in query or "look for" in query:
            intent = "product_search"
            for product_name in self.products:
                if product_name in query:
                    entities["product"] = product_name
                    break
        elif "order" in query or "status" in query or "where is" in query:
            intent = "order_status"
            order_ids = [k for k in self.orders.keys()]
            for order_id in order_ids:
                if order_id in query:
                    entities["order_id"] = order_id
                    break
        elif "help" in query or "support" in query or "agent" in query:
            intent = "customer_support"
        elif "hello" in query or "hi" in query:
            intent = "greet"

        return intent, entities

    def _dialogue_manager(self, intent, entities):
        if intent == "product_search":
            if "product" not in entities:
                return "What product are you looking for?", "ask_product"
        elif intent == "order_status":
            if "order_id" not in entities:
                return "Please provide your order ID.", "ask_order_id"
        self.current_intent = intent
        self.current_entities.update(entities)
        return None, "proceed"

    def _execute_action(self, intent, entities):
        if intent == "product_search":
            product = entities.get("product")
            if product and product in self.products:
                info = self.products[product]
                return f"We have {product} in {info['category']} for {info['price']}."
            else:
                return f"Sorry, I couldn't find {product} in our catalog."
        elif intent == "order_status":
            order_id = entities.get("order_id")
            if order_id and order_id in self.orders:
                order_info = self.orders[order_id]
                return f"Your order {order_id} for {order_info['item']} is currently {order_info['status']}."
            else:
                return f"Sorry, I couldn't find an order with ID {order_id}."
        elif intent == "customer_support":
            return "Connecting you to a customer support agent. Please wait."
        elif intent == "greet":
            return "Hello! How can I help you today?"
        else:
            return "I'm sorry, I didn't understand that. Can you please rephrase?"

    def process_query(self, query):
        if self.current_intent and self.current_entities:
            # If we are in a clarifying state, try to use the new input as an entity
            if self.current_intent == "product_search" and not self.current_entities.get("product"):
                for product_name in self.products:
                    if product_name in query.lower():
                        self.current_entities["product"] = product_name
                        break
            elif self.current_intent == "order_status" and not self.current_entities.get("order_id"):
                order_ids = [k for k in self.orders.keys()]
                for order_id in order_ids:
                    if order_id in query:
                        self.current_entities["order_id"] = order_id
                        break
            
            if self.current_intent and (self.current_entities.get("product") or self.current_entities.get("order_id") or self.current_intent == "customer_support"):
                response = self._execute_action(self.current_intent, self.current_entities)
                self.current_intent = None # Reset state
                self.current_entities = {}
                return response
            else:
                return "Still need more information. Can you be more specific?"

        intent, entities = self._nlu_process(query)
        clarification_needed, action = self._dialogue_manager(intent, entities)

        if clarification_needed:
            self.current_intent = intent # Store intent for follow-up
            self.current_entities = entities # Store entities for follow-up
            return clarification_needed
        elif action == "proceed":
            response = self._execute_action(intent, entities)
            self.current_intent = None
            self.current_entities = {}
            return response
        else:
            return "I'm sorry, I didn't understand that. Can you please rephrase?"


def main():
    assistant = EcommerceAssistant()
    print("Welcome to the Smart E-commerce Assistant! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Assistant: Goodbye!")
            break
        response = assistant.process_query(user_input)
        print(f"Assistant: {response}")

if __name__ == "__main__":
    main()