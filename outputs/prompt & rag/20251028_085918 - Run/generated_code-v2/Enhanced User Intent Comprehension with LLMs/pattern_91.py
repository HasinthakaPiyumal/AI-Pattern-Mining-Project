class ECommerceSupportAgent:
    def __init__(self):
        pass

    def _recognize_intent_and_entities(self, query):
        query = query.lower()
        intent = "unknown"
        entities = {}

        if "order status" in query or "where is my order" in query:
            intent = "order_status"
            # Simple entity extraction for order_id
            for word in query.split():
                if word.isdigit() and len(word) == 6: # Assuming 6-digit order IDs
                    entities["order_id"] = word
                    break
        elif "product" in query and ("details" in query or "info" in query):
            intent = "product_inquiry"
            # Simple entity extraction for product_name
            words = query.split()
            if "product" in words and words.index("product") + 1 < len(words):
                entities["product_name"] = words[words.index("product") + 1]
        elif "return" in query or "refund" in query:
            intent = "return_request"
            for word in query.split():
                if word.isdigit() and len(word) == 6: # Assuming 6-digit order IDs for returns
                    entities["order_id"] = word
                    break
        elif "shipping" in query or "delivery" in query:
            intent = "shipping_info"
            for word in query.split():
                if word.isdigit() and len(word) == 6: # Assuming 6-digit order IDs for shipping
                    entities["order_id"] = word
                    break
        elif "speak to human" in query or "talk to agent" in query or "escalate" in query:
            intent = "escalate_to_human"
        
        return intent, entities

    def _get_order_status(self, order_id):
        if order_id:
            return f"Simulating: Fetching status for order {order_id}. Current status: Shipped, expected delivery tomorrow."
        return "Simulating: I need an order ID to check the status."

    def _get_product_details(self, product_name):
        if product_name:
            return f"Simulating: Retrieving details for {product_name}. It is a high-quality item with excellent reviews."
        return "Simulating: Please specify the product name."

    def _initiate_return(self, order_id, reason=None):
        if order_id:
            return f"Simulating: Initiating return process for order {order_id}. A return label will be sent to your email. Reason: {reason if reason else 'Not specified'}."
        return "Simulating: I need an order ID to initiate a return."
    
    def _get_shipping_info(self, order_id):
        if order_id:
            return f"Simulating: Providing shipping information for order {order_id}. Your package is currently in transit. Tracking number: TRK123456789."
        return "Simulating: I need an order ID to get shipping details."

    def _escalate_to_human(self):
        return "Simulating: Connecting you to a human agent. Please wait a moment."

    def _generate_response(self, intent, entities, tool_result):
        if intent == "order_status":
            if "order_id" in entities:
                return f"No problem! {tool_result}"
            else:
                return "Could you please provide your 6-digit order ID so I can check its status?"
        elif intent == "product_inquiry":
            if "product_name" in entities:
                return f"Here are the details you requested: {tool_result}"
            else:
                return "What product are you interested in? Please tell me the name."
        elif intent == "return_request":
            if "order_id" in entities:
                return f"Alright, {tool_result}"
            else:
                return "To help you with a return, could you please provide your 6-digit order ID?"
        elif intent == "shipping_info":
            if "order_id" in entities:
                return f"I can help with that. {tool_result}"
            else:
                return "Please provide your 6-digit order ID to get shipping information."
        elif intent == "escalate_to_human":
            return tool_result
        else:
            return "I'm sorry, I couldn't understand your request. Could you please rephrase it or ask for a human agent?"

    def process_query(self, query):
        intent, entities = self._recognize_intent_and_entities(query)
        tool_result = None

        if intent == "order_status":
            tool_result = self._get_order_status(entities.get("order_id"))
        elif intent == "product_inquiry":
            tool_result = self._get_product_details(entities.get("product_name"))
        elif intent == "return_request":
            tool_result = self._initiate_return(entities.get("order_id"))
        elif intent == "shipping_info":
            tool_result = self._get_shipping_info(entities.get("order_id"))
        elif intent == "escalate_to_human":
            tool_result = self._escalate_to_human()
        else:
            # For unknown intent, we might still try to generate a fallback response
            pass

        return self._generate_response(intent, entities, tool_result)

if __name__ == "__main__":
    agent = ECommerceSupportAgent()
    print("Welcome to E-commerce Customer Support! How can I help you today? (Type 'exit' to quit)")

    while True:
        user_query = input("You: ")
        if user_query.lower() == 'exit':
            print("Agent: Goodbye!")
            break

        response = agent.process_query(user_query)
        print(f"Agent: {response}")
