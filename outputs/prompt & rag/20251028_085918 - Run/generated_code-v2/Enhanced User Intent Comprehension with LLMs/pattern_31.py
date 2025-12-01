from typing import Dict, Any, Callable

class ECommerceSystem:
    def get_order_details(self, order_id: str) -> str:
        if order_id == "ORDER123":
            return f"Order {order_id}: Status is 'Shipped', estimated delivery on 2023-10-27. Items: Laptop (x1), Mouse (x1)."
        return f"Sorry, I couldn't find details for order ID: {order_id}. Please check the ID and try again."

    def process_return(self, order_id: str, product_name: str) -> str:
        if order_id == "ORDER123" and product_name in ["Laptop", "Mouse"]:
            return f"Return initiated for {product_name} from order {order_id}. You will receive an email with return instructions shortly."
        return f"Unable to process return for {product_name} in order {order_id}. Please ensure the order and product details are correct."

    def fetch_product_data(self, product_name: str) -> str:
        product_data = {
            "Laptop": "High-performance laptop with 16GB RAM and 512GB SSD. Price: $1200.",
            "Mouse": "Ergonomic wireless mouse with customizable buttons. Price: $50."
        }
        if product_name in product_data:
            return f"Product '{product_name}': {product_data[product_name]}"
        return f"Sorry, I couldn't find information for product: {product_name}."

    def update_delivery_address(self, order_id: str, new_address: str) -> str:
        if order_id == "ORDER123":
            return f"Delivery address for order {order_id} has been updated to '{new_address}'. Please note, changes might take up to 24 hours to reflect."
        return f"Could not update delivery address for order {order_id}. Please verify the order ID."


class LLMSimulator:
    def predict_intent(self, query: str) -> Dict[str, Any]:
        query = query.lower()
        
        if "track" in query or "where is my order" in query or "order status" in query:
            order_id = self._extract_parameter(query, "order (id|number|no)? (.*)", 2) or "ORDER123" # Simulate finding a default or example
            return {"intent": "track_order", "parameters": {"order_id": order_id}, "confidence": 0.9, "ambiguous": False}
        
        if "return" in query or "send back" in query:
            order_id = self._extract_parameter(query, "order (id|number|no)? (.*)", 2) or "ORDER123"
            product_name = self._extract_parameter(query, "(product|item) (.*) from", 2) or "Laptop"
            if not order_id or not product_name:
                return {"intent": "clarify", "parameters": {}, "confidence": 0.7, "ambiguous": True, "clarification_needed": "Please provide both the order ID and the product name for the return.", "original_intent": "initiate_return"}
            return {"intent": "initiate_return", "parameters": {"order_id": order_id, "product_name": product_name}, "confidence": 0.9, "ambiguous": False}

        if "product info" in query or "details about" in query or "tell me about" in query:
            product_name = self._extract_parameter(query, "(about|for) (.*)") or "Laptop"
            if not product_name:
                 return {"intent": "clarify", "parameters": {}, "confidence": 0.7, "ambiguous": True, "clarification_needed": "Which product are you interested in?", "original_intent": "product_info"}
            return {"intent": "product_info", "parameters": {"product_name": product_name}, "confidence": 0.9, "ambiguous": False}

        if "change address" in query or "update delivery" in query:
            order_id = self._extract_parameter(query, "order (id|number|no)? (.*)", 2) or "ORDER123"
            new_address = self._extract_parameter(query, "to (.*)") or "123 Main St, Anytown, USA"
            if not order_id or not new_address:
                return {"intent": "clarify", "parameters": {}, "confidence": 0.7, "ambiguous": True, "clarification_needed": "Please provide the order ID and the new address.", "original_intent": "update_delivery_address"}
            return {"intent": "update_delivery_address", "parameters": {"order_id": order_id, "new_address": new_address}, "confidence": 0.9, "ambiguous": False}

        return {"intent": "unknown", "parameters": {}, "confidence": 0.5, "ambiguous": True, "clarification_needed": "I'm not sure how to help with that. Can you rephrase or ask about tracking orders, returns, product information, or updating delivery addresses?"}

    def _extract_parameter(self, text: str, pattern: str, group: int = 1) -> str or None:
        import re
        match = re.search(pattern, text)
        if match and len(match.groups()) >= group:
            return match.group(group).strip()
        return None


class CustomerSupportAgent:
    def __init__(self):
        self.ecommerce_system = ECommerceSystem()
        self.llm_simulator = LLMSimulator()
        self.tool_registry: Dict[str, Callable] = {
            "track_order": self.ecommerce_system.get_order_details,
            "initiate_return": self.ecommerce_system.process_return,
            "product_info": self.ecommerce_system.fetch_product_data,
            "update_delivery_address": self.ecommerce_system.update_delivery_address,
        }
        self.user_history: Dict[str, Any] = {}
        self.current_ambiguous_intent: Dict[str, Any] = {}

    def handle_query(self, query: str) -> str:
        if self.current_ambiguous_intent:
            # Attempt to resolve previous ambiguity
            original_intent_info = self.current_ambiguous_intent
            resolved_response = self._resolve_ambiguity(query, original_intent_info)
            if resolved_response:
                self.current_ambiguous_intent = {}
                return resolved_response

        intent_prediction = self.llm_simulator.predict_intent(query)
        intent = intent_prediction.get("intent")
        parameters = intent_prediction.get("parameters", {})
        ambiguous = intent_prediction.get("ambiguous", False)
        clarification_needed = intent_prediction.get("clarification_needed", "")

        if ambiguous and intent == "clarify": # Specific clarification needed by LLM
            self.current_ambiguous_intent = intent_prediction
            return clarification_needed

        if ambiguous:
            # General ambiguity or unknown intent
            return clarification_needed if clarification_needed else "I'm having trouble understanding. Could you please provide more details?"

        if intent == "unknown":
            return clarification_needed if clarification_needed else "I'm sorry, I don't understand that request. Can you ask about something else?"

        if intent in self.tool_registry:
            try:
                # Basic personalization: remember last order ID if provided
                if "order_id" in parameters and "last_order_id" not in self.user_history:
                    self.user_history["last_order_id"] = parameters["order_id"]
                elif "order_id" not in parameters and "last_order_id" in self.user_history:
                    # Use last known order_id if not explicitly provided in current query
                    parameters["order_id"] = self.user_history["last_order_id"]

                # Call the corresponding tool function
                result = self.tool_registry[intent](**parameters)
                return result
            except TypeError as e:
                return f"There was an error processing your request due to missing information: {e}. Please provide all necessary details."
            except Exception as e:
                return f"An unexpected error occurred: {e}"
        else:
            return "I'm sorry, I don't have a tool to handle that specific request."

    def _resolve_ambiguity(self, clarification_response: str, original_intent_info: Dict[str, Any]) -> str or None:
        original_intent = original_intent_info.get("original_intent")
        required_params_clarified = self._parse_clarification(clarification_response, original_intent_info)

        if original_intent and required_params_clarified:
            # Merge new parameters with any existing ones from the original query
            all_params = {**original_intent_info.get("parameters", {}), **required_params_clarified}
            try:
                # Call the corresponding tool function with resolved parameters
                result = self.tool_registry[original_intent](**all_params)
                return result
            except TypeError as e:
                return f"Still missing some information after clarification: {e}. Can you try again?"
            except Exception as e:
                return f"An unexpected error occurred during clarification: {e}"
        return None

    def _parse_clarification(self, response: str, original_intent_info: Dict[str, Any]) -> Dict[str, Any]:
        parsed_params = {}
        response = response.lower()

        if original_intent_info.get("original_intent") == "initiate_return":
            order_id = self.llm_simulator._extract_parameter(response, "order (id|number|no)? (.*)", 2)
            product_name = self.llm_simulator._extract_parameter(response, "(product|item) (.*)")
            if order_id: parsed_params["order_id"] = order_id
            if product_name: parsed_params["product_name"] = product_name

        elif original_intent_info.get("original_intent") == "product_info":
            product_name = self.llm_simulator._extract_parameter(response, "(product|item) (.*)")
            if product_name: parsed_params["product_name"] = product_name
        
        elif original_intent_info.get("original_intent") == "update_delivery_address":
            order_id = self.llm_simulator._extract_parameter(response, "order (id|number|no)? (.*)", 2)
            new_address = self.llm_simulator._extract_parameter(response, "to (.*)")
            if order_id: parsed_params["order_id"] = order_id
            if new_address: parsed_params["new_address"] = new_address

        return parsed_params


if __name__ == "__main__":
    agent = CustomerSupportAgent()
    print("Welcome to E-commerce Support! Ask me about your orders, returns, products, or updating delivery addresses. Type 'exit' to quit.")

    while True:
        user_query = input("You: ")
        if user_query.lower() == 'exit':
            print("Agent: Goodbye!")
            break
        
        response = agent.handle_query(user_query)
        print(f"Agent: {response}")
