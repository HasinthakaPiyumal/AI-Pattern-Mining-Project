
import json
import random
from typing import Dict, Any, List, Callable
import inspect

# data_handler.py
def load_ecommerce_data():
    print("Loading simulated e-commerce interaction data...")
    return [
        {"query": "Where is my order?", "intent": "check_order_status", "parameters": {"order_id": None}},
        {"query": "I want to return a product.", "intent": "process_return", "parameters": {"product_name": None}},
        {"query": "My delivery is late.", "intent": "check_order_status", "parameters": {"order_id": None}},
        {"query": "How can I track my package?", "intent": "track_shipping", "parameters": {"order_id": None}},
        {"query": "I need help with a refund.", "intent": "process_return", "parameters": {"product_name": None}},
        {"query": "Change my shipping address.", "intent": "update_shipping_info", "parameters": {"new_address": None}}
    ]

def save_interaction_data(interaction: dict):
    print(f"Saving new interaction data: {json.dumps(interaction)}")

# intent_recognizer.py
class IntentRecognizer:
    def __init__(self, training_data: List[Dict]):
        print("Initializing IntentRecognizer with a simulated foundation model...")
        self.training_data = training_data
        self.model = self._load_simulated_model()

    def _load_simulated_model(self):
        print("Simulated foundation model loaded.")
        return {"model_type": "transformer", "version": "1.0", "capabilities": ["intent_recognition", "entity_extraction"]}

    def recognize_intent(self, query: str) -> Dict[str, Any]:
        print(f"Recognizing intent for query: '{query}'")
        query_lower = query.lower()

        if "order" in query_lower or "delivery" in query_lower or "where is" in query_lower or "late" in query_lower:
            if "id" not in query_lower and "number" not in query_lower and "track" not in query_lower:
                return {"intent": "check_order_status", "parameters": {"order_id": None}, "confidence": 0.7, "needs_clarification": True, "clarification_type": "order_id"}
            elif "track" in query_lower:
                return {"intent": "track_shipping", "parameters": {"order_id": "12345"}, "confidence": 0.9, "needs_clarification": False}
            else:
                 return {"intent": "check_order_status", "parameters": {"order_id": "12345"}, "confidence": 0.9, "needs_clarification": False}
        elif "return" in query_lower or "refund" in query_lower or "unhappy" in query_lower:
            if "product" not in query_lower and "item" not in query_lower:
                 return {"intent": "process_return", "parameters": {"product_name": None}, "confidence": 0.7, "needs_clarification": True, "clarification_type": "product_name"}
            else:
                return {"intent": "process_return", "parameters": {"product_name": "unknown"}, "confidence": 0.9, "needs_clarification": False}
        elif "address" in query_lower or "shipping info" in query_lower:
            return {"intent": "update_shipping_info", "parameters": {"new_address": None}, "confidence": 0.8, "needs_clarification": True, "clarification_type": "new_address"}
        elif "hello" in query_lower or "hi" in query_lower:
            return {"intent": "greeting", "parameters": {}, "confidence": 1.0, "needs_clarification": False}
        elif "thank you" in query_lower or "thanks" in query_lower:
            return {"intent": "gratitude", "parameters": {}, "confidence": 1.0, "needs_clarification": False}
        else:
            return {"intent": "general_query", "parameters": {}, "confidence": 0.5, "needs_clarification": True, "clarification_type": "general"}

    def clarify_ambiguity(self, intent_data: Dict[str, Any]) -> str:
        clarification_type = intent_data.get("clarification_type")
        print(f"Ambiguity detected, generating clarification for type: {clarification_type}")

        if clarification_type == "order_id":
            return "Could you please provide your order ID or tracking number so I can assist you better?"
        elif clarification_type == "product_name":
            return "Which product would you like to return or inquire about for a refund?"
        elif clarification_type == "new_address":
            return "Please provide your new shipping address."
        elif clarification_type == "general":
            return "Could you please rephrase your request or provide more details?"
        else:
            return "I need a bit more information to understand your request. Can you tell me more?"

    def fine_tune_model(self, new_interaction_data: List[Dict]):
        print("Simulating fine-tuning the model with new interaction data...")
        self.training_data.extend(new_interaction_data)
        print(f"Model now has {len(self.training_data)} training examples.")

# response_generator.py
class ResponseGenerator:
    def __init__(self):
        print("Initializing ResponseGenerator...")
        self.responses = {
            "check_order_status": "Let me check the status of your order. Please provide your order ID if you haven't already.",
            "track_shipping": "I can help you track your package. What is your order ID or tracking number?",
            "process_return": "I understand you'd like to process a return or refund. Please specify the product name and reason for return.",
            "update_shipping_info": "I can update your shipping information. What is your new address?",
            "greeting": "Hello! How can I assist you with your e-commerce needs today?",
            "gratitude": "You're welcome! Is there anything else I can help you with?",
            "general_query": "I'm not entirely sure how to help with that. Could you please provide more details?",
            "fallback": "I apologize, I didn't quite understand your request. Could you please rephrase it?"
        }

    def generate_response(self, intent: str, parameters: Dict[str, Any] = None) -> str:
        if parameters is None:
            parameters = {}

        response = self.responses.get(intent, self.responses["fallback"])

        if intent == "check_order_status" and parameters.get("order_id"):
            return f"Checking order status for ID {parameters['order_id']}. Please wait a moment."
        if intent == "track_shipping" and parameters.get("order_id"):
            return f"Tracking your package with ID {parameters['order_id']}. It seems to be in transit."
        if intent == "process_return" and parameters.get("product_name"):
            return f"Initiating a return process for {parameters['product_name']}. I will guide you through the steps."
        if intent == "update_shipping_info" and parameters.get("new_address"):
            return f"Your shipping address has been successfully updated to {parameters['new_address']}."

        return response

# tool_executor.py
class ToolExecutor:
    def __init__(self):
        print("Initializing ToolExecutor with available e-commerce tools...")
        self._available_tools: Dict[str, Callable] = {
            "check_order_status": self._check_order_status,
            "process_return": self._process_return,
            "track_shipping": self._track_shipping,
            "update_shipping_info": self._update_shipping_info
        }

    def _check_order_status(self, order_id: str = None) -> str:
        if order_id:
            print(f"Tool: Checking status for order ID: {order_id}")
            return f"Order {order_id} is currently 'Shipped' and expected by tomorrow."
        else:
            return "Tool: To check order status, I need an order ID."

    def _process_return(self, product_name: str = None) -> str:
        if product_name:
            print(f"Tool: Initiating return for product: {product_name}")
            return f"Return for {product_name} has been initiated. You will receive an email with instructions."
        else:
            return "Tool: To process a return, I need the product name."

    def _track_shipping(self, order_id: str = None) -> str:
        if order_id:
            print(f"Tool: Tracking shipping for order ID: {order_id}")
            return f"Package for order {order_id} is currently in transit in your city."
        else:
            return "Tool: To track shipping, I need an order ID or tracking number."

    def _update_shipping_info(self, new_address: str = None) -> str:
        if new_address:
            print(f"Tool: Updating shipping address to: {new_address}")
            return f"Your shipping address has been successfully updated to {new_address}."
        else:
            return "Tool: To update shipping information, I need the new address."

    def execute_tool(self, intent: str, parameters: Dict[str, Any]) -> str:
        tool_function = self._available_tools.get(intent)
        if tool_function:
            print(f"Executing tool for intent: {intent} with parameters: {parameters}")
            try:
                sig = inspect.signature(tool_function)
                relevant_params = {k: v for k, v in parameters.items() if k in sig.parameters}
                return tool_function(**relevant_params)
            except TypeError as e:
                return f"Error executing tool {intent}: {e}. Missing or invalid parameters."
        else:
            return f"No specific tool found for intent: {intent}. Generating a general response instead."

# main.py
def run_eca_simulation():
    print("Starting Intelligent Customer Support Agent (ECA) simulation...")

    training_data = load_ecommerce_data()
    intent_recognizer = IntentRecognizer(training_data)
    response_generator = ResponseGenerator()
    tool_executor = ToolExecutor()

    user_queries = [
        "Hi there!",
        "Where is my stuff?",
        "My order 12345 is late.",
        "I need to send something back.",
        "I want to return the broken toaster.",
        "Can I change my delivery address?",
        "My tracking number is TRK98765. Where is it?",
        "Thanks!",
        "I have a general question."
    ]

    for i, query in enumerate(user_queries):
        print(f"\n--- User {i+1} Query: '{query}' ---")
        intent_data = intent_recognizer.recognize_intent(query)
        print(f"Recognized Intent Data: {json.dumps(intent_data)}")

        response_to_user = ""
        tool_output = ""

        if intent_data.get("needs_clarification"):
            clarification_question = intent_recognizer.clarify_ambiguity(intent_data)
            print(f"ECA: {clarification_question}")
            # Simulate user providing clarification
            if intent_data.get("clarification_type") == "order_id":
                clarified_query = input("User (clarification): ") # Simulate user input
                if "my order ID is" in clarified_query.lower():
                    order_id = clarified_query.split("is ")[-1].strip().replace(".", "")
                    intent_data["parameters"]["order_id"] = order_id
                    intent_data["needs_clarification"] = False
                elif "tracking number is" in clarified_query.lower():
                    order_id = clarified_query.split("is ")[-1].strip().replace(".", "")
                    intent_data["parameters"]["order_id"] = order_id # Using order_id for simplicity
                    intent_data["intent"] = "track_shipping" # Update intent based on clarification
                    intent_data["needs_clarification"] = False
                else:
                    print("ECA: Still unclear, trying a general response.")

            elif intent_data.get("clarification_type") == "product_name":
                clarified_query = input("User (clarification): ")
                intent_data["parameters"]["product_name"] = clarified_query.replace(".", "") # Simple extraction
                intent_data["needs_clarification"] = False

            elif intent_data.get("clarification_type") == "new_address":
                clarified_query = input("User (clarification): ")
                intent_data["parameters"]["new_address"] = clarified_query.replace(".", "")
                intent_data["needs_clarification"] = False
            
            if not intent_data.get("needs_clarification"): # Re-evaluate after clarification
                print(f"Re-evaluated Intent Data after clarification: {json.dumps(intent_data)}")

        if not intent_data.get("needs_clarification") and intent_data.get("intent") in tool_executor._available_tools:
            tool_output = tool_executor.execute_tool(intent_data["intent"], intent_data["parameters"])
            print(f"Tool Output: {tool_output}")

        if tool_output:
            # If a tool was executed and returned a specific message, use it
            response_to_user = tool_output
        else:
            # Otherwise, generate a general response based on the intent
            response_to_user = response_generator.generate_response(intent_data["intent"], intent_data["parameters"])

        print(f"ECA: {response_to_user}")

        # Simulate personalized learning by saving the interaction
        save_interaction_data({
            "original_query": query,
            "final_intent": intent_data["intent"],
            "final_parameters": intent_data["parameters"],
            "eca_response": response_to_user
        })

        # Simulate fine-tuning after each interaction for personalized learning
        # In a real system, this would happen periodically or in batches.
        intent_recognizer.fine_tune_model([{
            "query": query,
            "intent": intent_data["intent"],
            "parameters": intent_data["parameters"]
        }])

    print("\nECA simulation finished.")

if __name__ == "__main__":
    run_eca_simulation()
