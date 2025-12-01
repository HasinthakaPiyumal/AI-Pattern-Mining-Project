import random

class MockIntentRecognizer:
    def __init__(self):
        self.intents = {
            "order status": ["where is my order", "track my package", "what is the status of my shipment"],
            "return request": ["i want to return an item", "how do i send something back", "return policy"],
            "product complaint": ["product is broken", "item arrived damaged", "faulty product"],
            "shipping inquiry": ["shipping cost", "delivery options", "how long for delivery"],
            "account update": ["change my address", "update my payment info", "reset password"],
            "general question": ["hello", "hi", "can you help me"],
        }
        self.threshold = 0.7

    def predict_intent(self, query):
        query = query.lower()
        possible_intents = {}
        for intent, phrases in self.intents.items():
            for phrase in phrases:
                if phrase in query:
                    possible_intents[intent] = possible_intents.get(intent, 0) + 1
        
        if not possible_intents:
            return "unknown", 0.0

        most_common_intent = max(possible_intents, key=possible_intents.get)
        confidence = possible_intents[most_common_intent] / sum(possible_intents.values())

        if confidence >= self.threshold:
            return most_common_intent, confidence
        else:
            return "unclear", confidence


class AmbiguityResolver:
    def __init__(self):
        self.clarifying_questions = {
            "unclear": "Could you please elaborate on your request?",
            "order status": "Can you please provide your order number?",
            "return request": "What item would you like to return and what is the reason?",
            "product complaint": "Which product are you referring to and what is the issue?"
        }

    def resolve_ambiguity(self, intent, original_query=None):
        if intent in self.clarifying_questions:
            return self.clarifying_questions[intent]
        return "I'm sorry, I couldn't fully understand. Can you rephrase?"


class ToolActionMapper:
    def __init__(self):
        self.available_tools = {
            "order status": self._get_order_status,
            "return request": self._initiate_return,
            "product complaint": self._log_product_complaint,
            "shipping inquiry": self._get_shipping_info,
            "account update": self._update_account,
            "general question": self._handle_general_question,
            "escalate": self._escalate_to_human
        }

    def _get_order_status(self, order_id="unknown"):
        if order_id != "unknown":
            return f"Fetching status for order {order_id}. Please wait."
        return "Please provide your order ID to check the status."

    def _initiate_return(self, item_id="unknown", reason="unspecified"):
        if item_id != "unknown" and reason != "unspecified":
            return f"Initiating return for item {item_id} due to {reason}."
        return "To initiate a return, please tell me the item and the reason."

    def _log_product_complaint(self, product_name="unknown", issue="unspecified"):
        if product_name != "unknown" and issue != "unspecified":
            return f"Logging complaint for {product_name}: {issue}. Someone will get back to you."
        return "Please tell me the product name and describe the issue."
    
    def _get_shipping_info(self):
        return "Shipping costs and delivery times vary by location and product. Please provide your location and the product you are interested in for more details."

    def _update_account(self):
        return "I can help you update your account details. What information would you like to change?"

    def _handle_general_question(self):
        return "How can I assist you further?"

    def _escalate_to_human(self):
        return "I am connecting you to a human agent. Please hold."

    def execute_tool(self, intent, entities=None):
        if intent in self.available_tools:
            if entities:
                return self.available_tools[intent](**entities)
            return self.available_tools[intent]()
        return "I'm sorry, I don't have a tool to handle that specific request yet."


class ResponseGenerator:
    def generate_response(self, tool_result, intent="general", original_query=""):
        if tool_result:
            return tool_result
        elif intent == "unknown":
            return "I'm not sure how to help with that. Could you please rephrase or provide more details?"
        elif intent == "unclear":
            return "I'm still trying to understand. Can you provide more specific information?"
        return "Is there anything else I can help you with?"


class DialogueManager:
    def __init__(self):
        self.irm = MockIntentRecognizer()
        self.arm = AmbiguityResolver()
        self.tam = ToolActionMapper()
        self.rgm = ResponseGenerator()
        self.conversation_history = []
        self.current_state = "INITIAL"
        self.pending_intent = None
        self.extracted_entities = {}

    def _extract_entities(self, query, intent):
        entities = {}
        query_lower = query.lower()
        if intent == "order status":
            # Simple regex to find a potential order ID (e.g., 3-5 digits)
            import re
            match = re.search(r'\b(?:order number|id|#)?\s*(\d{3,5})\b', query_lower)
            if match:
                entities["order_id"] = match.group(1)
        elif intent == "return request":
            if "item" in query_lower:
                # Placeholder for more sophisticated item extraction
                entities["item_id"] = "user_specified_item"
            if "damaged" in query_lower or "broken" in query_lower:
                entities["reason"] = "damaged"
            elif "wrong size" in query_lower:
                entities["reason"] = "wrong size"
            else:
                entities["reason"] = "other"
        elif intent == "product complaint":
            # Placeholder for product name extraction
            if "product" in query_lower:
                entities["product_name"] = "user_specified_product"
            if "not working" in query_lower or "faulty" in query_lower:
                entities["issue"] = "not working"
            elif "missing part" in query_lower:
                entities["issue"] = "missing part"
            else:
                entities["issue"] = "general issue"

        return entities

    def converse(self, user_input):
        self.conversation_history.append(("user", user_input))
        response = ""
        tool_result = None

        if self.current_state == "AWAITING_CLARIFICATION":
            # Try to re-evaluate intent with clarified input
            predicted_intent, confidence = self.irm.predict_intent(user_input)
            if predicted_intent == self.pending_intent and confidence > 0.8:
                self.current_state = "INITIAL"
                current_intent = self.pending_intent
                self.pending_intent = None
                self.extracted_entities.update(self._extract_entities(user_input, current_intent))
            else:
                # If clarification didn't help, re-ask or escalate
                response = self.arm.resolve_ambiguity(self.pending_intent, user_input)
                self.conversation_history.append(("bot", response))
                return response
        else:
            predicted_intent, confidence = self.irm.predict_intent(user_input)
            current_intent = predicted_intent
            self.extracted_entities = self._extract_entities(user_input, current_intent)

        if current_intent == "unclear" or confidence < 0.7:
            self.pending_intent = current_intent # Store for follow-up
            self.current_state = "AWAITING_CLARIFICATION"
            response = self.arm.resolve_ambiguity(current_intent, user_input)
        elif current_intent == "unknown":
            response = self.rgm.generate_response(None, current_intent)
        else:
            # Clear intent, proceed to tool mapping
            tool_result = self.tam.execute_tool(current_intent, self.extracted_entities)
            response = self.rgm.generate_response(tool_result, current_intent)
            self.current_state = "INITIAL"
            self.pending_intent = None
            self.extracted_entities = {}

        self.conversation_history.append(("bot", response))
        return response


if __name__ == "__main__":
    assistant = DialogueManager()
    print("Smart Customer Support Assistant: Hello! How can I assist you today?")

    while True:
        user_query = input("You: ")
        if user_query.lower() in ["exit", "quit", "bye"]:
            print("Smart Customer Support Assistant: Goodbye! Have a great day!")
            break
        
        response = assistant.converse(user_query)
        print(f"Smart Customer Support Assistant: {response}")