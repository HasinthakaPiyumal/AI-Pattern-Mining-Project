import re

class CustomerSupportAssistant:
    def __init__(self):
        # Simulating fine-tuned foundation model's understanding through keywords
        self.intents = {
            "order_status": ["track my order", "where is my package", "order status", "my delivery", "check order"],
            "initiate_return": ["return an item", "send back", "return process", "i want to return"],
            "damaged_item": ["item arrived broken", "damaged product", "defective item", "faulty"],
            "product_info": ["product details", "about this item", "specifications", "tell me about"],
            "cancel_order": ["cancel my order", "stop my order"],
        }
        self.clarification_questions = {
            "general_problem": "Could you please specify which order you are referring to? Are you experiencing issues with shipping, product quality, or something else?",
            "vague_query": "I\'m having trouble understanding. Could you please provide more details or rephrase your request?",
            "order_related": "To help you with your order, could you please provide your order number?",
        }
        self.responses = {
            "order_status": "Please provide your order number so I can check its status for you.",
            "initiate_return": "To initiate a return, please visit our \'Returns\' page on the website or provide your order number for assistance.",
            "damaged_item": "I\'m sorry to hear that. To help you, please provide your order number and a brief description of the damage.",
            "product_info": "Which product are you interested in? Please provide the product name or ID.",
            "cancel_order": "To cancel an order, please provide your order number. Note that cancellations are only possible for orders that haven\'t been shipped yet.",
            "no_intent_found": "I\'m not sure I understand your request. Could you please rephrase it?",
            "clarified_order_status": "Thank you. For order [ORDER_NUMBER], the current status is: [STATUS].",
            "clarified_damaged_item": "Thank you for the information. We\'ve logged your report for order [ORDER_NUMBER] regarding a damaged item. Our team will review it and contact you shortly.",
            "clarified_cancel_order": "Order [ORDER_NUMBER] has been marked for cancellation. You will receive an email confirmation shortly.",
        }
        self.user_session_context = {} # Stores context for each user's current conversation

    def _recognize_intent(self, query):
        query_lower = query.lower()
        # More sophisticated keyword matching could involve regex, embeddings, or actual FM calls
        for intent, keywords in self.intents.items():
            if any(keyword in query_lower for keyword in keywords):
                return intent
        return "unknown"

    def _extract_order_number(self, text):
        match = re.search(r'\b\d{6,}\b', text) # Matches 6 or more digits, bounded by word boundaries
        if match:
            return match.group(0)
        return None

    def process_query(self, user_id, query):
        if user_id not in self.user_session_context:
            self.user_session_context[user_id] = {"awaiting_input_for": None, "previous_intent": None}

        current_context = self.user_session_context[user_id]
        order_number = self._extract_order_number(query)

        # Handle multi-turn conversation based on previous context
        if current_context["awaiting_input_for"] == "order_number":
            if order_number:
                previous_intent = current_context["previous_intent"]
                current_context["awaiting_input_for"] = None # Clear context
                current_context["previous_intent"] = None # Clear context

                if previous_intent == "order_status":
                    # Simulate dynamic status based on order_number
                    status = "shipped and expected by tomorrow" if int(order_number) % 2 == 0 else "processing, estimated delivery in 3-5 business days"
                    return self.responses["clarified_order_status"].replace("[ORDER_NUMBER]", order_number).replace("[STATUS]", status)
                elif previous_intent == "damaged_item":
                    return self.responses["clarified_damaged_item"].replace("[ORDER_NUMBER]", order_number)
                elif previous_intent == "cancel_order":
                    return self.responses["clarified_cancel_order"].replace("[ORDER_NUMBER]", order_number)
            else:
                return "I still need your order number. Can you please provide it?"

        # Recognize intent for the current query
        intent = self._recognize_intent(query)

        if intent == "unknown":
            if any(word in query.lower() for word in ["problem", "issue", "help", "trouble", "confused"]):
                return self.clarification_questions["general_problem"]
            else:
                return self.clarification_questions["vague_query"]
        else:
            if intent in ["order_status", "damaged_item", "cancel_order"]:
                if order_number:
                    # If order number is already in the query, process directly
                    if intent == "order_status":
                        status = "shipped and expected by tomorrow" if int(order_number) % 2 == 0 else "processing, estimated delivery in 3-5 business days"
                        return self.responses["clarified_order_status"].replace("[ORDER_NUMBER]", order_number).replace("[STATUS]", status)
                    elif intent == "damaged_item":
                        return self.responses["clarified_damaged_item"].replace("[ORDER_NUMBER]", order_number)
                    elif intent == "cancel_order":
                        return self.responses["clarified_cancel_order"].replace("[ORDER_NUMBER]", order_number)
                else:
                    # If no order number, ask for it and set context
                    current_context["awaiting_input_for"] = "order_number"
                    current_context["previous_intent"] = intent # Store intent for next turn
                    if intent == "order_status":
                        return self.responses["order_status"]
                    elif intent == "damaged_item":
                        return self.responses["damaged_item"]
                    elif intent == "cancel_order":
                        return self.responses["cancel_order"]
            elif intent == "initiate_return":
                return self.responses["initiate_return"]
            elif intent == "product_info":
                return self.responses["product_info"]
        return self.responses["no_intent_found"] # Fallback

# Main execution block for demonstration
def main_demonstration():
    assistant = CustomerSupportAssistant()

    print("--- Smart Customer Support Assistant Demo ---")
    print("Type 'exit' to end the conversation.\n")

    user_id = "customer_A" # Simulating a single user session for simplicity in main demo

    while True:
        user_query = input(f"You ({user_id}): ")
        if user_query.lower() == "exit":
            print("Assistant: Goodbye!")
            break
        
        assistant_response = assistant.process_query(user_id, user_query)
        print(f"Assistant: {assistant_response}")

if __name__ == "__main__":
    main_demonstration()