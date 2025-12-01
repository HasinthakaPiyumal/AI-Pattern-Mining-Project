
# intent_classifier.py
def classify_intent(query: str) -> dict:
    query_lower = query.lower()
    if "order" in query_lower and ("status" in query_lower or "track" in query_lower):
        return {"intent": "order_status", "confidence": 0.95}
    elif "return" in query_lower or "exchange" in query_lower:
        return {"intent": "returns", "confidence": 0.90}
    elif "product" in query_lower or "item" in query_lower or "specifications" in query_lower:
        return {"intent": "product_info", "confidence": 0.88}
    elif "bill" in query_lower or "charge" in query_lower or "invoice" in query_lower:
        return {"intent": "billing", "confidence": 0.85}
    elif "issue" in query_lower or "technical" in query_lower or "broken" in query_lower:
        return {"intent": "technical_support", "confidence": 0.82}
    else:
        return {"intent": "other", "confidence": 0.60}

# dialogue_manager.py
class DialogueManager:
    def __init__(self):
        self.context = {}

    def manage_dialogue(self, user_query: str, predicted_intent: dict, current_context: dict) -> tuple:
        self.context = current_context

        intent = predicted_intent["intent"]
        confidence = predicted_intent["confidence"]

        if confidence < 0.75 and intent == "other":
            return {"state": "clarifying", "last_query": user_query}, "I'm not entirely sure what you mean. Could you please rephrase or provide more details?"
        
        if intent == "order_status" and "order_id" not in self.context:
            if "order number" not in user_query.lower() and not any(char.isdigit() for char in user_query):
                return {"state": "clarifying", "intent": intent, "last_query": user_query}, "To check your order status, please provide your order number."
            else:
                # Mock entity extraction for order_id
                import re
                match = re.search(r'\b\d{6,}\b', user_query) # Basic digit sequence as order ID
                if match:
                    self.context["order_id"] = match.group(0)
                    self.context["state"] = "confirmed"
                    self.context["confirmed_intent"] = intent
                    return self.context, None
                else:
                    return {"state": "clarifying", "intent": intent, "last_query": user_query}, "Could you please provide a valid order number?"
        
        self.context["state"] = "confirmed"
        self.context["confirmed_intent"] = intent
        return self.context, None

# response_handler.py
class ResponseHandler:
    def __init__(self):
        self.knowledge_base = {
            "order_status": "You can check your order status by logging into your account or using the 'Track Order' link on our website with your order number.",
            "returns": "Please visit our Returns & Refunds page for detailed instructions on how to return an item. You usually have 30 days from purchase.",
            "product_info": "Could you please specify which product you are interested in? Our website has detailed descriptions and specifications for all items.",
            "billing": "For billing inquiries, you can view your past invoices in your account settings or contact our billing department directly.",
            "technical_support": "For technical issues, please describe your problem in detail or visit our troubleshooting guide. If needed, we can connect you with a technical agent.",
            "other": "I'm sorry, I cannot assist with that specific request at the moment. Would you like to speak to a human agent?"
        }
        self.user_profiles = {
            "user_123": {"preferred_language": "English", "past_purchases": ["electronics", "books"]},
            "user_456": {"preferred_language": "Spanish", "past_purchases": ["clothing"]}
        }

    def generate_response(self, confirmed_intent: str, entities: dict = None, user_id: str = "default_user") -> str:
        base_response = self.knowledge_base.get(confirmed_intent, self.knowledge_base["other"])
        
        # Simple personalization (mock)
        user_profile = self.user_profiles.get(user_id, {})
        if user_profile and confirmed_intent == "product_info" and "past_purchases" in user_profile:
            if "electronics" in user_profile["past_purchases"]:
                base_response += " Perhaps you're looking for our latest gadgets?"

        if confirmed_intent == "order_status" and entities and "order_id" in entities:
            base_response = f"I've noted your order number: {entities['order_id']}. Please use this to track your order on our website. {self.knowledge_base['order_status']}"

        return base_response

# main_assistant.py
def main():
    print("Welcome to the Smart Customer Support Assistant! Type 'exit' to quit.")
    
    dialogue_manager = DialogueManager()
    response_handler = ResponseHandler()
    
    current_context = {}
    user_id = "user_123" # Mock user ID

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Thank you for contacting support. Goodbye!")
            break

        # 1. Intent Classification
        predicted_intent = classify_intent(user_input)
        
        # 2. Dialogue Management
        updated_context, clarifying_question = dialogue_manager.manage_dialogue(user_input, predicted_intent, current_context)
        current_context = updated_context

        if clarifying_question:
            print(f"Assistant: {clarifying_question}")
            continue
        
        confirmed_intent = current_context.get("confirmed_intent", predicted_intent["intent"])
        entities = {"order_id": current_context.get("order_id")}

        # 3. Response Handling
        assistant_response = response_handler.generate_response(confirmed_intent, entities, user_id)
        print(f"Assistant: {assistant_response}")

        # Reset context if the conversation reaches a resolution for a specific intent
        if confirmed_intent != "other" and "state" in current_context and current_context["state"] == "confirmed":
             # For simplicity, reset after confirmed intent and response, 
             # in a real system, more sophisticated context management would be used.
            current_context = {}

if __name__ == "__main__":
    main()
