class SmartCustomerSupportAgent:
    def __init__(self):
        self.intent_mappings = {
            "product_info": {"keywords": ["product", "details", "information", "specs"], "response": "Please specify the product you are interested in. I can provide details like features, pricing, and availability.", "high_confidence_response": "To get product details, please provide the product name or ID."}, 
            "order_status": {"keywords": ["order", "status", "where is my", "delivery", "shipment"], "response": "To check your order status, please provide your order number.", "high_confidence_response": "What is your order number?"},
            "returns_policy": {"keywords": ["return", "policy", "refund", "exchange"], "response": "Our return policy allows returns within 30 days of purchase with a valid receipt. Do you have a specific return query?", "high_confidence_response": "Our return policy is 30 days. Do you need details on how to initiate a return?"},
            "technical_support": {"keywords": ["technical", "issue", "broken", "error", "faulty"], "response": "It sounds like you need technical assistance. Please describe your issue in more detail, and I can route you to the correct specialist.", "high_confidence_response": "Please describe your technical issue so I can connect you to the right support."},
            "contact_human": {"keywords": ["speak to human", "agent", "representative", "call me"], "response": "I understand you'd like to speak with a human agent. Please provide your contact information and a brief reason for your call, and I will connect you to the next available representative.", "high_confidence_response": "Connecting you to a human agent. Please hold."}
        }

        self.user_history = {}

    def _identify_intent(self, query, user_id):
        identified_intent = "unknown"
        max_confidence = 0
        query_lower = query.lower()

        for intent, data in self.intent_mappings.items():
            matched_keywords = [keyword for keyword in data["keywords"] if keyword in query_lower]
            current_confidence = len(matched_keywords) / len(data["keywords"])

            # Basic personalization: if a user frequently uses a keyword for an intent, boost confidence
            if user_id in self.user_history and intent in self.user_history[user_id]:
                current_confidence += self.user_history[user_id][intent] * 0.1  # Small boost

            if current_confidence > max_confidence:
                max_confidence = current_confidence
                identified_intent = intent

        return identified_intent, max_confidence

    def _dialogue_manager(self, query, identified_intent, confidence):
        if confidence >= 0.8: # High confidence
            return self.intent_mappings[identified_intent].get("high_confidence_response", self.intent_mappings[identified_intent]["response"])
        elif confidence >= 0.4: # Medium confidence, ask for clarification
            clarification_needed = input(f"I think you are asking about {identified_intent.replace('_', ' ')}. Could you please provide more details? (Yes/No): ").lower()
            if clarification_needed == "yes":
                return self.intent_mappings[identified_intent]["response"]
            else:
                return "Please rephrase your query, or tell me more about what you need assistance with."
        else: # Low confidence
            return "I'm having trouble understanding your request. Could you please rephrase it or be more specific?"

    def _update_user_history(self, user_id, intent):
        if user_id not in self.user_history:
            self.user_history[user_id] = {}
        self.user_history[user_id][intent] = self.user_history[user_id].get(intent, 0) + 1

    def chat(self, user_query, user_id="default_user"):
        intent, confidence = self._identify_intent(user_query, user_id)
        
        if intent != "unknown":
            self._update_user_history(user_id, intent)

        response = self._dialogue_manager(user_query, intent, confidence)
        return response

if __name__ == "__main__":
    agent = SmartCustomerSupportAgent()
    print("Welcome to Smart Customer Support! Type 'exit' to end the conversation.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Thank you for contacting support. Goodbye!")
            break
        
        # In a real application, user_id would come from session management
        agent_response = agent.chat(user_input, user_id="user_123") 
        print(f"Agent: {agent_response}")