class SmartChatbot:
    def __init__(self):
        self.conversation_history = []
        self.intents = {
            "check_order_status": ["order status", "my order", "where is my stuff"],
            "password_reset": ["reset password", "forgot password", "change my pass"],
            "technical_support": ["technical issue", "problem with", "error", "not working"],
            "billing_inquiry": ["bill", "invoice", "charge", "payment"],
            "product_information": ["product info", "about", "details on"],
            "general_greeting": ["hello", "hi", "hey"],
            "thank_you": ["thank you", "thanks"],
            "goodbye": ["bye", "goodbye", "see you"],
            "unknown": []
        }
        self.current_clarification_context = None

    def _recognize_intent(self, query):
        detected_intents = []
        query_lower = query.lower()

        for intent, keywords in self.intents.items():
            if intent == "unknown":
                continue
            for keyword in keywords:
                if keyword in query_lower:
                    detected_intents.append(intent)
                    break
        
        if not detected_intents:
            return {"intent": "unknown", "confidence": 0.8}
        elif len(detected_intents) == 1:
            return {"intent": detected_intents[0], "confidence": 0.95}
        else:
            # Simulate ambiguity with multiple high-scoring intents
            return {"intent": "ambiguous", "possible_intents": detected_intents, "confidence": 0.6}

    def _generate_clarification(self, possible_intents):
        if "check_order_status" in possible_intents and "product_information" in possible_intents:
            return "Are you asking about the status of an existing order or information about a product?"
        elif "password_reset" in possible_intents and "technical_support" in possible_intents:
            return "Are you experiencing a technical issue or do you need help resetting your password?"
        elif "billing_inquiry" in possible_intents and "product_information" in possible_intents:
            return "Are you inquiring about a billing issue or seeking information about a product?"
        else:
            return f"It seems like you might be asking about a few things: {', '.join(possible_intents)}. Could you please clarify?"

    def _execute_action(self, intent, query):
        if intent == "check_order_status":
            return "Please provide your order number so I can check its status for you."
        elif intent == "password_reset":
            return "I can help you reset your password. I'll send a reset link to your registered email address."
        elif intent == "technical_support":
            return "I understand you're having a technical issue. Please describe it in more detail, and I'll connect you with a specialist if needed."
        elif intent == "billing_inquiry":
            return "For billing inquiries, I can look up your recent invoices or connect you to our billing department."
        elif intent == "product_information":
            return "What product are you interested in? I can provide details, specifications, and availability."
        elif intent == "general_greeting":
            return "Hello! How can I assist you today?"
        elif intent == "thank_you":
            return "You're welcome! Is there anything else I can help you with?"
        elif intent == "goodbye":
            return "Goodbye! Have a great day."
        else:
            return "I'm not sure how to handle that request at the moment. Can you please rephrase or provide more details?"

    def _generate_response(self, intent_result, query):
        if intent_result["intent"] == "ambiguous":
            self.current_clarification_context = intent_result["possible_intents"]
            return self._generate_clarification(intent_result["possible_intents"])
        elif intent_result["intent"] == "unknown":
            return "I'm not quite sure what you mean. Could you please rephrase your request?"
        else:
            self.current_clarification_context = None
            return self._execute_action(intent_result["intent"], query)

    def chat(self):
        print("Welcome to the Smart Customer Support Chatbot! Type 'exit' to end the conversation.")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                print("Chatbot: Goodbye!")
                break

            self.conversation_history.append({"user": user_input})

            intent_result = self._recognize_intent(user_input)
            response = self._generate_response(intent_result, user_input)
            
            self.conversation_history.append({"bot": response})
            print(f"Chatbot: {response}")

            # Simple follow-up for clarification context
            if self.current_clarification_context and "clarify" in response.lower():
                while True:
                    user_clarification = input("You (clarifying): ")
                    self.conversation_history.append({"user": user_clarification})
                    
                    # Re-evaluate intent with clarification
                    clarified_intent_result = self._recognize_intent(user_clarification)
                    if clarified_intent_result["intent"] != "ambiguous" and clarified_intent_result["intent"] != "unknown":
                        response = self._execute_action(clarified_intent_result["intent"], user_clarification)
                        self.conversation_history.append({"bot": response})
                        print(f"Chatbot: {response}")
                        self.current_clarification_context = None
                        break
                    else:
                        print("Chatbot: I still need more clarification. Can you be more specific?")

if __name__ == "__main__":
    chatbot = SmartChatbot()
    chatbot.chat()
