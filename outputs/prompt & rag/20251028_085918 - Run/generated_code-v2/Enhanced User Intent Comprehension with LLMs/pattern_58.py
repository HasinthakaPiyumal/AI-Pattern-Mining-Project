class ECommerceChatbot:
    def __init__(self):
        self.intents = {
            "order_status": {
                "keywords": ["order", "status", "where is my", "track"],
                "response": "To check your order status, please provide your order number.",
                "follow_up": "What is your order number?"
            },
            "return_policy": {
                "keywords": ["return", "policy", "how to return", "refund"],
                "response": "Our return policy allows returns within 30 days of purchase for a full refund. Items must be unused and in original packaging.",
                "follow_up": None
            },
            "shipping_info": {
                "keywords": ["shipping", "delivery", "shipment", "cost"],
                "response": "Shipping costs and delivery times vary based on your location and selected shipping method.",
                "follow_up": "Where are you located, and what shipping speed are you interested in?"
            },
            "product_info": {
                "keywords": ["product", "details", "about", "specifications"],
                "response": "Could you please specify which product you are interested in?",
                "follow_up": "Which product are you asking about?"
            },
            "greeting": {
                "keywords": ["hello", "hi", "hey"],
                "response": "Hello! How can I assist you today?",
                "follow_up": None
            },
            "thanks": {
                "keywords": ["thank you", "thanks"],
                "response": "You're welcome! Is there anything else I can help with?",
                "follow_up": None
            }
        }
        self.user_session = {}

    def _identify_intent(self, user_query):
        user_query_lower = user_query.lower()
        for intent_name, intent_data in self.intents.items():
            for keyword in intent_data["keywords"]:
                if keyword in user_query_lower:
                    return intent_name
        return "unknown"

    def _resolve_ambiguity(self, intent):
        if intent in self.intents and self.intents[intent]["follow_up"]:
            return self.intents[intent]["follow_up"]
        return None

    def _generate_response(self, intent, user_query):
        if intent == "unknown":
            return "I'm sorry, I don't understand your request. Could you please rephrase it or be more specific?"

        response = self.intents[intent]["response"]

        # Simple personalization based on remembered preferences (placeholder)
        if "preferred_shipping" in self.user_session and intent == "shipping_info":
            response += f" I see your preferred shipping is {self.user_session['preferred_shipping']}."

        return response

    def _remember_preference(self, user_query):
        # A very basic example of personalized learning
        if "fast shipping" in user_query.lower():
            self.user_session["preferred_shipping"] = "express"
        elif "standard shipping" in user_query.lower():
            self.user_session["preferred_shipping"] = "standard"

    def chat(self):
        print("Welcome to the E-commerce Customer Support Chatbot! Type 'exit' to end the conversation.")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                print("Chatbot: Goodbye!")
                break

            self._remember_preference(user_input) # Attempt to learn preferences

            identified_intent = self._identify_intent(user_input)
            response = self._generate_response(identified_intent, user_input)

            print(f"Chatbot: {response}")

            # Handle ambiguity
            if identified_intent != "unknown":
                follow_up_question = self._resolve_ambiguity(identified_intent)
                if follow_up_question:
                    print(f"Chatbot: {follow_up_question}")
                    # In a real system, we'd wait for a specific answer to this follow-up.
                    # For this simulation, we just ask and continue.

if __name__ == "__main__":
    chatbot = ECommerceChatbot()
    chatbot.chat()
