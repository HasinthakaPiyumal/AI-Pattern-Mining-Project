class CustomerSupportAssistant:
    def __init__(self):
        self.intents_data = {
            "billing_query": {
                "keywords": ["bill", "invoice", "charge", "payment", "amount"],
                "response": "You can check your latest bill and payment history on our website: https://www.telecomco.com/billing. Would you like me to help you navigate there?"
            },
            "technical_support": {
                "keywords": ["internet", "no service", "router", "connection", "wifi", "speed", "technical"],
                "clarification": "Could you please tell me more about your technical issue? Are you experiencing problems with internet, TV, or phone service?",
                "response": "For technical support, please visit our troubleshooting guide at https://www.telecomco.com/support/technical or consider restarting your modem and router."
            },
            "plan_change": {
                "keywords": ["upgrade", "downgrade", "change plan", "new plan", "subscription"],
                "clarification": "Are you looking to upgrade your current plan, downgrade, or explore new plan options?",
                "response": "You can view and manage your plan options at https://www.telecomco.com/plans. Our sales team can also assist you with this."
            },
            "contact_support": {
                "keywords": ["speak to someone", "agent", "human", "talk to representative", "customer service"],
                "clarification": "I can connect you to a customer service representative. What is the nature of your query so I can route you to the correct department (e.g., billing, technical support, sales)?",
                "response": "Please hold while I connect you to the next available representative based on your query. Estimated wait time: 5 minutes."
            }
        }

    def _identify_intents(self, query):
        found_intents = set()
        query_lower = query.lower()
        for intent_name, data in self.intents_data.items():
            for keyword in data["keywords"]:
                if keyword in query_lower:
                    found_intents.add(intent_name)
                    break
        return list(found_intents)

    def _get_response(self, identified_intents, user_query):
        if not identified_intents:
            return "I'm sorry, I couldn't understand your request. Could you please rephrase it or provide more details?"

        if len(identified_intents) == 1:
            intent = identified_intents[0]
            if "response" in self.intents_data[intent]:
                return self.intents_data[intent]["response"]
            elif "clarification" in self.intents_data[intent]:
                return self.intents_data[intent]["clarification"]
        else:
            # Multiple potential intents, require clarification
            clarification_questions = []
            for intent in identified_intents:
                if "clarification" in self.intents_data[intent]:
                    clarification_questions.append(self.intents_data[intent]["clarification"])

            if clarification_questions:
                return "I detect a few possible intentions. " + " ".join(clarification_questions)
            else:
                return "Your query is a bit ambiguous. Could you please provide more specific details?"
        return "Something went wrong. Please try again."

    def run(self):
        print("Hello! I am your AI Customer Support Assistant. How can I help you today?")
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("Assistant: Goodbye! Have a great day.")
                break

            identified_intents = self._identify_intents(user_input)
            response = self._get_response(identified_intents, user_input)
            print(f"Assistant: {response}")

if __name__ == "__main__":
    assistant = CustomerSupportAssistant()
    assistant.run()