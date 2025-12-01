class AmbiguousDemonstrationsChatbot:
    def __init__(self):
        self.ambiguous_demonstrations = {
            "where is my order": {
                "interpretations": ["Shipping Status", "Estimated Delivery Date", "Tracking Link"],
                "responses": {
                    "Shipping Status": "Your order #12345 is currently being processed.",
                    "Estimated Delivery Date": "Your order #12345 is estimated to be delivered by October 26, 2023.",
                    "Tracking Link": "Here is your tracking link: https://example.com/track/12345"
                }
            },
            "i have a problem": {
                "interpretations": ["Technical Issue", "Billing Issue", "Product Complaint"],
                "responses": {
                    "Technical Issue": "Please describe your technical issue in more detail.",
                    "Billing Issue": "Could you please provide your order number or account details for billing assistance?",
                    "Product Complaint": "We're sorry to hear that. What product are you experiencing issues with?"
                }
            }
        }
        self.clear_responses = {
            "what is your return policy": "Our return policy allows returns within 30 days of purchase with a valid receipt.",
            "hello": "Hello! How can I assist you today?",
            "thank you": "You're welcome! Let me know if you need anything else."
        }

    def _detect_ambiguity(self, query):
        for ambiguous_query in self.ambiguous_demonstrations.keys():
            if ambiguous_query in query.lower():
                return ambiguous_query, self.ambiguous_demonstrations[ambiguous_query]
        return None, None

    def _generate_clarification_prompt(self, interpretations):
        prompt = "Your query can be interpreted in a few ways. Which of the following best describes what you're looking for?\n"
        for i, interpretation in enumerate(interpretations):
            prompt += f"{i + 1}. {interpretation}\n"
        prompt += "Please enter the number corresponding to your choice: "
        return prompt

    def _get_llm_response(self, query, clarified_intent=None):
        if clarified_intent:
            for ambiguous_query_key, data in self.ambiguous_demonstrations.items():
                if ambiguous_query_key in query.lower() and clarified_intent in data["responses"]:
                    return data["responses"][clarified_intent]
        
        for clear_query_key, response in self.clear_responses.items():
            if clear_query_key in query.lower():
                return response

        return "I'm sorry, I don't have enough information to answer that. Could you please rephrase or provide more details?"

    def ask_chatbot(self, query):
        ambiguous_query_key, ambiguous_data = self._detect_ambiguity(query)

        if ambiguous_query_key:
            clarification_prompt = self._generate_clarification_prompt(ambiguous_data["interpretations"])
            print(clarification_prompt, end='')
            try:
                choice = int(input())
                if 1 <= choice <= len(ambiguous_data["interpretations"]):
                    clarified_intent = ambiguous_data["interpretations"][choice - 1]
                    return self._get_llm_response(query, clarified_intent)
                else:
                    return "Invalid choice. Please try again with a valid number."
            except ValueError:
                return "Invalid input. Please enter a number."
        else:
            return self._get_llm_response(query)


def main():
    chatbot = AmbiguousDemonstrationsChatbot()
    print("Welcome to the E-commerce Support Chatbot! Type 'exit' to quit.")

    while True:
        user_query = input("\nYou: ")
        if user_query.lower() == 'exit':
            break
        
        response = chatbot.ask_chatbot(user_query)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    main()