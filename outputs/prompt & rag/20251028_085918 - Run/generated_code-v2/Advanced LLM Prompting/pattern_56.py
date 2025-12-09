class EmotionPromptingChatbot:
    def __init__(self, emotional_prompt_prefix="This is a critical issue for the customer and requires utmost care and precision in the response. Please provide a detailed and empathetic solution."):
        self.emotional_prompt_prefix = emotional_prompt_prefix

    def _augment_query_with_emotion(self, user_query):
        return f"{self.emotional_prompt_prefix} {user_query}"

    def _simulate_llm_response(self, prompted_query):
        # In a real application, this would call an actual LLM API
        # For demonstration, we'll return a placeholder response based on the prompt
        if "critical issue" in prompted_query.lower() or "utmost care" in prompted_query.lower():
            return "Thank you for bringing this to our attention. We understand this is important and are dedicated to finding the best solution for you. Could you please provide more details about your issue?"
        else:
            return "I understand. Please tell me more about what you need assistance with."

    def get_chatbot_response(self, user_query):
        augmented_query = self._augment_query_with_emotion(user_query)
        print(f"[DEBUG] Augmented Query sent to LLM: {augmented_query}") # For internal debugging
        llm_response = self._simulate_llm_response(augmented_query)
        return llm_response

def main():
    print("Welcome to the Emotion-Prompting Customer Support Chatbot (Type 'exit' to quit)")
    chatbot = EmotionPromptingChatbot()

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break

        response = chatbot.get_chatbot_response(user_input)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    main()