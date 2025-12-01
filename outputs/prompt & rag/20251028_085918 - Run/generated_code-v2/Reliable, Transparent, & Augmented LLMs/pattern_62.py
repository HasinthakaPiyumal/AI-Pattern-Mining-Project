
import random

class ConfidentChatbot:
    """
    A chatbot that provides answers along with a confidence score.
    It simulates interaction with an LLM that is prompted to give a verbalized score.
    """

    def __init__(self, llm_model_name="Simulated LLM"): # Placeholder for actual LLM
        self.llm_model_name = llm_model_name
        print(f"ConfidentChatbot initialized using {self.llm_model_name}.")

    def _get_llm_response_with_confidence(self, user_query: str) -> dict:
        """
        Simulates an LLM call that includes a request for a confidence score.
        In a real application, this would involve calling an actual LLM API
        and parsing its structured response.
        """
        # This is where the 'Verbalized Score' pattern is applied in the prompt.
        # The LLM would be instructed to respond in a specific format.
        prompt = f"""
        You are a helpful customer support assistant. Provide a concise answer to the following question. 
        After your answer, state your confidence level for the answer on a scale from 1 to 10.
        Format your response as: "Answer: [Your Answer] Confidence: [1-10]"

        Question: {user_query}
        """

        print(f"\n[DEBUG] Sending prompt to LLM: '{prompt.strip()}'")

        # --- Simulation of LLM Response ---
        # In a real scenario, an LLM would process the 'prompt' and return a string
        # that needs to be parsed. Here, we simulate that process.
        simulated_answers = {
            "shipping cost": "Standard shipping within the country costs $5.99. International shipping rates vary by destination.",
            "return policy": "You can return most items within 30 days of purchase, provided they are unused and in original packaging. Some exclusions apply, please check our full return policy online.",
            "payment methods": "We accept major credit cards (Visa, Mastercard, Amex), PayPal, and Apple Pay.",
            "reset password": "To reset your password, go to the login page and click on 'Forgot Password'. Follow the instructions sent to your registered email address.",
            "contact support": "You can reach our support team via email at support@example.com or call us at 1-800-555-0123 during business hours."
        }

        # Simple keyword matching for simulation
        answer_key = None
        for key in simulated_answers.keys():
            if key in user_query.lower():
                answer_key = key
                break

        if answer_key:
            answer = simulated_answers[answer_key]
            # Simulate a higher confidence for matched answers
            confidence = random.randint(7, 10)
        else:
            answer = "I'm sorry, I couldn't find a specific answer for that. Please try rephrasing your question or contact a human agent."
            # Simulate lower confidence for unknown questions
            confidence = random.randint(1, 5)
        # --- End Simulation ---

        # Simulating the parsing of LLM output
        # A real LLM might return a single string like: "Answer: ... Confidence: ..."
        # We then need to extract these parts.
        llm_output = f"Answer: {answer} Confidence: {confidence}"

        # Parse the simulated LLM output
        try:
            parts = llm_output.split("Confidence:")
            extracted_answer = parts[0].replace("Answer:", "").strip()
            extracted_confidence = int(parts[1].strip())
        except (ValueError, IndexError):
            # Fallback for unexpected LLM output format
            extracted_answer = llm_output
            extracted_confidence = 0 # Indicate parsing failure or low confidence

        return {"answer": extracted_answer, "confidence": extracted_confidence}

    def ask(self, query: str) -> str:
        """
        Asks the chatbot a question and returns the answer with a confidence score.
        """
        print(f"\nUser: {query}")
        response = self._get_llm_response_with_confidence(query)

        final_response = (
            f"Chatbot: {response['answer']}\n" 
            f"On a scale of 1 to 10, I am {response['confidence']} confident in this answer."
        )
        return final_response

# --- Main application loop for demonstration ---
if __name__ == "__main__":
    chatbot = ConfidentChatbot()

    print("\n--- Confident Customer Support Chatbot (Type 'exit' to quit) ---")
    while True:
        user_input = input("\nYour question: ")
        if user_input.lower() == 'exit':
            print("Exiting chatbot. Goodbye!")
            break
        
        chat_response = chatbot.ask(user_input)
        print(chat_response)

