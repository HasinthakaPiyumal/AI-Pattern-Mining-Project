class ConstitutionalChatbot:
    def __init__(self, principles):
        self.principles = principles
        # In a real implementation, you would load your base LLM here
        # self.base_llm = load_llm("your-llm-model")
        # You might also have a separate LLM for critique/revision or use the same one with different prompts

    def _generate_initial_response(self, query):
        """Simulates an LLM generating an initial response."""
        # This would be an actual call to your LLM API or local model
        # Example: response = self.base_llm.generate(query)
        print(f"\n--- Initial LLM thought for query: '{query}' ---")
        if "sensitive financial advice" in query.lower():
            return "Based on your query, here is some general financial information and common investment strategies."
        elif "medical diagnosis" in query.lower() or "medication" in query.lower():
            return "I am not equipped to provide medical diagnoses or recommend medication. Here is some general health information."
        elif "political opinion" in query.lower():
            return "I aim to provide neutral and factual information on political topics, avoiding personal opinions."
        else:
            return f"Hello! How can I assist you with '{query}' today?"

    def _critique_response(self, response, query):
        """
        Simulates an AI critiquing the response against the constitutional principles.
        In a real system, an LLM would analyze the response and principles.
        """
        print(f"--- Critiquing response: '{response}' ---")
        critiques = []
        if ("medical diagnosis" in query.lower() or "medication" in query.lower()) and "medical diagnoses or recommend medication" not in response.lower():
            critiques.append("The response attempts to provide medical advice or diagnosis, violating the 'Do no harm' principle.")
        if "sensitive financial advice" in query.lower() and "certified financial advisor" not in response.lower():
            critiques.append("The response provides general information for sensitive financial advice, which might be perceived as unhelpful or insufficient, though it avoids harm. Consider if it fully adheres to 'Be helpful and honest' and advises professional consultation.")
        if "political opinion" in query.lower() and "neutral and factual information" not in response.lower():
             critiques.append("The response might express an opinion or not explicitly state neutrality on a political topic, potentially violating 'Be unbiased and objective'.")
        if not critiques:
            critiques.append("The response generally adheres to the constitutional principles.")
        return critiques

    def _revise_response(self, response, critiques):
        """
        Simulates an AI revising the response based on the critiques.
        In a real system, an LLM would take the response and critiques to generate a new version.
        """
        print(f"--- Revising response based on critiques: {critiques} ---")
        revised_response = response
        for critique in critiques:
            if "medical advice or diagnosis" in critique:
                revised_response = "As an AI, I cannot provide medical diagnoses or recommend medication. Please consult a qualified healthcare professional for any health concerns or medical advice."
            elif "unhelpful or insufficient" in critique:
                revised_response = "I can provide general information regarding your financial query, but for personalized and sensitive financial advice, I strongly recommend consulting a certified financial advisor or a financial institution."
            elif "not explicitly state neutrality on a political topic" in critique:
                revised_response = "As an AI, I do not hold personal opinions. I can provide neutral and factual information regarding political topics. Do you have a specific question you'd like factual information on?"
            # Add more revision rules based on critique types as needed
        
        # If no specific revision rule fired but critiques exist, try to refine generic statements
        if revised_response == response and len(critiques) > 0 and "The response generally adheres to the constitutional principles." not in critiques:
             return f"Response needs refinement based on critique. Original: '{response}'"
        return revised_response

    def chat(self, query, max_iterations=3):
        """
        Conducts a chat interaction with iterative critique and revision.
        """
        current_response = self._generate_initial_response(query)
        print(f"Initial AI Response: {current_response}")

        for i in range(max_iterations):
            print(f"\n--- Iteration {i+1} of Constitutional Alignment ---")
            critiques = self._critique_response(current_response, query)

            if "The response generally adheres to the constitutional principles." in critiques and len(critiques) == 1:
                print("Response aligns with principles. No further revision needed.")
                break
            else:
                print(f"Critiques identified: {critiques}")
                new_response = self._revise_response(current_response, critiques)
                if new_response == current_response and "The response generally adheres to the constitutional principles." not in critiques:
                    print("Revision did not change the response; further refinement may be needed manually or with a more sophisticated revision model.")
                    break # Avoid infinite loop if revision doesn't change anything
                current_response = new_response
                print(f"Revised AI Response: {current_response}")

        print("\n--- Final Aligned AI Response ---")
        return current_response

# Example Usage:
# Define a simple set of constitutional principles for an ethical chatbot
constitutional_principles = [
    "1. Be helpful and honest.",
    "2. Do no harm, especially concerning sensitive topics like health, finance, or legal matters.",
    "3. Be unbiased and objective; do not express personal opinions or political stances.",
    "4. Protect user privacy and do not ask for or store sensitive personal information.",
    "5. Do not generate hate speech, discriminatory content, explicit material, or promote illegal activities.",
    "6. Refer to human experts for complex or sensitive issues where AI assistance is insufficient or inappropriate."
]

# Initialize the chatbot with the defined principles
ethical_chatbot = ConstitutionalChatbot(constitutional_principles)

# --- Test Queries ---

print("\n========================================")
print("\n--- Test Query 1: Medical advice ---")
response1 = ethical_chatbot.chat("Can you tell me if this rash looks serious and what medication I should take?")
print(f"Final Chatbot Output: {response1}")

print("\n========================================")
print("\n--- Test Query 2: Sensitive financial advice ---")
response2 = ethical_chatbot.chat("I need sensitive financial advice on investing my life savings to minimize taxes. Can you guide me on specific stocks?")
print(f"Final Chatbot Output: {response2}")

print("\n========================================")
print("\n--- Test Query 3: General information ---")
response3 = ethical_chatbot.chat("What are the main causes of climate change?")
print(f"Final Chatbot Output: {response3}")

print("\n========================================")
print("\n--- Test Query 4: Potentially controversial political opinion ---")
response4 = ethical_chatbot.chat("What's your opinion on the latest government policy regarding economic stimulus?")
print(f"Final Chatbot Output: {response4}")

print("\n========================================")
print("\n--- Test Query 5: Harmless general query ---")
response5 = ethical_chatbot.chat("Tell me a fun fact about pandas.")
print(f"Final Chatbot Output: {response5}")