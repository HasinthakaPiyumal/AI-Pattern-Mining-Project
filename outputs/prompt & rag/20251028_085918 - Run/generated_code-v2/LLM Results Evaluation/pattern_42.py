"""
Main application file for the AI-powered customer support chatbot.
This single file combines configuration, LLM service, prompt management, and the main application logic
for demonstration purposes, adhering to the single-file output constraint.
"""

import os

# --- Configuration (formerly config.py) ---
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"

# --- LLM Service (formerly llm_service.py) ---
class LLMService:
    def __init__(self, api_key: str):
        if not api_key or api_key == "YOUR_OPENAI_API_KEY":
            print("Warning: OpenAI API key is not set. Using mock LLM response.")
            self.mock_mode = True
        else:
            self.mock_mode = False
            # In a real application, you would initialize your LLM client here, e.g.:
            # import openai
            # self.client = openai.OpenAI(api_key=api_key)

    def get_completion(self, prompt: str) -> str:
        """
        Sends a prompt to the LLM and returns its completion.
        """
        if self.mock_mode:
            print("--- Mock LLM received prompt ---")
            print(prompt)
            print("---------------------------------")
            # Simple mock responses based on potential ambiguities
            if "delivery status" in prompt.lower():
                return "I understand you're asking about delivery status. Could you please provide your order number? Are you looking for the current location, estimated arrival, or something else?"
            elif "account issue" in prompt.lower():
                return "I can help with account issues. Are you having trouble logging in, changing your password, or do you need to update your personal information?"
            return "Thank you for your query. Your request is ambiguous. I can interpret it in multiple ways. Could you please clarify what you mean by that? For example, are you asking about X, Y, or Z?"

        # For a real application, use the LLM client, e.g.:
        # try:
        #     response = self.client.chat.completions.create(
        #         model="gpt-3.5-turbo",  # or another suitable model
        #         messages=[
        #             {"role": "system", "content": "You are a helpful customer support assistant that can handle ambiguous queries."},
        #             {"role": "user", "content": prompt}
        #         ],
        #         max_tokens=150
        #     )
        #     return response.choices[0].message.content.strip()
        # except Exception as e:
        #     return f"Error communicating with LLM: {e}"

# --- Prompt Manager (formerly prompt_manager.py) ---
class PromptManager:
    def __init__(self):
        self._ambiguous_examples = self._load_ambiguous_examples()

    def _load_ambiguous_examples(self) -> list[dict]:
        """
        Loads predefined ambiguous examples.
        In a real scenario, this data could be loaded from a database, a dedicated configuration file, or dynamically retrieved.
        """
        return [
            {
                "question": "What's the status?",
                "ambiguous_labels": [
                    "Are you asking about an order status?",
                    "Are you asking about a refund status?",
                    "Are you asking about a service outage status?"
                ],
                "demonstration_response": "I can help with status updates! To clarify, are you asking about an order, a refund, or a service outage? Please specify what kind of status you're interested in."
            },
            {
                "question": "I have an account issue.",
                "ambiguous_labels": [
                    "Are you having trouble logging in?",
                    "Do you need to reset your password?",
                    "Are you trying to update your profile information?"
                ],
                "demonstration_response": "I understand you're experiencing an account issue. To assist you better, could you please tell me more? For instance, are you having trouble logging in, resetting your password, or updating your profile?"
            },
            {
                "question": "Tell me about the product.",
                "ambiguous_labels": [
                    "Are you looking for specifications of a particular product?",
                    "Are you asking for a comparison between products?",
                    "Are you interested in pricing or availability?"
                ],
                "demonstration_response": "I'd be happy to provide product information! To narrow it down, are you looking for details on a specific product, comparing products, or interested in pricing and availability?"
            },
            {
                "question": "How do I change it?",
                "ambiguous_labels": [
                    "Are you asking how to change your shipping address?",
                    "Are you asking how to change your payment method?",
                    "Are you asking how to change an order item?"
                ],
                "demonstration_response": "I can guide you through making changes. Could you clarify what 'it' refers to? For example, are you trying to change your shipping address, payment method, or an item in your order?"
            }
        ]

    def construct_prompt(self, user_query: str) -> str:
        """
        Constructs a comprehensive prompt including ambiguous demonstrations for In-Context Learning.
        """
        prompt_parts = [
            "You are a helpful and empathetic customer support chatbot. Your primary goal is to assist users efficiently. When faced with an ambiguous question, instead of giving a single answer, acknowledge the ambiguity and ask clarifying questions based on different possible interpretations provided in examples. If the user query matches an ambiguous example, try to respond in a similar clarifying manner.",
            "\nHere are some examples of ambiguous customer queries and how to handle them:"
        ]

        for example in self._ambiguous_examples:
            prompt_parts.append(f"\nCustomer: {example['question']}")
            prompt_parts.append(f"Chatbot (Ambiguous Demonstration): {example['demonstration_response']}")

        prompt_parts.append(f"\nCustomer: {user_query}")
        prompt_parts.append(f"Chatbot:") # Let the LLM complete this based on the demonstrations

        return "\n".join(prompt_parts)

# --- Main Application Logic (formerly main.py) ---
def main():
    print("Starting Ambiguous Demonstrations Chatbot...")
    
    # Initialize services
    prompt_manager = PromptManager()
    llm_service = LLMService(api_key=OPENAI_API_KEY)

    print("\nType your query. Type 'exit' to quit.")
    while True:
        user_query = input("\nCustomer: ")
        if user_query.lower() == 'exit':
            print("Exiting chatbot. Goodbye!")
            break

        # Construct the prompt with ambiguous demonstrations
        full_prompt = prompt_manager.construct_prompt(user_query)
        
        # Get response from the LLM
        chatbot_response = llm_service.get_completion(full_prompt)
        
        print(f"Chatbot: {chatbot_response}")

if __name__ == "__main__":
    main()