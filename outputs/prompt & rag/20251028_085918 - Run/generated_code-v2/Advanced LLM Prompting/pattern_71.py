class BiasAwareChatbot:
    def __init__(self):
        self.bias_mitigation_instruction = """
You are an unbiased and fair customer support assistant for an e-commerce platform. 
Always provide neutral, respectful, and helpful responses, avoiding any stereotypes, 
personal opinions, or discriminatory language. Focus on facts and product information.
"""

    def _simulate_llm_response(self, prompt: str) -> str:
        if "expensive" in prompt.lower() and "poor" in prompt.lower():
            return "I understand you're looking for affordable options. We have a wide range of products at different price points. Could you please specify the type of product you are interested in?"
        elif "female developer" in prompt.lower() or "male nurse" in prompt.lower():
            return "At our company, we believe in equal opportunities and value diverse talents. We have many skilled individuals in all roles, regardless of gender. How can I assist you with your query?"
        elif "old people" in prompt.lower() or "young people" in prompt.lower():
            return "Our products are designed to be user-friendly and accessible for everyone. Please let me know what specific features or products you are interested in, and I'll be happy to provide details."
        else:
            return f"Thank you for your query. How may I assist you further regarding {prompt.strip()}?"

    def generate_unbiased_response(self, user_query: str) -> str:
        full_prompt = f"{self.bias_mitigation_instruction}\n\nCustomer Query: {user_query}\nAssistant Response:"
        print(f"\n--- Full Prompt Sent to LLM ---\n{full_prompt}\n------------------------------\n")
        response = self._simulate_llm_response(full_prompt)
        return response


def main():
    chatbot = BiasAwareChatbot()
    print("Welcome to the Bias-Aware Customer Support Chatbot. Type 'exit' to quit.")

    while True:
        user_input = input("\nCustomer: ")
        if user_input.lower() == 'exit':
            break
        
        response = chatbot.generate_unbiased_response(user_input)
        print(f"Assistant: {response}")

if __name__ == "__main__":
    main()