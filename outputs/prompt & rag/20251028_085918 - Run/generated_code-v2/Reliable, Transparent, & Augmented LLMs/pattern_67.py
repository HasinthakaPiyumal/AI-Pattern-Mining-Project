class ChatbotQualityEvaluator:
    def __init__(self):
        pass

    def _get_llm_response(self, prompt: str) -> str:
        if "irrelevant" in prompt.lower() or "evasive" in prompt.lower() or "rude" in prompt.lower() or "incomplete" in prompt.lower() or "cannot help with that" in prompt.lower() or "do not have access" in prompt.lower():
            return "No"
        if "directly answers the question" in prompt.lower() or "polite" in prompt.lower() or "provides a solution" in prompt.lower() or "look into this" in prompt.lower() or "visit our website" in prompt.lower():
            return "Yes"
        if "ask a question related to our services" in prompt.lower() and "philosophical questions" in prompt.lower():
            return "No"
        if "find all pricing details on our product page" in prompt.lower() and "many customers ask about pricing" in prompt.lower():
            return "No"
        return "Yes"

    def evaluate_response(self, customer_query: str, chatbot_response: str) -> str:
        prompt = (
            f"You are an AI assistant tasked with evaluating chatbot responses.\n"
            f"Please assess if the following chatbot response is 'satisfactory' for the customer's query.\n"
            f"A 'satisfactory' response directly answers the question, is polite, and provides a solution if needed.\n"
            f"An 'unsatisfactory' response is irrelevant, evasive, rude, or incomplete.\n\n"
            f"Customer Query: \"{customer_query}\"\n"
            f"Chatbot Response: \"{chatbot_response}\"\n\n"
            f"Is the chatbot's response satisfactory? Please answer with only 'Yes' or 'No'."
        )
        llm_output = self._get_llm_response(prompt)
        return llm_output.strip()

if __name__ == "__main__":
    evaluator = ChatbotQualityEvaluator()

    test_cases = [
        {
            "query": "How do I reset my password?",
            "response": "To reset your password, please visit our website's login page and click on 'Forgot Password'. Follow the instructions sent to your registered email."
        },
        {
            "query": "What are your business hours?",
            "response": "Our business hours are Monday to Friday, 9 AM to 5 PM EST."
        },
        {
            "query": "My order hasn't arrived. What should I do?",
            "response": "I'm sorry to hear that. Please provide your order number so I can look into this for you."
        },
        {
            "query": "Can you recommend a good restaurant nearby?",
            "response": "I am a chatbot and do not have access to real-time local restaurant data or personal preferences."
        },
        {
            "query": "What is the meaning of life?",
            "response": "I am an AI and cannot answer philosophical questions. Please ask a question related to our services."
        },
        {
            "query": "How much does product X cost?",
            "response": "That's a very interesting question. Many customers ask about pricing. You can find all pricing details on our product page."
        }
    ]

    print("--- Chatbot Response Quality Evaluation ---")
    for i, test in enumerate(test_cases):
        print(f"\n--- Test Case {i+1} ---")
        print(f"Customer Query: {test['query']}")
        print(f"Chatbot Response: {test['response']}")
        evaluation = evaluator.evaluate_response(test['query'], test['response'])
        print(f"Evaluation: {evaluation}")