class GenerativeModelSimulator:
    def __init__(self):
        self.knowledge_base = {
            "shipping status": "Your order #12345 is currently in transit and expected to arrive by next Friday.",
            "return policy": "You can return any item within 30 days of purchase, provided it is in its original condition.",
            "payment methods": "We accept Visa, Mastercard, American Express, and PayPal."
        }
        self.special_tokens_mapping = {
            "[QA]": "question answering",
            "[SUMMARY]": "summarization",
            "[RESPOND]": "response generation",
            "[PARAPHRASE]": "paraphrasing"
        }

    def generate(self, input_text: str) -> str:
        if not input_text:
            return ""

        for token, task_name in self.special_tokens_mapping.items():
            if input_text.startswith(token):
                content = input_text[len(token):].strip()
                if task_name == "question answering":
                    for key, value in self.knowledge_base.items():
                        if key in content.lower():
                            return f"Simulated QA for '{content}': {value}"
                    return f"Simulated QA for '{content}': I don't have information on that specific question in my knowledge base."
                elif task_name == "summarization":
                    return f"Simulated Summary of: '{content[:50]}...'"
                elif task_name == "response generation":
                    if "issue with my order" in content.lower():
                        return f"Simulated Response to '{content}': We apologize for the inconvenience. Please provide your order number so we can investigate."
                    return f"Simulated Response to '{content}': Thank you for contacting us. How can I further assist you?"
                elif task_name == "paraphrasing":
                    return f"Simulated Paraphrase of '{content}': It seems you are asking about '{content.lower().replace('problem', 'issue').replace('help', 'assistance')}'"
        return f"Simulated default generation for: {input_text}"

class CustomerSupportAgent:
    def __init__(self, model_simulator: GenerativeModelSimulator):
        self.model_simulator = model_simulator

    def handle_question_answering(self, question: str, context: str = "") -> str:
        input_text = f"[QA] {question} {context}".strip()
        return self.model_simulator.generate(input_text)

    def summarize_ticket(self, transcript: str) -> str:
        input_text = f"[SUMMARY] {transcript}"
        return self.model_simulator.generate(input_text)

    def generate_response(self, query: str, context: str = "") -> str:
        input_text = f"[RESPOND] {query} {context}".strip()
        return self.model_simulator.generate(input_text)

    def paraphrase_query(self, ambiguous_statement: str) -> str:
        input_text = f"[PARAPHRASE] {ambiguous_statement}"
        return self.model_simulator.generate(input_text)

# Example Usage:
if __name__ == "__main__":
    simulator = GenerativeModelSimulator()
    agent = CustomerSupportAgent(simulator)

    print("--- Question Answering ---")
    qa_context = "Customer inquired about their order. Order #12345."
    print(agent.handle_question_answering("What is the status of my shipping?", qa_context))
    print(agent.handle_question_answering("What is your return policy?"))
    print(agent.handle_question_answering("Do you accept Bitcoin?"))

    print("\n--- Support Ticket Summarization ---")
    transcript = "Customer: Hi, I have an issue with my recent purchase. Agent: Can you please provide your order number? Customer: Yes, it's 67890. Agent: Thank you. Let me check. Customer: The item arrived damaged. Agent: I understand. We can arrange a return or replacement for you." 
    print(agent.summarize_ticket(transcript))

    print("\n--- Automated Response Generation ---")
    print(agent.generate_response("I have an issue with my order."))
    print(agent.generate_response("I want to know about your services."))

    print("\n--- Customer Query Paraphrasing ---")
    print(agent.paraphrase_query("I have a big problem with my item."))
    print(agent.paraphrase_query("Can you help me with this?"))
