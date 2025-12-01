import random

class ChatbotResponseInput:
    def __init__(self, customer_query: str, chatbot_response: str):
        self.customer_query = customer_query
        self.chatbot_response = chatbot_response

class LLMEvaluator:
    def __init__(self, likert_scale: list):
        self.likert_scale = likert_scale
        self.evaluation_criteria = [
            "accuracy",
            "helpfulness",
            "tone",
            "completeness",
        ]

    def _construct_prompt(self, query: str, response: str) -> str:
        criteria_str = ", ".join(self.evaluation_criteria)
        scale_str = ", ".join(self.likert_scale)
        prompt = (
            f"Evaluate the following chatbot response to a customer query based on {criteria_str}. "
            f"Rate the response using only one of the following categories: {scale_str}. "
            f"\nCustomer Query: {query}\nChatbot Response: {response}\nRating:"
        )
        return prompt

    def _mock_llm_call(self, prompt: str) -> str:
        # In a real application, this would involve an API call to an LLM (e.g., OpenAI, Google Generative AI)
        # For demonstration, we'll return a random rating from the Likert scale.
        return random.choice(self.likert_scale)

    def evaluate_response(self, customer_query: str, chatbot_response: str) -> str:
        prompt = self._construct_prompt(customer_query, chatbot_response)
        rating = self._mock_llm_call(prompt)
        return rating

class OutputAndReportingModule:
    def report_evaluation(self, query: str, response: str, rating: str):
        print(f"--- Chatbot Response Evaluation ---")
        print(f"Customer Query: {query}")
        print(f"Chatbot Response: {response}")
        print(f"LLM Rating: {rating}")
        print(f"-----------------------------------")

if __name__ == "__main__":
    # Define the Likert scale
    likert_scale_categories = ["Very Poor", "Poor", "Neutral", "Good", "Excellent"]

    # Initialize the evaluator and reporting module
    evaluator = LLMEvaluator(likert_scale_categories)
    reporter = OutputAndReportingModule()

    # Example usage
    query1 = "What are your operating hours?"
    response1 = "Our operating hours are from 9 AM to 5 PM, Monday to Friday."
    rating1 = evaluator.evaluate_response(query1, response1)
    reporter.report_evaluation(query1, response1, rating1)

    query2 = "I need help resetting my password. The link you sent isn't working."
    response2 = "I apologize for the inconvenience. Please try clearing your browser cache and cookies, then try the link again. If it still doesn't work, let me know, and I'll escalate your issue to a human agent."
    rating2 = evaluator.evaluate_response(query2, response2)
    reporter.report_evaluation(query2, response2, rating2)

    query3 = "Tell me about the history of quantum physics."
    response3 = "I am a customer support chatbot and cannot provide detailed information on academic subjects. Please refer to specialized resources."
    rating3 = evaluator.evaluate_response(query3, response3)
    reporter.report_evaluation(query3, response3, rating3)