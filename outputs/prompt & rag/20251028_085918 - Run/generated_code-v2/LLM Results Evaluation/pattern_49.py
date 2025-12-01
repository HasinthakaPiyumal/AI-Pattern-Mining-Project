import json

class LLMClient:
    def generate_response(self, prompt: str) -> str:
        # Simulate an LLM response in JSON format
        # In a real application, this would call an actual LLM API
        if "incorrect information" in prompt.lower():
            return json.dumps({
                "correctness": {"score": 2, "explanation": "The response contains factually incorrect information about the return policy."},
                "completeness": {"score": 4, "explanation": "The response partially addresses the query but misses details on return shipping."},
                "relevance": {"score": 3, "explanation": "The response is somewhat relevant but gets sidetracked."},
                "politeness_tone": {"score": 5, "explanation": "The tone is polite and professional."},
                "clarity": {"score": 3, "explanation": "The language is a bit ambiguous regarding the next steps."}
            })
        elif "long shipping times" in prompt.lower() and "apologize" in prompt.lower():
            return json.dumps({
                "correctness": {"score": 5, "explanation": "The information about shipping times is accurate."},
                "completeness": {"score": 5, "explanation": "The response fully addresses the shipping delay and offers a solution."},
                "relevance": {"score": 5, "explanation": "The response is highly relevant to the customer\'s issue."},
                "politeness_tone": {"score": 5, "explanation": "The tone is empathetic, apologetic, and professional."},
                "clarity": {"score": 5, "explanation": "The response is clear, concise, and easy to understand."}
            })
        else:
            return json.dumps({
                "correctness": {"score": 4, "explanation": "The information provided is generally correct."},
                "completeness": {"score": 3, "explanation": "The response could be more detailed in some areas."},
                "relevance": {"score": 4, "explanation": "The response is relevant to the query."},
                "politeness_tone": {"score": 4, "explanation": "The tone is appropriate."},
                "clarity": {"score": 4, "explanation": "The response is reasonably clear."}
            })

class CustomerSupportEvaluator:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def _construct_evaluation_prompt(self, customer_query: str, support_response: str) -> str:
        prompt = f"""You are an expert customer support quality assurance evaluator. Your task is to assess the quality of a support agent's response to a customer's query based on the following criteria:

Customer Query: {customer_query}
Support Agent Response: {support_response}

Evaluation Criteria:
1.  **Correctness (1-5):** Is the information provided accurate and factually sound?
2.  **Completeness (1-5):** Does the response address all aspects of the customer's query adequately?
3.  **Relevance (1-5):** Is the response directly pertinent to the customer's question or issue?
4.  **Politeness/Tone (1-5):** Is the language respectful, empathetic, and professional?
5.  **Clarity (1-5):** Is the response easy to understand, unambiguous, and well-structured?

Provide your evaluation as a JSON object with a 'score' (integer from 1 to 5) and an 'explanation' (string) for each criterion. Do not include any other text outside the JSON object.

Example JSON format:
{{
    "correctness": {{"score": 4, "explanation": "The information is mostly accurate."}},
    "completeness": {{"score": 3, "explanation": "Missing some details."}}
    // ... and so on for all criteria
}}

Your JSON evaluation:
"""
        return prompt

    def evaluate_response(self, customer_query: str, support_response: str) -> dict:
        prompt = self._construct_evaluation_prompt(customer_query, support_response)
        llm_output_json_str = self.llm_client.generate_response(prompt)
        try:
            evaluation_results = json.loads(llm_output_json_str)
            return evaluation_results
        except json.JSONDecodeError:
            return {"error": "Failed to parse LLM output as JSON", "raw_output": llm_output_json_str}

def main():
    llm_client = LLMClient()
    evaluator = CustomerSupportEvaluator(llm_client)

    print("--- Evaluation Scenario 1: Good Response ---")
    query1 = "My order #12345 has not arrived. Can you tell me its status?"
    response1 = "Hello! I apologize for the delay with your order #12345. It appears there was a logistics issue, and we are working to resolve it. We expect it to be delivered within the next 2-3 business days. You can track its updated status using this link: [tracking link]. Thank you for your patience."
    results1 = evaluator.evaluate_response(query1, response1)
    print(json.dumps(results1, indent=2))

    print("\n--- Evaluation Scenario 2: Response with some issues (simulated incorrect info) ---")
    query2 = "What is your return policy for electronics purchased more than 30 days ago?"
    response2 = "Our return policy for all electronics is 60 days, no questions asked. Just bring it to any store with your receipt. You will receive a full refund, and we offer free return shipping for online orders even if the item is not faulty. We also provide a 2-year warranty on all electronics, which covers accidental damage."
    # Simulating a response that contains incorrect information about the return policy
    results2 = evaluator.evaluate_response(query2, response2 + " (Note: This response contains incorrect information about the return policy for demonstration purposes.)")
    print(json.dumps(results2, indent=2))

    print("\n--- Evaluation Scenario 3: Response addressing long shipping times with apology ---")
    query3 = "I'm very upset about the long shipping times for my recent purchase. This is unacceptable!"
    response3 = "Dear customer, I sincerely apologize for the inconvenience caused by the longer-than-expected shipping times. We understand your frustration. We are actively working with our logistics partners to improve delivery speeds. As a token of our apology, we'd like to offer you a 15% discount on your next purchase. Your understanding is greatly appreciated."
    results3 = evaluator.evaluate_response(query3, response3)
    print(json.dumps(results3, indent=2))

if __name__ == "__main__":
    main()