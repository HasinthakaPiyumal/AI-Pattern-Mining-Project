import json
import os

# Placeholder for LLM API Key and Model
# In a real application, these would be loaded securely, e.g., from environment variables
# LLM_API_KEY = os.getenv("LLM_API_KEY")
# LLM_MODEL = "gpt-3.5-turbo"

class ChatEvaluator:
    def __init__(self):
        # Initialize LLM client here if using a real API
        # For this example, we'll simulate the LLM response.
        pass

    def _generate_llm_response(self, prompt: str) -> str:
        # This is a simulated LLM response for demonstration purposes.
        # In a real application, you would make an API call to an LLM provider.
        # Example with OpenAI:
        # from openai import OpenAI
        # client = OpenAI(api_key=LLM_API_KEY)
        # response = client.chat.completions.create(
        #     model=LLM_MODEL,
        #     messages=[{"role": "user", "content": prompt}],
        #     response_format={ "type": "json_object" }
        # )
        # return response.choices[0].message.content

        # Simulated JSON response based on the architecture's example
        if "lack empathy" in prompt.lower() or "partially resolved" in prompt.lower():
            return json.dumps({
                "clarity": "Good",
                "accuracy": "Excellent",
                "empathy": "Fair",
                "resolution": "Partial",
                "score": 3.5,
                "reasoning": "The agent provided accurate information but lacked empathy in tone and only partially resolved the issue by suggesting a workaround instead of a direct solution."
            })
        else:
            return json.dumps({
                "clarity": "Excellent",
                "accuracy": "Excellent",
                "empathy": "Excellent",
                "resolution": "Full",
                "score": 5.0,
                "reasoning": "The agent fully understood the query, provided accurate and empathetic responses, and resolved the issue completely."
            })


    def evaluate_response(self, customer_query: str, agent_response: str) -> dict:
        prompt = f"""
        You are an AI assistant designed to evaluate customer support chat interactions.
        Your task is to assess the agent's response based on clarity, accuracy, empathy, and resolution.
        Provide your evaluation in a JSON format with the following keys:
        'clarity': A rating (e.g., 'Poor', 'Fair', 'Good', 'Excellent') of how clear the agent's response was.
        'accuracy': A rating of how accurate the information provided by the agent was.
        'empathy': A rating of how empathetic the agent's tone and language were.
        'resolution': A rating of whether the issue was 'Unresolved', 'Partial', or 'Full'.
        'score': A numerical score between 1.0 and 5.0, where 5.0 is excellent.
        'reasoning': A brief explanation for your ratings and overall score.

        Customer Query: {customer_query}
        Agent Response: {agent_response}

        Please provide your evaluation in JSON format only.
        """

        llm_output = self._generate_llm_response(prompt)
        try:
            evaluation = json.loads(llm_output)
            return evaluation
        except json.JSONDecodeError:
            print(f"Error: LLM did not return valid JSON. Output: {llm_output}")
            return {"error": "Invalid JSON output from LLM", "raw_output": llm_output}


if __name__ == "__main__":
    evaluator = ChatEvaluator()

    # Example 1: Good response
    query1 = "My internet is not working. What should I do?"
    response1 = "I understand this is frustrating. Please try restarting your router and modem. If that doesn't work, contact us again for further assistance."
    print("\n--- Evaluation for Example 1 ---")
    evaluation1 = evaluator.evaluate_response(query1, response1)
    print(json.dumps(evaluation1, indent=2))

    # Example 2: Response with some issues
    query2 = "I was charged incorrectly for my last bill. Can you help?"
    response2 = "The system shows your charges are correct. You can view your detailed bill online if you need more information."
    print("\n--- Evaluation for Example 2 ---")
    evaluation2 = evaluator.evaluate_response(query2, response2)
    print(json.dumps(evaluation2, indent=2))