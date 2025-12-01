
import os
from openai import OpenAI

class CustomerSupportEvaluator:
    def __init__(self, api_key=None, model_a="gpt-3.5-turbo", model_b="gpt-4", evaluator_model="gpt-4"):
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided. Please set OPENAI_API_KEY environment variable or pass it to the constructor.")
        self.client = OpenAI(api_key=api_key)
        self.model_a = model_a
        self.model_b = model_b
        self.evaluator_model = evaluator_model

    def _generate_response(self, query: str, model: str) -> str:
        """Simulates an LLM generating a response to a query."""
        print(f"Generating response using {model}...")
        try:
            chat_completion = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful customer support assistant."},
                    {"role": "user", "content": query}
                ],
                temperature=0.7,
                max_tokens=200
            )
            return chat_completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating response with {model}: {e}")
            return "[Error generating response]"

    def generate_pairwise_responses(self, query: str):
        """Generates two responses using different models/strategies."""
        response_a = self._generate_response(query, self.model_a)
        response_b = self._generate_response(query, self.model_b)
        return response_a, response_b

    def evaluate_responses(self, query: str, response1: str, response2: str) -> dict:
        """Uses an evaluator LLM to compare two responses."""
        print(f"Evaluating responses using {self.evaluator_model}...")
        prompt = f"""
        As an expert customer support evaluator, your task is to compare two responses to a customer query.
        Evaluate them based on helpfulness, clarity, conciseness, and tone.

        Customer Query: {query}

        --- Response 1 ---
        {response1}

        --- Response 2 ---
        {response2}
        ---

        Which response is superior? Provide your judgment (Response 1, Response 2, or Equal) and a brief justification.
        Format your output as follows:
        Judgment: [Response 1 / Response 2 / Equal]
        Justification: [Your explanation here]
        """

        try:
            chat_completion = self.client.chat.completions.create(
                model=self.evaluator_model,
                messages=[
                    {"role": "system", "content": "You are an impartial and expert evaluator of customer support responses."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=300
            )
            evaluation_output = chat_completion.choices[0].message.content.strip()

            # Parse the output
            judgment_line = next((line for line in evaluation_output.split('\n') if line.startswith("Judgment:")), None)
            justification_line = next((line for line in evaluation_output.split('\n') if line.startswith("Justification:")), None)

            judgment = judgment_line.split(": ")[1] if judgment_line else "N/A"
            justification = justification_line.split(": ")[1] if justification_line else "N/A"

            return {
                "judgment": judgment,
                "justification": justification,
                "raw_evaluation_output": evaluation_output
            }
        except Exception as e:
            print(f"Error evaluating responses with {self.evaluator_model}: {e}")
            return {
                "judgment": "Error",
                "justification": str(e),
                "raw_evaluation_output": ""
            }

    def run_evaluation(self, customer_query: str):
        """Orchestrates the full evaluation process."""
        print(f"\n--- Customer Query ---\n{customer_query}")

        response_a, response_b = self.generate_pairwise_responses(customer_query)

        print(f"\n--- Response from Model A ({self.model_a}) ---\n{response_a}")
        print(f"\n--- Response from Model B ({self.model_b}) ---\n{response_b}")

        evaluation_result = self.evaluate_responses(customer_query, response_a, response_b)

        print("\n--- Evaluation Result ---")
        print(f"Judgment: {evaluation_result['judgment']}")
        print(f"Justification: {evaluation_result['justification']}")
        print("\nNote: The order of inputs can heavily affect LLM evaluation. For robust A/B testing, consider randomizing input order or performing multiple evaluations.\n")

        return {
            "query": customer_query,
            "response_a": response_a,
            "response_b": response_b,
            "evaluation": evaluation_result
        }

if __name__ == "__main__":
    # Make sure to set your OPENAI_API_KEY environment variable
    # os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

    # Example Usage:
    evaluator = CustomerSupportEvaluator(
        model_a="gpt-3.5-turbo",  # Model for first response
        model_b="gpt-4",         # Model for second response
        evaluator_model="gpt-4"  # Model for evaluation
    )

    query1 = "My internet is not working. What should I do?"
    evaluator.run_evaluation(query1)

    print("\n" + "="*80 + "\n")

    query2 = "I want to cancel my subscription. How can I proceed?"
    evaluator.run_evaluation(query2)

    print("\n" + "="*80 + "\n")

    query3 = "What are the benefits of upgrading to the premium plan?"
    evaluator.run_evaluation(query3)

