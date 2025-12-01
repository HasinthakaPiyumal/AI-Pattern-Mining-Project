import re

class PromptEngineeringModule:
    def __init__(self, scoring_scale=(1, 5)):
        self.scoring_scale = scoring_scale

    def create_evaluation_prompt(self, customer_query: str, chatbot_response: str) -> str:
        min_score, max_score = self.scoring_scale
        prompt = f"""Evaluate the following chatbot response to a customer query.
Rate the response on a linear scale from {min_score} to {max_score} for helpfulness, accuracy, and politeness.

Customer Query: {customer_query}

Chatbot Response: {chatbot_response}

Evaluation Criteria:
- Helpfulness: How well does the response address the customer's need?
- Accuracy: Is the information provided correct and free of errors?
- Politeness: Is the tone appropriate and respectful?

Output your evaluation in the following JSON format:
{{"helpfulness": [score], "accuracy": [score], "politeness": [score]}}
"""
        return prompt

class LLMInteractionModule:
    def simulate_llm_response(self, prompt: str) -> str:
        # This is a simulated LLM response. In a real application,
        # this would involve an API call to an actual LLM.
        if "discount" in prompt.lower() and "sorry" not in prompt.lower():
            return '{"helpfulness": 4, "accuracy": 5, "politeness": 4}'
        elif "shipping" in prompt.lower() and "tracking" in prompt.lower():
            return '{"helpfulness": 5, "accuracy": 5, "politeness": 5}'
        else:
            return '{"helpfulness": 3, "accuracy": 3, "politeness": 3}'

class ScoreParsingModule:
    def parse_llm_output(self, llm_output: str) -> dict:
        try:
            scores = eval(llm_output) # Using eval for simplicity, but json.loads is safer for actual JSON
            if isinstance(scores, dict) and all(key in scores for key in ["helpfulness", "accuracy", "politeness"]):
                return scores
            else:
                raise ValueError("LLM output does not match expected score format.")
        except (SyntaxError, ValueError) as e:
            print(f"Error parsing LLM output: {e}. Output was: {llm_output}")
            return {"helpfulness": None, "accuracy": None, "politeness": None}

def evaluate_chatbot_response(customer_query: str, chatbot_response: str) -> dict:
    prompt_engineer = PromptEngineeringModule()
    llm_interactor = LLMInteractionModule()
    score_parser = ScoreParsingModule()

    prompt = prompt_engineer.create_evaluation_prompt(customer_query, chatbot_response)
    llm_output = llm_interactor.simulate_llm_response(prompt)
    evaluation_scores = score_parser.parse_llm_output(llm_output)

    return evaluation_scores

if __name__ == "__main__":
    # Example Usage 1
    query1 = "I want to know if there's a discount for bulk orders on your new line of products."
    response1 = "Yes, we offer a 10% discount on orders over $500 from our new product line. Please use code BULK10 at checkout."
    print(f"\n--- Evaluation 1 ---")
    print(f"Query: {query1}")
    print(f"Response: {response1}")
    scores1 = evaluate_chatbot_response(query1, response1)
    print(f"Evaluation Scores: {scores1}")

    # Example Usage 2
    query2 = "What is the estimated shipping time for orders to New York and can I track my package?"
    response2 = "Standard shipping to New York typically takes 3-5 business days. You will receive a tracking number via email once your order has shipped."
    print(f"\n--- Evaluation 2 ---")
    print(f"Query: {query2}")
    print(f"Response: {response2}")
    scores2 = evaluate_chatbot_response(query2, response2)
    print(f"Evaluation Scores: {scores2}")

    # Example Usage 3 (Generic/Less Specific Response)
    query3 = "I have a question about my recent order, number 12345."
    response3 = "Please provide more details about your issue so I can assist you better."
    print(f"\n--- Evaluation 3 ---")
    print(f"Query: {query3}")
    print(f"Response: {response3}")
    scores3 = evaluate_chatbot_response(query3, response3)
    print(f"Evaluation Scores: {scores3}")
