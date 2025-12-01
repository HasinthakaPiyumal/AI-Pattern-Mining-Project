import re

class LLMMock:
    def generate_response(self, prompt: str) -> str:
        # Simulate an LLM response with a linear score (1-5) and justification.
        # In a real application, this would be an API call or an actual LLM inference.
        if "excellent" in prompt.lower() or "perfect" in prompt.lower():
            return "Score: 5 - The agent provided an excellent, clear, and very helpful response."
        elif "good" in prompt.lower() or "helpful" in prompt.lower():
            return "Score: 4 - The agent's response was good and addressed the query effectively."
        elif "adequate" in prompt.lower() or "sufficient" in prompt.lower():
            return "Score: 3 - The response was adequate, but could be more detailed or empathetic."
        elif "poor" in prompt.lower() or "unclear" in prompt.lower():
            return "Score: 2 - The agent's response was poor and didn't fully resolve the issue."
        else:
            return "Score: 1 - The response was unhelpful and did not address the customer's needs."

def create_evaluation_prompt(customer_query: str, agent_response: str) -> str:
    prompt = f"""
    Evaluate the following customer support agent's response to a customer query.
    Rate the agent's response on a linear scale from 1 to 5, where 1 is very poor and 5 is excellent.
    Consider clarity, helpfulness, accuracy, and tone.

    Customer Query: {customer_query}

    Agent Response: {agent_response}

    Provide the score as 'Score: [1-5]' followed by a brief justification.
    """
    return prompt

def parse_llm_output(llm_output: str) -> tuple[int | None, str | None]:
    score_match = re.search(r"Score:\s*(\d+)", llm_output)
    justification_match = re.search(r"Score:\s*\d+\s*-\s*(.*)", llm_output)

    score = int(score_match.group(1)) if score_match else None
    justification = justification_match.group(1).strip() if justification_match else None

    return score, justification

def evaluate_customer_support_response(customer_query: str, agent_response: str) -> dict:
    llm_mock = LLMMock()

    # Prompt Engineering
    evaluation_prompt = create_evaluation_prompt(customer_query, agent_response)

    # LLM Integration (using mock)
    llm_raw_output = llm_mock.generate_response(evaluation_prompt)

    # Output Parsing
    score, justification = parse_llm_output(llm_raw_output)

    return {"score": score, "justification": justification}

if __name__ == "__main__":
    print("--- AI-powered Customer Support Response Evaluator ---")
    print("Enter 'exit' at any time to quit.\n")

    while True:
        customer_query_input = input("Enter Customer Query (or 'exit'): ")
        if customer_query_input.lower() == 'exit':
            break

        agent_response_input = input("Enter Agent Response (or 'exit'): ")
        if agent_response_input.lower() == 'exit':
            break

        evaluation_result = evaluate_customer_support_response(customer_query_input, agent_response_input)

        print(f"\n--- Evaluation Result ---")
        print(f"Score: {evaluation_result['score'] if evaluation_result['score'] is not None else 'N/A'}/5")
        print(f"Justification: {evaluation_result['justification'] if evaluation_result['justification'] is not None else 'N/A'}")
        print("-------------------------\n")