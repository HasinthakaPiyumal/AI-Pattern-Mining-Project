import json

class CustomerInteraction:
    def __init__(self, query: str, agent_response: str):
        self.query = query
        self.agent_response = agent_response

class AutoCoTStepGenerator:
    def generate_cot_steps(self, evaluation_instructions: str, interaction: CustomerInteraction) -> list[str]:
        # Simulate AutoCoT step generation based on instructions and interaction
        # In a real-world scenario, this would likely involve a smaller LLM or more complex logic.
        steps = [
            f"Consider the customer's query: '{interaction.query}'.",
            f"Analyze the agent's response: '{interaction.agent_response}'.",
            "Evaluate if the agent's response directly addresses the customer's query.",
            "Assess the clarity, helpfulness, and completeness of the agent's response.",
            "Determine if the response maintains a professional and empathetic tone.",
            "Based on the above, provide a score and detailed reasoning."
        ]
        return steps

class PromptConstructor:
    def construct_geval_prompt(self, evaluation_instructions: str, interaction: CustomerInteraction, cot_steps: list[str]) -> str:
        cot_section = "\n".join([f"- {step}" for step in cot_steps])

        prompt = (
            f"Evaluation Instructions: {evaluation_instructions}\n\n"
            f"Customer Query: {interaction.query}\n"
            f"Agent Response: {interaction.agent_response}\n\n"
            f"Follow these Chain-of-Thought steps to evaluate the response:\n"
            f"{cot_section}\n\n"
            f"Based on the evaluation instructions and the Chain-of-Thought steps, provide a score (1-5) and detailed reasoning for the agent's response. The reasoning should explicitly refer to each step.\n"
            f"Format your output as a JSON object with 'score' (integer) and 'reasoning' (string) fields."
        )
        return prompt

class SimulatedEvaluationLLMOrchestrator:
    def evaluate(self, prompt: str) -> str:
        # Simulate an LLM's response
        # In a real application, this would be an API call to OpenAI, Llama 2, etc.
        if "directs" in prompt and "complete" in prompt:
            score = 5
            reasoning = (
                "The agent's response directly addressed the customer's query (Step 1, 2, 3). "
                "It was clear, helpful, and complete, covering all aspects of the query (Step 4). "
                "The tone was professional and empathetic (Step 5). "
                "Overall, an excellent response (Step 6)."
            )
        else:
            score = 3
            reasoning = (
                "The agent's response partially addressed the customer's query (Step 1, 2, 3). "
                "It lacked some clarity and completeness, leaving some aspects unaddressed (Step 4). "
                "The tone was generally professional (Step 5), but there's room for improvement in providing a comprehensive answer (Step 6)."
            )
        return json.dumps({"score": score, "reasoning": reasoning})

class EvaluationParser:
    def parse_llm_output(self, llm_output: str) -> dict:
        try:
            return json.loads(llm_output)
        except json.JSONDecodeError:
            return {"score": None, "reasoning": "Error parsing LLM output."}

class OutputModule:
    def display_results(self, evaluation_results: dict):
        print("\n--- Evaluation Results ---")
        print(f"Score: {evaluation_results.get('score', 'N/A')}/5")
        print(f"Reasoning: {evaluation_results.get('reasoning', 'No reasoning provided.')}")
        print("--------------------------\n")


if __name__ == "__main__":
    # Initialize components
    cot_generator = AutoCoTStepGenerator()
    prompt_constructor = PromptConstructor()
    llm_orchestrator = SimulatedEvaluationLLMOrchestrator()
    eval_parser = EvaluationParser()
    output_module = OutputModule()

    # Example 1: Good Response
    print("Evaluating Example 1: Good Response")
    eval_instructions_1 = "Evaluate the agent's response for its accuracy, completeness, and professionalism."
    interaction_1 = CustomerInteraction(
        query="My internet is not working. What should I do?",
        agent_response="Please restart your router and modem. If that doesn't work, ensure all cables are securely connected. You can also visit our troubleshooting guide at example.com/help for more steps. If the issue persists, contact technical support at 1-800-XXX-XXXX."
    )

    cot_steps_1 = cot_generator.generate_cot_steps(eval_instructions_1, interaction_1)
    geval_prompt_1 = prompt_constructor.construct_geval_prompt(eval_instructions_1, interaction_1, cot_steps_1)
    llm_raw_output_1 = llm_orchestrator.evaluate(geval_prompt_1)
    parsed_results_1 = eval_parser.parse_llm_output(llm_raw_output_1)
    output_module.display_results(parsed_results_1)

    # Example 2: Mediocre Response
    print("Evaluating Example 2: Mediocre Response")
    eval_instructions_2 = "Assess the agent's ability to provide a clear and helpful solution to the customer's problem."
    interaction_2 = CustomerInteraction(
        query="I can't log into my account. My password isn't working.",
        agent_response="You can reset your password. Just click on the 'Forgot Password' link on the login page."
    )

    cot_steps_2 = cot_generator.generate_cot_steps(eval_instructions_2, interaction_2)
    geval_prompt_2 = prompt_constructor.construct_geval_prompt(eval_instructions_2, interaction_2, cot_steps_2)
    llm_raw_output_2 = llm_orchestrator.evaluate(geval_prompt_2)
    parsed_results_2 = eval_parser.parse_llm_output(llm_raw_output_2)
    output_module.display_results(parsed_results_2)

    # Example 3: Simulating a bad response to show lower score
    print("Evaluating Example 3: Bad Response")
    eval_instructions_3 = "Evaluate if the agent's response is helpful and completely resolves the customer's issue."
    interaction_3 = CustomerInteraction(
        query="My package hasn't arrived. What's the status?",
        agent_response="Packages sometimes get delayed."
    )

    cot_steps_3 = cot_generator.generate_cot_steps(eval_instructions_3, interaction_3)
    geval_prompt_3 = prompt_constructor.construct_geval_prompt(eval_instructions_3, interaction_3, cot_steps_3)
    llm_raw_output_3 = llm_orchestrator.evaluate(geval_prompt_3)
    parsed_results_3 = eval_parser.parse_llm_output(llm_raw_output_3)
    output_module.display_results(parsed_results_3)
