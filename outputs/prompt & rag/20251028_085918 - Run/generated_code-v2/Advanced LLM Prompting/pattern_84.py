import os
from dotenv import load_dotenv
import openai
import random
import json

load_dotenv()

class LLMInteractionModule:
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate_text(self, prompt_text, model="gpt-3.5-turbo", temperature=0.7):
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error during LLM interaction: {e}"

class PromptGenerator:
    def __init__(self, llm_module: LLMInteractionModule):
        self.llm_module = llm_module

    def generate_initial_prompt(self, customer_query_context: str) -> str:
        prompt_template = f"""Generate an initial customer support prompt template for an e-commerce agent dealing with the following customer query context:
Context: {customer_query_context}

The prompt should guide the agent to provide helpful and concise information. Focus on empathy and resolution.
Example Output: 'Hello! Thank you for contacting us about [issue]. To help you best, could you please provide [necessary information]?'
"""
        return self.llm_module.generate_text(prompt_template)

class PromptOptimizer:
    def __init__(self, llm_module: LLMInteractionModule):
        self.llm_module = llm_module

    def optimize_prompt(self, current_prompt: str, feedback_score: int) -> str:
        feedback_description = ""
        if feedback_score >= 4:
            feedback_description = "This prompt was highly effective and led to good customer satisfaction. Suggest minor refinements or alternative phrasings to make it even better."
        elif feedback_score == 3:
            feedback_description = "This prompt was moderately effective but could be improved for clarity or conciseness. Suggest specific improvements."
        else:
            feedback_description = "This prompt was not very effective. It needs significant improvements in clarity, empathy, or directness. Provide a completely rephrased and improved prompt."

        prompt_template = f"""Given the following customer support prompt and its effectiveness feedback, suggest an improved version of the prompt.

Current Prompt: {current_prompt}
Feedback (score out of 5): {feedback_score}
Feedback Description: {feedback_description}

Provide only the improved prompt. Do not include any additional commentary.
"""
        return self.llm_module.generate_text(prompt_template)

class FeedbackSimulator:
    def simulate_feedback(self, prompt: str) -> int:
        # Simulate feedback based on prompt characteristics or randomly
        # For demonstration, let's make it somewhat random but lean towards improving with iterations
        # A more sophisticated version would analyze prompt content for keywords, length, etc.
        return random.randint(1, 5)

class Orchestrator:
    def __init__(self, llm_module: LLMInteractionModule):
        self.prompt_generator = PromptGenerator(llm_module)
        self.prompt_optimizer = PromptOptimizer(llm_module)
        self.feedback_simulator = FeedbackSimulator()

    def run_optimization_cycle(self, customer_query_context: str, iterations: int = 3):
        print(f"\n--- Starting Prompt Optimization for: {customer_query_context} ---")

        # Step 1: Generate initial prompt
        initial_prompt = self.prompt_generator.generate_initial_prompt(customer_query_context)
        print(f"\nInitial Prompt:\n{initial_prompt}")

        current_prompt = initial_prompt
        for i in range(iterations):
            print(f"\n--- Optimization Iteration {i+1} ---")
            
            # Step 2: Simulate feedback
            feedback_score = self.feedback_simulator.simulate_feedback(current_prompt)
            print(f"Simulated Feedback Score: {feedback_score}/5")

            # Step 3: Optimize prompt
            optimized_prompt = self.prompt_optimizer.optimize_prompt(current_prompt, feedback_score)
            print(f"Optimized Prompt:\n{optimized_prompt}")
            current_prompt = optimized_prompt
        
        print(f"\n--- Optimization Complete ---")
        print(f"Final Optimized Prompt:\n{current_prompt}")
        return current_prompt

if __name__ == "__main__":
    # Ensure OPENAI_API_KEY is set in your .env file or environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set. Please create a .env file or set the variable.")
    else:
        llm_interaction = LLMInteractionModule()
        orchestrator = Orchestrator(llm_interaction)

        customer_query_1 = "Customer wants to know the return policy for a defective item purchased 30 days ago."
        orchestrator.run_optimization_cycle(customer_query_1, iterations=3)

        print("\n" + "="*80 + "\n")

        customer_query_2 = "Customer is asking about the estimated delivery date for order #XYZ789. They placed the order last week."
        orchestrator.run_optimization_cycle(customer_query_2, iterations=2)
