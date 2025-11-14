class ReasoningEngine:
    """Manages multi-step reasoning and metacognitive techniques for complex queries."""

    def __init__(self, llm_service, prompt_manager):
        self.llm_service = llm_service
        self.prompt_manager = prompt_manager

    def prompt_chain_reasoning(self, initial_query: str, max_steps: int = 3) -> str:
        """Implements a simple prompt chain for multi-step problem solving."""
        print(f"[ReasoningEngine] Starting Prompt Chain for query: \'{initial_query[:50]}...\'")
        current_response = ""
        thought_process = []
        context = {"original_query": initial_query}

        for step in range(1, max_steps + 1):
            if step == 1:
                step_prompt = self.prompt_manager.get_prompt(f"Decompose the following customer query into its core components and propose the first step to address it: {initial_query}", prompt_type="role_prompt", context={"role": "problem decomposer", "goal": "break down complex problems"})
            else:
                step_prompt = self.prompt_manager.get_prompt(f"Based on the previous steps and current response:\nPrevious Response: {current_response}\nOriginal Query: {initial_query}\nWhat is the next logical step to resolve this issue?", prompt_type="role_prompt", context={"role": "problem solver", "goal": "continue resolving the issue"})

            step_response = self.llm_service.generate(step_prompt, context=context, role="system", style="analytical")
            thought_process.append(f"Step {step}: {step_response}")
            current_response += f"\n[Step {step}]: {step_response}"

            if "issue resolved" in step_response.lower() or "no further steps" in step_response.lower():
                print(f"[ReasoningEngine] Prompt Chain concluded at step {step}.")
                break
        
        final_answer_prompt = self.prompt_manager.get_prompt(f"Based on the following thought process, provide a concise and helpful final answer to the original customer query:\n\nOriginal Query: {initial_query}\nThought Process:\n{'\n'.join(thought_process)}\n\nFinal Answer:", prompt_type="role_prompt", context={"role": "customer support agent", "goal": "provide a helpful final answer"})
        final_answer = self.llm_service.generate(final_answer_prompt, context=context, role="system", style="helpful")

        return final_answer

    def rephrase_and_respond(self, query: str) -> str:
        """Metacognitive technique: Rephrase the query for clarity before responding."""
        print(f"[ReasoningEngine] Applying Rephrase and Respond for query: \'{query[:50]}...\'")
        rephrase_prompt = self.prompt_manager.get_prompt(f"Rephrase the following customer query to ensure clear understanding, then generate a response based on the rephrased query:\n\nCustomer Query: {query}\nRephrased Query:", prompt_type="role_prompt", context={"role": "clarifier", "goal": "ensure clear understanding"})
        
        rephrased_query = self.llm_service.generate(rephrase_prompt, role="system", style="clear")
        # In a real scenario, the LLM would likely generate both rephrased and initial response
        # For this mock, we'll simulate separate generation for demonstration.
        
        response_prompt = self.prompt_manager.get_prompt(rephrased_query, prompt_type="default")
        response = self.llm_service.generate(response_prompt, role="user", style="neutral")
        return f"(Rephrased: {rephrased_query}) {response}"

    def rereading_for_deeper_understanding(self, query: str, initial_response: str) -> str:
        """Metacognitive technique: Reread the query and initial response for discrepancies/deeper context."""
        print(f"[ReasoningEngine] Applying Rereading for query: \'{query[:50]}...\'")
        reread_prompt = self.prompt_manager.get_prompt(f"Consider the original customer query and the initial response. Are there any nuances or missed details in the query that the initial response did not fully address? If so, generate an improved response.\n\nOriginal Query: {query}\nInitial Response: {initial_response}\nImproved Response (if any, otherwise confirm initial response is adequate):", prompt_type="role_prompt", context={"role": "critical analyzer", "goal": "identify missed nuances"})
        
        improved_response = self.llm_service.generate(reread_prompt, role="system", style="analytical")
        
        if "initial response is adequate" in improved_response.lower() or "no further improvement" in improved_response.lower():
            print("[ReasoningEngine] Rereading confirmed initial response is adequate.")
            return initial_response
        else:
            print("[ReasoningEngine] Rereading led to an improved response.")
            return improved_response

    def metacognitive_prompting(self, query: str) -> str:
        """Integrates a general metacognitive self-reflection process."""
        print(f"[ReasoningEngine] Applying Metacognitive Prompting for query: \'{query[:50]}...\'")
        metacog_prompt = self.prompt_manager.get_prompt(f"Before generating a final answer, think step-by-step about the customer's query. Consider potential ambiguities, ethical implications, and the best way to provide a helpful and accurate response. Finally, provide the response.\n\nCustomer Query: {query}\nThought Process:\n", prompt_type="role_prompt", context={"role": "self-reflector", "goal": "think critically and ethically"})
        
        metacog_response = self.llm_service.generate(metacog_prompt, role="system", style="reflective")
        
        # For simplicity, we assume the LLM directly outputs the thought process followed by the answer.
        # In a real system, we might parse the thought process and then ask for the final answer.
        return metacog_response