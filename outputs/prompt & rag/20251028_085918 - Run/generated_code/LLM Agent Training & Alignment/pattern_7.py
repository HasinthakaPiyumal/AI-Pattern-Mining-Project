import random
import time

# --- 1. Behavior Cloning (Simulated Initial Agent Policy) ---
class BehaviorClonedLLM:
    """
    Simulates a Large Language Model pre-trained via Behavior Cloning
    on human customer support interactions.
    """
    def __init__(self, model_name="telecom_agent_bc_v1"):
        self.model_name = model_name
        self.knowledge_base = {
            "internet slow": "Please try restarting your router and modem. If the issue persists, we can run a remote diagnostic.",
            "bill inquiry": "Could you please provide your account number? I can then look up your latest bill details.",
            "new plan": "We have several exciting new plans! Are you interested in internet, TV, or mobile services?",
            "technical issue": "I understand this is frustrating. Can you describe the problem in more detail?",
            "account details": "I can help with that. For security, please verify your name and date of birth.",
            "router not working": "Is the power light on? Please check all cable connections."
        }

    def generate_response(self, prompt, max_new_tokens=100):
        """
        Generates a response based on the prompt, simulating initial BC-trained behavior.
        In a real scenario, this would be an actual LLM inference.
        """
        time.sleep(0.1) # Simulate processing time
        prompt_lower = prompt.lower()
        for key, response in self.knowledge_base.items():
            if key in prompt_lower:
                return response
        return "I'm sorry, I'm not sure how to respond to that. Could you please rephrase or provide more details?"

# --- 2. Reward Model (Simulated) ---
class RewardModel:
    """
    Simulates a Reward Model trained on human preferences for agent responses.
    Higher scores indicate higher human preference.
    """
    def __init__(self, model_name="preference_predictor_v1"):
        self.model_name = model_name

    def predict_score(self, conversation_context, agent_response):
        """
        Predicts a preference score for an agent's response given the conversation context.
        In a real system, this would be a neural network.
        Here, it's a heuristic based on keywords and length.
        """
        score = 0.5 # Base score

        # Heuristics for a telecommunications customer support agent
        if "sorry" in agent_response.lower() or "apologize" in agent_response.lower():
            score += 0.1 # Empathy often preferred
        if "solution" in agent_response.lower() or "resolve" in agent_response.lower():
            score += 0.15 # Problem-solving focus
        if "restart" in agent_response.lower() or "troubleshoot" in agent_response.lower():
            score += 0.05 # Actionable advice
        if "account number" in agent_response.lower() or "verify" in agent_response.lower():
            score -= 0.05 # Can be perceived as intrusive if asked too early or without context
        if "not sure" in agent_response.lower():
            score -= 0.2 # Lack of confidence
        if len(agent_response) < 20:
            score -= 0.05 # Too short might not be helpful
        if "please" in agent_response.lower():
            score += 0.02 # Politeness

        # Simulate context dependency (very basic)
        if "internet slow" in conversation_context and "router" in agent_response.lower():
            score += 0.1
        if "bill inquiry" in conversation_context and "account number" in agent_response.lower():
            score += 0.08 # Contextually appropriate

        return min(1.0, max(0.0, score + random.uniform(-0.1, 0.1))) # Add some noise

# --- 3. Advanced AI Customer Support Agent (Integrating RLHF/Rejection Sampling) ---
class AdvancedCustomerSupportAgent:
    """
    An advanced AI agent integrating Behavior Cloning, Reward Model,
    and Rejection Sampling for improved human alignment and sample-efficient RL strategies.
    """
    def __init__(self):
        self.initial_llm = BehaviorClonedLLM()
        self.reward_model = RewardModel()
        self.conversation_history = []
        self.reference_solutions = {
            "internet_slow_diagnosis": [
                "1. Restart your router and modem.",
                "2. Check all cable connections for looseness.",
                "3. Run a speed test at speedtest.net to confirm the issue.",
                "4. If steps 1-3 don't work, we can run a remote diagnostic or schedule a technician."
            ],
            "bill_dispute_process": [
                "1. Verify your account details for security.",
                "2. Review your billing history for any unusual charges.",
                "3. If a discrepancy is found, we can escalate this to our billing specialist team."
            ]
        }

    def _generate_candidate_responses(self, prompt, num_candidates=3):
        """Generates multiple candidate responses for a given prompt using the initial LLM."""
        candidates = []
        for _ in range(num_candidates):
            response = self.initial_llm.generate_response(prompt)
            # Simple perturbation for diverse candidates in simulation
            if _ == 1 and "not sure" in response: 
                 response = "I'm still learning. Could you provide more specific details or examples?"
            elif _ == 2 and "not sure" in response:
                 response = "Let me try to rephrase your request. Are you asking about...?"
            elif _ > 0 and len(response) > 50: # Slightly vary longer responses
                response = response.replace("If the issue persists", "Should it still be an issue")
            candidates.append(response)
        return candidates

    def _select_best_response_rejection_sampling(self, prompt, candidates):
        """
        Uses the Reward Model to select the best response from candidates
        (Rejection Sampling / Best-of-N).
        """
        scored_candidates = []
        # Formulate context for reward model prediction
        context = " ".join(self.conversation_history + [f"User: {prompt}"])
        for candidate in candidates:
            score = self.reward_model.predict_score(context, candidate)
            scored_candidates.append((score, candidate))
        
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_response = scored_candidates[0]
        print(f"DEBUG: Scored candidates: {[(round(s, 2), r[:50] + '...') if len(r) > 50 else (round(s, 2), r) for s, r in scored_candidates]}")
        return best_response, best_score

    def handle_inquiry(self, user_inquiry, num_candidates=3):
        """
        Handles a customer inquiry, applying the full advanced agent pipeline:
        Behavior Cloning for initial response generation,
        Reward Model and Rejection Sampling for human alignment,
        and sample-efficient strategies for multi-stage tasks.
        """
        self.conversation_history.append(f"User: {user_inquiry}")
        print(f"\nUser: {user_inquiry}")

        # --- Apply Sample-Efficient RL Strategy (Reference Reuse for Multi-stage Tasks) ---
        # This simulates detecting a multi-stage task (e.g., complex troubleshooting)
        # and injecting a pre-optimized, human-preferred sequence of steps.
        # In a full RL setup, this would involve fine-tuning a sub-policy on this specific task.
        current_conversation_str = " ".join(self.conversation_history).lower()
        if "internet slow" in current_conversation_str and len(self.conversation_history) > 3 and \
           any(step in current_conversation_str for step in ["tried restarting", "still slow", "next steps"]):
            print("\n--- Agent Action: Applying sample-efficient strategy for 'internet_slow_diagnosis' ---")
            agent_response = "I understand the internet issue is persisting. Let's follow a structured diagnostic plan:\n"
            for step in self.reference_solutions["internet_slow_diagnosis"]:
                agent_response += f"- {step}\n"
            agent_response += "Please go through these steps and let me know the outcome." 
            self.conversation_history.append(f"Agent: {agent_response}")
            print(f"Agent: {agent_response}")
            return agent_response

        # --- Behavior Cloning and Rejection Sampling for general inquiries ---
        candidate_responses = self._generate_candidate_responses(user_inquiry, num_candidates)
        final_response, score = self._select_best_response_rejection_sampling(
            user_inquiry, candidate_responses
        )

        self.conversation_history.append(f"Agent: {final_response}")
        print(f"Agent (Preference Score: {score:.2f}): {final_response}")
        return final_response

# --- Example Usage ---
if __name__ == "__main__":
    agent = AdvancedCustomerSupportAgent()

    print("--- Scenario 1: Basic Inquiry with Behavior Cloning & Rejection Sampling ---")
    agent.handle_inquiry("My internet is really slow today.")
    agent.handle_inquiry("I need to check my last bill, it seems higher than usual.")

    print("\n--- Scenario 2: Multi-stage Task with Sample-Efficient Reference Reuse ---")
    agent.handle_inquiry("Hi, my internet connection is terrible. I can't even stream videos.")
    agent.handle_inquiry("I've already tried restarting the router and it didn't help. It's still super slow.") 
    agent.handle_inquiry("What are the absolute next steps for troubleshooting my slow internet? I need a clear plan.") # This should trigger the reference reuse
    agent.handle_inquiry("Thank you for the detailed steps, I will try those now.")

    print("\n--- Scenario 3: Another General Inquiry ---")
    agent.handle_inquiry("I also want to upgrade my mobile plan. What options are available?")

    print("\n--- End of Simulation ---")
