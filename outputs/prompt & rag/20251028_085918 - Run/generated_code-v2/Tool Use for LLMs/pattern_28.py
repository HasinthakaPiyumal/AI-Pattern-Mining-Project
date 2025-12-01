from rule_based_policy import RuleBasedPolicy
# from trained_policy import TrainedPolicy # To be implemented later

class CustomerSupportAgent:
    """
    Orchestrates the policy (rule-based or learned) and an LLM to respond to customer queries.
    """
    def __init__(self, rule_policy: RuleBasedPolicy, llm_pipeline=None):
        self.rule_policy = rule_policy
        self.current_policy = rule_policy # Start with the rule-based policy
        self.llm_pipeline = llm_pipeline # A placeholder for a generic LLM pipeline (e.g., from transformers)

    def set_policy(self, policy):
        """
        Switches the active policy of the agent (e.g., from rule-based to a learned one).
        """
        self.current_policy = policy

    def respond_to_query(self, query: str) -> str:
        """
        Generates a response to a customer query using the current policy and potentially the LLM.
        """
        # 1. Try to get a response from the current policy (e.g., rule-based or trained policy)
        policy_response = self.current_policy.respond(query)

        if policy_response:
            return f"[POLICY] {policy_response}"
        elif self.llm_pipeline:
            # 2. If policy doesn't have a direct answer, use the LLM to generate a response
            try:
                # A simple prompt for the LLM. In a real application, this would be more sophisticated.
                prompt = f"As a customer support agent, respond to the following: '{query}'"
                llm_output = self.llm_pipeline(prompt, max_length=150, num_return_sequences=1, early_stopping=True)
                return f"[LLM] {llm_output[0]['generated_text'].strip()}"
            except Exception as e:
                return f"[ERROR] LLM failed to generate a response: {e}"
        else:
            return "I'm sorry, I don't have enough information to respond to that query."

    def __str__(self):
        return f"CustomerSupportAgent using {type(self.current_policy).__name__} and LLM: {'Active' if self.llm_pipeline else 'Inactive'}"
