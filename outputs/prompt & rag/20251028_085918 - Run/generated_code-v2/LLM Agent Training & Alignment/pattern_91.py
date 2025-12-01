
class BehaviorCloningModel:
    def __init__(self):
        self.trained_policy = []

    def train(self, demonstrations):
        for demo in demonstrations:
            self.trained_policy.append({
                "query": demo["query"],
                "actions": demo["actions"]
            })

    def predict_action(self, query, observation=None):
        for policy_entry in self.trained_policy:
            if query in policy_entry["query"]:
                return policy_entry["actions"][0] if policy_entry["actions"] else "No action"
        return "Search knowledge base"


class RewardModel:
    def __init__(self):
        self.preferences = []

    def train(self, comparisons):
        for comp in comparisons:
            self.preferences.append({
                "query": comp["query"],
                "response_A": comp["response_A"],
                "response_B": comp["response_B"],
                "preferred_response": comp["preferred_response"]
            })

    def predict_reward(self, query, response):
        for pref_entry in self.preferences:
            if query in pref_entry["query"]:
                if pref_entry["preferred_response"] == "A" and response == pref_entry["response_A"]:
                    return 1.0
                elif pref_entry["preferred_response"] == "B" and response == pref_entry["response_B"]:
                    return 1.0
        return 0.5


def collect_demonstrations():
    demonstrations = [
        {
            "query": "My internet is not working.",
            "actions": ["Run diagnostic tool", "Check router status", "Reset modem"],
            "observations": ["Diagnostic tool reports no issues", "Router lights are off", "Modem rebooting"]
        },
        {
            "query": "I want to change my subscription plan.",
            "actions": ["Access customer account", "Display available plans", "Confirm plan change"],
            "observations": ["Account details loaded", "Plans displayed", "Confirmation pending"]
        }
    ]
    return demonstrations


def collect_comparisons():
    comparisons = [
        {
            "query": "How do I reset my password?",
            "response_A": "Go to settings, click security, then 'Reset Password'.",
            "response_B": "Navigate to your profile, select 'Account Security', and choose the 'Forgot Password' option to receive a reset link.",
            "preferred_response": "B"
        },
        {
            "query": "What are your operating hours?",
            "response_A": "Mon-Fri, 9 AM - 5 PM. Sat, 10 AM - 2 PM. Closed Sundays.",
            "response_B": "Our support team is available from 9 AM to 5 PM GMT on weekdays, and from 10 AM to 2 PM GMT on Saturdays. We are closed on Sundays and public holidays.",
            "preferred_response": "B"
        }
    ]
    return comparisons


class CustomerSupportAgent:
    def __init__(self, bc_model: BehaviorCloningModel, rm_model: RewardModel):
        self.bc_model = bc_model
        self.rm_model = rm_model
        self.context = []

    def _interact_with_system(self, action: str) -> str:
        if "search knowledge base" in action.lower():
            return "Observation: Found relevant articles on password reset and subscription plans."
        elif "run diagnostic tool" in action.lower():
            return "Observation: Network diagnostics initiated. No immediate issues detected."
        elif "access customer account" in action.lower():
            return "Observation: Customer account details loaded successfully."
        else:
            return f"Observation: System executed '{action}'. Generic response for now."

    def _generate_response(self, query: str, actions_taken: list, observations_made: list) -> str:
        combined_info = f"Based on your query '{query}' and actions taken: {', '.join(actions_taken)} and observations: {', '.join(observations_made)}. "
        
        if "password reset" in query.lower():
            return combined_info + "Please navigate to your account security settings and use the 'Forgot Password' option."
        elif "subscription plan" in query.lower():
            return combined_info + "I've accessed your account. We have several plans available. Which one would you like to know more about?"
        elif "internet not working" in query.lower():
            return combined_info + "I've run diagnostics. Can you please try rebooting your modem?"
        else:
            return combined_info + "I understand your request and am working to find the best solution for you."

    def resolve_issue(self, customer_query: str) -> str:
        self.context.append({"role": "customer", "text": customer_query})

        current_actions = []
        current_observations = []
        max_interaction_steps = 3
        
        for step in range(max_interaction_steps):
            action = self.bc_model.predict_action(customer_query, current_observations)
            current_actions.append(action)
            
            observation = self._interact_with_system(action)
            current_observations.append(observation)
            self.context.append({"role": "agent_action", "text": action})
            self.context.append({"role": "system_observation", "text": observation})
            
            if "No action" in action or "Confirmation pending" in observation:
                break
        
        final_response = self._generate_response(customer_query, current_actions, current_observations)
        self.context.append({"role": "agent_response", "text": final_response})
        
        reward_score = self.rm_model.predict_reward(customer_query, final_response)
        
        return final_response

if __name__ == "__main__":
    expert_demonstrations = collect_demonstrations()
    human_comparisons = collect_comparisons()

    bc_model = BehaviorCloningModel()
    bc_model.train(expert_demonstrations)

    rm_model = RewardModel()
    rm_model.train(human_comparisons)

    agent = CustomerSupportAgent(bc_model, rm_model)

    print("--- Simulating Customer Interactions ---")
    agent.resolve_issue("My internet is totally down, what should I do?")
    agent.resolve_issue("I want to upgrade my current subscription plan to a premium one.")
    agent.resolve_issue("I forgot my password, how can I reset it?")
