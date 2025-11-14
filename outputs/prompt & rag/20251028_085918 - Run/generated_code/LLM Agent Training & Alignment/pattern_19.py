import random

class Agent:
    """Represents the core language model for the customer support agent."""
    def __init__(self, model_name="BaseLLM"):
        self.model_name = model_name
        self.knowledge_base = {}
        print(f"Agent initialized with model: {self.model_name}")

    def generate_response(self, query: str, num_candidates: int = 1) -> list[str]:
        """Generates multiple candidate responses for a given query."""
        print(f"Agent generating {num_candidates} responses for: '{query}'")
        responses = []
        for i in range(num_candidates):
            # Simulate response generation - in a real scenario, this would be an LLM API call
            if query in self.knowledge_base:
                response = f"{self.knowledge_base[query]} (Candidate {i+1})"
            else:
                response = f"I'm sorry, I need more information about '{query}' to provide a precise answer. (Candidate {i+1})"
            responses.append(response)
        return responses

    def train_behavior_cloning(self, demonstrations: list[dict]):
        """Simulates training the agent's initial skills from human demonstrations."""
        print("\n--- Training Agent with Behavior Cloning ---")
        for demo in demonstrations:
            query = demo["query"]
            response = demo["response"]
            self.knowledge_base[query] = response # Simple update for demonstration
            print(f"Learned: '{query}' -> '{response}'")
        print("Behavior Cloning training complete.")

    def fine_tune_rlhf(self, rlhf_data: list[dict]):
        """Placeholder for fine-tuning the agent using RLHF."""
        print("\n--- Fine-tuning Agent with RLHF ---")
        # In a real system, this would involve using a library like TRL with a reward model
        print(f"Simulating RLHF fine-tuning on {len(rlhf_data)} data points...")
        print("RLHF fine-tuning complete (conceptual).")


class RewardModel:
    """Evaluates the quality of agent responses based on human preferences."""
    def __init__(self, model_name="PreferenceClassifier"):
        self.model_name = model_name
        self.preferences = []
        print(f"Reward Model initialized with model: {self.model_name}")

    def predict_reward(self, query: str, response: str) -> float:
        """Predicts a reward score for a given query-response pair."""
        # Simulate reward prediction based on some heuristic or a trained model
        # Higher score means better response
        if "I'm sorry" in response:
            return random.uniform(0.1, 0.4) # Lower reward for apologies/lack of info
        elif "personalized" in response or "troubleshoot" in response:
            return random.uniform(0.7, 0.99) # Higher reward for relevant keywords
        else:
            return random.uniform(0.4, 0.8)

    def train_from_preferences(self, preference_data: list[dict]):
        """Simulates training the reward model from human preference comparisons."""
        print("\n--- Training Reward Model from Human Preferences ---")
        self.preferences.extend(preference_data)
        # In a real system, this would involve training a classification/ranking model
        # e.g., using a dataset of (query, response_A, response_B, chosen_response)
        print(f"Simulating reward model training on {len(preference_data)} new preferences...")
        print("Reward Model training complete (conceptual).")


class DataCollector:
    """Manages the collection of demonstration and preference data."""
    def __init__(self):
        self.demonstrations = []
        self.preferences = []
        print("Data Collector initialized.")

    def add_demonstration(self, query: str, human_response: str):
        """Adds a human demonstration (query, ideal response)."""
        self.demonstrations.append({"query": query, "response": human_response})
        print(f"Collected demonstration: '{query}' -> '{human_response}'")

    def add_preference(self, query: str, response_a: str, response_b: str, chosen_response: str):
        """Adds a human preference (which response was better)."""
        self.preferences.append({"query": query, "response_a": response_a, "response_b": response_b, "chosen": chosen_response})
        print(f"Collected preference for '{query}': '{chosen_response}' was better.")


class Trainer:
    """Orchestrates the training processes for the agent and reward model."""
    def __init__(self, agent: Agent, reward_model: RewardModel, data_collector: DataCollector):
        self.agent = agent
        self.reward_model = reward_model
        self.data_collector = data_collector
        print("Trainer initialized.")

    def run_initial_training(self):
        """Performs initial skill acquisition via behavior cloning."""
        self.agent.train_behavior_cloning(self.data_collector.demonstrations)

    def optimize_with_human_feedback(self):
        """Optimizes the system using human feedback (reward modeling and RLHF)."""
        self.reward_model.train_from_preferences(self.data_collector.preferences)

        # For RLHF, we'd typically generate new data using the current agent,
        # get rewards from the reward model, and then fine-tune the agent.
        # This is a conceptual representation.
        rlhf_training_data = []
        for pref in self.data_collector.preferences:
            # Simulate converting preference into a format for RLHF training
            # e.g., agent generates responses, reward model assigns scores
            # then policy is updated based on these scores.
            rlhf_training_data.append({"query": pref["query"], "good_response": pref["chosen"], "bad_response": pref["response_a"] if pref["chosen"] != pref["response_a"] else pref["response_b"]})

        self.agent.fine_tune_rlhf(rlhf_training_data)


class CustomerSupportSystem:
    """Simulates the customer support environment, including interaction and rejection sampling."""
    def __init__(self, agent: Agent, reward_model: RewardModel):
        self.agent = agent
        self.reward_model = reward_model
        print("Customer Support System initialized.")

    def handle_customer_query(self, query: str, num_candidates: int = 3) -> str:
        """Handles a customer query, using rejection sampling to pick the best response."""
        print(f"\n--- Customer Query: '{query}' ---")
        candidate_responses = self.agent.generate_response(query, num_candidates)

        if not candidate_responses:
            return "No response could be generated."

        best_response = None
        highest_reward = -float('inf')

        print("Evaluating candidate responses with Reward Model:")
        for i, response in enumerate(candidate_responses):
            reward = self.reward_model.predict_reward(query, response)
            print(f"  Candidate {i+1}: '{response}' (Reward: {reward:.2f})")
            if reward > highest_reward:
                highest_reward = reward
                best_response = response

        print(f"Selected Best Response: '{best_response}' (Reward: {highest_reward:.2f})")
        return best_response


# --- Main Execution --- #
if __name__ == "__main__":
    # 1. Initialize components
    agent = Agent()
    reward_model = RewardModel()
    data_collector = DataCollector()
    trainer = Trainer(agent, reward_model, data_collector)
    customer_system = CustomerSupportSystem(agent, reward_model)

    # 2. Simulate Dual Data Collection
    print("\n===== Simulating Dual Data Collection =====")
    data_collector.add_demonstration(
        "How do I reset my password?",
        "You can reset your password by going to the 'Forgot Password' link on the login page and following the instructions."
    )
    data_collector.add_demonstration(
        "My internet is not working.",
        "Please try restarting your router and modem. If the issue persists, contact our technical support hotline."
    )

    agent_response_a = "Please restart your device to fix the internet issue."
    agent_response_b = "Try restarting your router and modem. This often resolves connectivity problems."
    data_collector.add_preference(
        "My internet is not working.",
        agent_response_a,
        agent_response_b,
        agent_response_b # Human prefers B
    )

    agent_response_c = "I cannot log in."
    agent_response_d = "To log in, please enter your username and password. If you forgot your password, click the 'Forgot Password' link."
    data_collector.add_preference(
        "I cannot log in.",
        agent_response_c,
        agent_response_d,
        agent_response_d # Human prefers D
    )

    # 3. Run Initial Training (Behavior Cloning)
    trainer.run_initial_training()

    # 4. Handle a query before optimization
    print("\n===== Handling Query BEFORE Optimization =====")
    customer_system.handle_customer_query("How do I reset my password?")
    customer_system.handle_customer_query("My printer is not working.")

    # 5. Optimize with Human Feedback (Reward Modeling & RLHF)
    trainer.optimize_with_human_feedback()

    # 6. Handle a query AFTER optimization (demonstrating improved behavior/rejection sampling)
    print("\n===== Handling Query AFTER Optimization =====")
    customer_system.handle_customer_query("My internet is not working.")
    customer_system.handle_customer_query("I cannot log in.")
    customer_system.handle_customer_query("Can you help me troubleshoot my software installation?", num_candidates=5)

    print("\n--- Simulation Complete ---")
