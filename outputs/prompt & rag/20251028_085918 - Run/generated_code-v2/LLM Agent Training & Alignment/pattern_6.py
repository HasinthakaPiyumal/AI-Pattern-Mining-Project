class CustomerSupportEnvironment:
    def __init__(self):
        self.knowledge_base = {
            "billing": ["invoice details", "payment methods", "subscription plans"],
            "technical": ["troubleshooting steps", "device compatibility", "software updates"],
            "account": ["password reset", "profile update", "data privacy"]
        }
        self.queries = [
            "I have a question about my latest bill.",
            "My software isn't working after the update.",
            "How can I change my password?",
            "What are my payment options?"
        ]
        self.current_query_index = 0

    def get_query(self):
        if self.current_query_index >= len(self.queries):
            self.current_query_index = 0
        query = self.queries[self.current_query_index]
        self.current_query_index += 1
        return query

    def simulate_browsing(self, query):
        relevant_references = []
        query_lower = query.lower()
        if "bill" in query_lower or "payment" in query_lower:
            relevant_references.extend(self.knowledge_base["billing"])
        if "software" in query_lower or "update" in query_lower or "working" in query_lower:
            relevant_references.extend(self.knowledge_base["technical"])
        if "password" in query_lower or "account" in query_lower:
            relevant_references.extend(self.knowledge_base["account"])
        return list(set(relevant_references))

    def get_feedback(self, response, query, references):
        reward = 0
        if "helpful" in response.lower() or "solution" in response.lower():
            reward += 1
        if any(ref.lower() in response.lower() for ref in references):
            reward += 0.5 * len(references)
        if len(response) > 20 and len(response) < 150:
            reward += 0.2
        return reward

class RLResponseGenerator:
    def __init__(self):
        self.policy_parameters = {"creativity": 0.5, "conciseness": 0.3, "accuracy_weight": 0.8}

    def browse_for_references(self, environment, query):
        return environment.simulate_browsing(query)

    def generate_response(self, query, references):
        base_response = f"Hello! Regarding your query: \"{query}\". "
        if references:
            base_response += "Based on the following information: " + ", ".join(references) + ". "
        
        response_quality = (self.policy_parameters["creativity"] + self.policy_parameters["conciseness"]) / 2
        if response_quality > 0.6:
            base_response += "We aim to provide you with the most helpful solution."
        elif response_quality > 0.3:
            base_response += "We are working to get you a precise answer."
        else:
            base_response += "We are looking into this for you."
        return base_response

    def update_response_generation_policy(self, loss):
        # Simulate a policy update based on a loss (e.g., from RL training)
        self.policy_parameters["creativity"] = max(0.1, min(1.0, self.policy_parameters["creativity"] - loss * 0.01))
        self.policy_parameters["conciseness"] = max(0.1, min(1.0, self.policy_parameters["conciseness"] - loss * 0.005))
        self.policy_parameters["accuracy_weight"] = max(0.1, min(1.0, self.policy_parameters["accuracy_weight"] + loss * 0.02))

class ReferenceStorage:
    def __init__(self):
        self.stored_references = []

    def add_references(self, references):
        self.stored_references.append(references)

    def get_references(self):
        if self.stored_references:
            return self.stored_references.pop(0) # Get and remove the oldest set of references
        return []

class RewardCalculator:
    def calculate_reward(self, response, query, references):
        reward = 0
        if any(keyword in response.lower() for keyword in query.lower().split()):
            reward += 0.5
        if any(ref.lower() in response.lower() for ref in references):
            reward += 1.0
        if len(response) > 50 and len(response) < 200:
            reward += 0.3
        return reward


# --- Training Orchestration ---
num_full_episodes = 5
num_answer_only_episodes_per_full = 3

environment = CustomerSupportEnvironment()
agent = RLResponseGenerator()
reference_storage = ReferenceStorage()
reward_calculator = RewardCalculator()

print("Starting RL training with Reference Reuse...")

for i in range(num_full_episodes):
    print(f"\n--- Full Episode {i+1} ---")
    query = environment.get_query()
    print(f"Customer Query: {query}")

    # Phase 1: Browsing
    collected_references = agent.browse_for_references(environment, query)
    reference_storage.add_references(collected_references)
    print(f"Collected References: {collected_references}")

    # Phase 2: Answer Generation (initial)
    response = agent.generate_response(query, collected_references)
    reward = reward_calculator.calculate_reward(response, query, collected_references)
    print(f"Initial Response: {response}")
    print(f"Initial Reward: {reward:.2f}")

    # Simulate RL update for the full episode
    # In a real scenario, loss would be derived from reward via RL algorithm
    mock_loss_full = (1 - reward / 3.0) # Scale reward to a loss estimate
    agent.update_response_generation_policy(mock_loss_full)
    print(f"Agent policy updated after full episode (loss: {mock_loss_full:.2f})")

    # Phase 3: Answer-Only Episodes with Reference Reuse
    for j in range(num_answer_only_episodes_per_full):
        print(f"  --- Answer-Only Sub-Episode {j+1} ---")
        reused_references = reference_storage.get_references() # Re-use the references
        if not reused_references:
            print("    No references to reuse. Skipping answer-only episode.")
            break

        # Agent focuses solely on answer generation with fixed references
        response_reused = agent.generate_response(query, reused_references)
        reward_reused = reward_calculator.calculate_reward(response_reused, query, reused_references)
        print(f"    Reused References: {reused_references}")
        print(f"    Answer-Only Response: {response_reused}")
        print(f"    Answer-Only Reward: {reward_reused:.2f}")

        # Simulate RL update for the answer-only episode
        mock_loss_answer_only = (1 - reward_reused / 3.0)
        agent.update_response_generation_policy(mock_loss_answer_only)
        print(f"    Agent policy updated after answer-only episode (loss: {mock_loss_answer_only:.2f})")

print("\nRL training with Reference Reuse completed.")
print("Final Agent Policy Parameters:", agent.policy_parameters)
