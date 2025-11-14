
import random
import time

# --- Mocking external libraries for demonstration purposes ---
# In a real application, you would use actual libraries like transformers, trl, torch/tensorflow.
class MockLLM:
    def __init__(self, name="Mock-LLM"):
        self.name = name
        self.knowledge_base = [
            "I can help you with order status.",
            "Please provide your order number for assistance.",
            "Our return policy allows returns within 30 days.",
            "For technical support, please visit our FAQ page.",
            "We offer free shipping on orders over $50."
        ]
        self.finetuned_knowledge = []

    def generate(self, prompt, num_return_sequences=1):
        print(f"[{self.name}] Generating {num_return_sequences} responses for: '{prompt}'")
        responses = []
        for _ in range(num_return_sequences):
            # Simulate varied responses, sometimes using finetuned knowledge
            if self.finetuned_knowledge and random.random() < 0.7:
                response = random.choice(self.finetuned_knowledge)
            else:
                response = random.choice(self.knowledge_base)
            responses.append(f"LLM Response: {response} (prompt: {prompt[:30]}...)")
            time.sleep(0.05) # Simulate generation time
        return responses

    def fine_tune(self, data):
        print(f"[{self.name}] Fine-tuning with {len(data)} data points...")
        for item in data:
            # In a real scenario, this would update model weights.
            # Here, we simulate by adding to 'finetuned_knowledge'
            if isinstance(item, tuple) and len(item) == 2: # Assuming (query, response) for BC
                self.finetuned_knowledge.append(item[1])
            elif isinstance(item, str):
                self.finetuned_knowledge.append(item)
        print(f"[{self.name}] Fine-tuning complete. New knowledge acquired.")

# --- 1. Dual Data Collection for Agentic LLM Training ---
class DataCollector:
    def __init__(self):
        self.demonstrations = []  # Human demonstrations (query, ideal_response)
        self.preferences = []     # Human preferences (query, response_A, response_B, preferred_index)

    def collect_demonstration(self, query: str, human_response: str):
        """Collects a human demonstration of an ideal response."""
        self.demonstrations.append((query, human_response))
        print(f"[DataCollector] Collected demonstration: '{query}' -> '{human_response}'")

    def collect_preference(self, query: str, response_A: str, response_B: str, preferred_index: int):
        """Collects human feedback on preferred responses (0 for A, 1 for B)."""
        self.preferences.append((query, response_A, response_B, preferred_index))
        print(f"[DataCollector] Collected preference for '{query}': A='{response_A}', B='{response_B}', Preferred={'A' if preferred_index == 0 else 'B'}")

    def get_demonstrations(self):
        return self.demonstrations

    def get_preferences(self):
        return self.preferences

# --- 2. Human Feedback for Quality Optimization (Reward Modeling & RLHF) ---
class RewardModel:
    def __init__(self):
        # In a real scenario, this would be a neural network trained to predict human preference.
        # Here, we use a simple heuristic for demonstration.
        self.heuristic_keywords = {"helpful": 1.5, "sorry": -0.5, "thank you": 0.8, "policy": 1.2, "order": 1.0}

    def predict_score(self, query: str, response: str) -> float:
        """Predicts a 'reward' score for a given response based on its quality/alignment."""
        score = 0.0
        response_lower = response.lower()
        for keyword, value in self.heuristic_keywords.items():
            if keyword in response_lower:
                score += value
        # Add a bonus for length, as longer helpful responses might be better
        score += len(response) / 100.0
        print(f"[RewardModel] Scored response '{response[:50]}...' for query '{query[:30]}...': {score:.2f}")
        return score

    def train(self, preference_data: list):
        """Simulates training the reward model using human preference data."""
        print(f"[RewardModel] Training with {len(preference_data)} preference data points...")
        # In a real system, this would involve updating the model's weights
        # based on which response was preferred by humans.
        for query, res_a, res_b, preferred_idx in preference_data:
            score_a = self.predict_score(query, res_a)
            score_b = self.predict_score(query, res_b)
            # Adjust internal 'weights' or heuristics based on preference
            if (preferred_idx == 0 and score_a < score_b) or \
               (preferred_idx == 1 and score_b < score_a):
                print(f"  [RewardModel] Adjusting for preference. Preferred: {('A' if preferred_idx == 0 else 'B')}")
        print("[RewardModel] Training complete.")


# --- 3. RLHF Agent (incorporating Behavior Cloning, Rejection Sampling, Sample-Efficient RL) ---
class RLHFAgent:
    def __init__(self, base_llm: MockLLM, reward_model: RewardModel):
        self.llm = base_llm
        self.reward_model = reward_model

    def perform_behavior_cloning(self, demonstrations: list):
        """Trains the LLM using human demonstrations (Behavior Cloning)."""
        print("\n--- Performing Behavior Cloning (Initial Skill Acquisition) ---")
        bc_data = []
        for query, human_response in demonstrations:
            bc_data.append((query, human_response))
        self.llm.fine_tune(bc_data)
        print("Behavior Cloning complete.")

    def rejection_sample(self, query: str, candidate_responses: list) -> str:
        """Selects the best response from candidates using the Reward Model (Rejection Sampling)."""
        print(f"\n--- Performing Rejection Sampling (Best-of-N) for query: '{query[:30]}...' ---")
        if not candidate_responses:
            return "I am sorry, I couldn't generate a response."

        scores = []
        for i, response in enumerate(candidate_responses):
            score = self.reward_model.predict_score(query, response)
            scores.append((score, response))
        
        best_response = max(scores, key=lambda x: x[0])[1]
        print(f"  Selected best response (score {max(scores, key=lambda x: x[0])[0]:.2f}): '{best_response[:50]}...'\n")
        return best_response

    def perform_rlhf_training_step(self, query: str, context: str, human_preferred_response: str):
        """Simulates a single RLHF training step for Sample-Efficient RL."""
        print(f"\n--- Performing RLHF Training Step for query: '{query[:30]}...' ---")
        # 1. Generate responses from current LLM
        current_responses = self.llm.generate(context, num_return_sequences=3)

        # 2. Get rewards for generated responses from Reward Model
        response_scores = [(self.reward_model.predict_score(query, res), res) for res in current_responses]
        
        # Simulate comparing against a 'human preferred' response to derive a policy gradient signal
        # In a real scenario, this involves PPO/similar algorithms and actual preference data (not single preferred response)
        print(f"  Generated responses and scores: {[(round(s, 2), r[:30]) for s, r in response_scores]}")
        
        # Here, we simplify: if the LLM's best response isn't close to human preference, we 'fine-tune' it.
        best_generated_response = max(response_scores, key=lambda x: x[0])[1]
        
        if self.reward_model.predict_score(query, human_preferred_response) > self.reward_model.predict_score(query, best_generated_response) + 0.5: # Arbitrary threshold
            print("  LLM's best response significantly worse than human preference. Simulating policy update...")
            # This is where the LLM would be updated based on the reward signal
            # For demonstration, we simply add the preferred response to its finetuning knowledge
            self.llm.fine_tune([(query, human_preferred_response)])
        else:
            print("  LLM's best response is adequate or better. No significant policy update needed.")
        print("RLHF Training Step complete.")


# --- Main Customer Support Agent Orchestrator --- 
class CustomerSupportAgent:
    def __init__(self):
        self.llm = MockLLM("CustomerServiceLLM")
        self.data_collector = DataCollector()
        self.reward_model = RewardModel()
        self.rlhf_agent = RLHFAgent(self.llm, self.reward_model)

    def run_initial_setup_and_training(self):
        print("\n===== Initial Setup and Training Phase =====")

        # Simulate Dual Data Collection
        print("\n--- Simulating Dual Data Collection ---")
        self.data_collector.collect_demonstration(
            "Where is my order?", "Please provide your order number and I can check its status for you."
        )
        self.data_collector.collect_demonstration(
            "How do I return an item?", "You can initiate a return from your order history within 30 days of purchase."
        )
        self.data_collector.collect_preference(
            "What's your return policy?",
            "Our return policy allows returns within 30 days.",
            "You can return items if they are unopened.",
            0 # Prefers A
        )
        self.data_collector.collect_preference(
            "Do you have free shipping?",
            "We offer free shipping on all orders.",
            "Yes, for orders over $50.",
            1 # Prefers B
        )

        # 1. Behavior Cloning for Initial Skill Acquisition
        self.rlhf_agent.perform_behavior_cloning(self.data_collector.get_demonstrations())

        # 2. Train Reward Model with Human Feedback
        self.reward_model.train(self.data_collector.get_preferences())

        print("\n===== Initial Setup and Training Complete =====\n")

    def handle_customer_query(self, query: str) -> str:
        print(f"\n--- Agent handling customer query: '{query}' ---")
        # Generate multiple candidate responses
        candidate_responses = self.llm.generate(query, num_return_sequences=3)

        # 3. Rejection Sampling (Best-of-N) using the Reward Model
        best_response = self.rlhf_agent.rejection_sample(query, candidate_responses)
        
        print(f"[CustomerSupportAgent] Final Agent Response: {best_response}")
        return best_response

    def continuously_improve(self, num_iterations: int = 2):
        print("\n===== Continuous Improvement Phase (Sample-Efficient RL & RLHF) =====")
        for i in range(num_iterations):
            print(f"\n--- Improvement Iteration {i+1}/{num_iterations} ---")
            # Simulate gathering new data during operation
            new_query = f"I need help with my recent order {random.randint(1000, 9999)}."
            human_ideal_response = f"For order {new_query.split()[-1]}, please confirm your name and email. I will then assist you."
            self.data_collector.collect_demonstration(new_query, human_ideal_response)

            # Simulate a situation where RLHF is needed (e.g., LLM gave a sub-optimal response)
            llm_initial_response = self.llm.generate(new_query, num_return_sequences=1)[0]
            self.data_collector.collect_preference(
                new_query,
                llm_initial_response,
                human_ideal_response,
                1 # Human prefers the ideal response
            )

            # Retrain Reward Model with new preferences
            self.reward_model.train(self.data_collector.get_preferences())

            # Perform Sample-Efficient RLHF training step using new data and updated reward model
            # This simulates focusing training on high-impact phases where the model struggled.
            self.rlhf_agent.perform_rlhf_training_step(new_query, new_query, human_ideal_response)

        print("\n===== Continuous Improvement Complete =====\n")

# --- Main Execution --- 
if __name__ == "__main__":
    agent = CustomerSupportAgent()
    agent.run_initial_setup_and_training()

    print("\n====================== AGENT IN OPERATION ======================")
    agent.handle_customer_query("What is the status of my order 12345?")
    agent.handle_customer_query("Can I get a refund for a damaged item?")
    agent.handle_customer_query("I forgot my password, what should I do?")

    agent.continuously_improve()

    print("\n====================== AGENT AFTER IMPROVEMENT ======================")
    agent.handle_customer_query("What is the status of my order 67890?")
    agent.handle_customer_query("I want to return a product I bought last week.")
    agent.handle_customer_query("Tell me more about your privacy policy.")

