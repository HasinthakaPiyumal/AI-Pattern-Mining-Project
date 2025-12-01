import random

# 1. KnowledgeBase Simulation
class KnowledgeBase:
    """
    A simulated knowledge base for retrieving relevant information.
    In a real system, this would involve database queries or search APIs.
    """
    def __init__(self, articles):
        self.articles = articles

    def retrieve_references(self, query, num_references=2):
        """
        Simulates retrieving a fixed number of references related to a query.
        """
        # print(f"  KnowledgeBase: Retrieving references for query: '{query}'")
        relevant_refs = [
            f"Reference {i+1} for '{query}': {random.choice(self.articles)}"
            for i in range(num_references)
        ]
        return relevant_refs

# 2. Chatbot Environment
class ChatbotEnvironment:
    """
    Simulates the multi-stage customer support task environment.
    It has a 'browsing' phase and an 'answering' phase.
    """
    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base
        self.current_query = None
        self.collected_references = []
        self.state = "browsing" # Current phase of the episode
        self.step_count = 0
        self.max_phase_steps = 3 # Max steps per browsing/answering phase

    def reset(self, user_query):
        """
        Resets the environment for a new user query (a new episode).
        """
        self.current_query = user_query
        self.collected_references = []
        self.state = "browsing"
        self.step_count = 0
        print(f"\n--- New Episode for Query: '{user_query}' ---")
        return {"query": self.current_query, "state": self.state, "references": self.collected_references}

    def step(self, action, fixed_references=None):
        """
        Takes an action and updates the environment state.
        `fixed_references` is used for 'answering-only' replay episodes.
        """
        reward = 0
        done = False
        info = {"collected_references": self.collected_references[:]} # Return a copy

        if self.state == "browsing":
            if action == "browse":
                retrieved_refs = self.knowledge_base.retrieve_references(self.current_query)
                self.collected_references.extend(retrieved_refs)
                reward += 0.1 # Small positive reward for successful browsing
                print(f"  Agent browsed. Collected {len(retrieved_refs)} new refs. Total: {len(self.collected_references)}.")
                
                self.step_count += 1
                if self.step_count >= self.max_phase_steps:
                    self.state = "answering" # Transition to answering phase
                    self.step_count = 0 # Reset step count for new phase
                    print("  Browsing phase complete. Transitioning to answering.")
                # else, stay in browsing for more steps
            else:
                reward -= 0.5 # Penalty for wrong action
                print("  Agent attempted to answer during browsing phase. Penalty.")
                done = True # End episode for incorrect action
        
        elif self.state == "answering":
            if action == "answer":
                references_to_use = fixed_references if fixed_references is not None else self.collected_references
                
                if not references_to_use:
                    print("  Agent attempted to answer without references. Penalty.")
                    reward -= 1.0
                else:
                    # Simulate answer generation quality (higher is better)
                    answer_quality = random.uniform(0.5, 1.0) 
                    # Reward is significantly tied to answer quality
                    reward += answer_quality * 5.0 
                    print(f"  Agent answered using {len(references_to_use)} references. Simulated quality: {answer_quality:.2f}, Reward: {reward:.2f}.")
                done = True # Answering always ends the episode
            else:
                reward -= 0.5 # Penalty for wrong action
                print("  Agent attempted to browse during answering phase. Penalty.")
                done = True # End episode for incorrect action
        
        # If done for any reason (wrong action, answer submitted)
        return {"query": self.current_query, "state": self.state, "references": self.collected_references}, reward, done, info

# 3. Simple RL Agent (Policy Gradient inspired for illustration)
class RLAgent:
    """
    A very simplified RL agent to demonstrate policy updates.
    It has two "actions": "browse" (index 0) and "answer" (index 1).
    The policy weights represent the agent's "preference" or "skill" for these actions.
    """
    def __init__(self, num_actions=2, learning_rate=0.05):
        self.action_map = {0: "browse", 1: "answer"}
        # Initial policy weights (can be interpreted as initial skill/propensity)
        self.policy_weights = [0.5] * num_actions 
        self.learning_rate = learning_rate

    def select_action(self, current_state):
        """
        Selects an action based on the current environment state.
        For simplicity, this example forces the correct action based on the phase
        to ensure the environment flow. In a real RL agent, this would be learned.
        """
        if current_state["state"] == "browsing":
            action_idx = 0  # Force "browse"
            # print("  RLAgent: Decided to BROWSE (forced by state).")
        elif current_state["state"] == "answering":
            action_idx = 1  # Force "answer"
            # print("  RLAgent: Decided to ANSWER (forced by state).")
        else:
            # Fallback for unexpected states, should not happen in this setup
            action_idx = random.randint(0, len(self.action_map) - 1)
            print(f"  RLAgent: Decided randomly (state unknown). Action: {self.action_map[action_idx]}")

        return self.action_map[action_idx], action_idx

    def update_policy(self, episode_rewards, episode_actions_indices):
        """
        Updates the agent's policy based on the rewards received during an episode.
        This is a highly simplified policy update mechanism for demonstration.
        A positive reward for an action strengthens its policy weight.
        """
        total_reward = sum(episode_rewards)
        # print(f"  RLAgent: Updating policy with total episode reward: {total_reward:.2f}")

        # Simple update: increase the weight for actions that led to positive rewards
        # and decrease for negative rewards.
        for action_idx, reward in zip(episode_actions_indices, episode_rewards):
            self.policy_weights[action_idx] += self.learning_rate * reward
            # Clip weights to prevent them from becoming too extreme
            self.policy_weights[action_idx] = max(0.1, min(1.0, self.policy_weights[action_idx]))
        
        # Normalize weights to sum to 1 (representing probabilities for choosing actions)
        sum_weights = sum(self.policy_weights)
        self.policy_weights = [w / sum_weights for w in self.policy_weights]

        print(f"  RLAgent: Policy updated. Current weights for [browse, answer]: {self.policy_weights}")


# Main Training Loop
def main():
    """
    Main function to run the RL training with sample-efficient reference reuse.
    """
    kb_articles = [
        "Troubleshooting WiFi issues: restart router, check cables, update drivers.",
        "How to reset password: visit 'Forgot Password' link on login page, follow email instructions.",
        "Billing questions: log in to your account, navigate to 'Billing & Payments' section.",
        "Product features: consult the comprehensive user manual, typically on page 15-20.",
        "Contact support: call our 24/7 helpline at 1-800-XXX-XXXX or use live chat.",
        "Refund policy: items can be returned within 30 days with original receipt."
    ]
    knowledge_base = KnowledgeBase(kb_articles)
    env = ChatbotEnvironment(knowledge_base)
    agent = RLAgent()

    user_queries = [
        "My internet connection keeps dropping.",
        "I need help with my account password.",
        "Where can I see my latest invoice?",
        "How do I talk to a customer service representative?",
        "What is your return policy?"
    ]

    num_main_episodes = 10
    num_answer_only_replays = 5 # Number of additional "answering-only" episodes

    print("--- Starting Sample-Efficient RL Chatbot Training ---")

    for episode_num in range(num_main_episodes):
        print(f"\n===== Main Training Episode {episode_num + 1}/{num_main_episodes} =====")
        query = random.choice(user_queries)
        
        # --- Phase 1: Full Browse and Answer Episode ---
        state = env.reset(query)
        done = False
        full_episode_rewards = []
        full_episode_actions_indices = []
        collected_references_for_replay = []

        print("--- Running Full Browse & Answer Phase for initial experience ---")
        while not done:
            action, action_idx = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            full_episode_rewards.append(reward)
            full_episode_actions_indices.append(action_idx)
            state = next_state
            
            # Capture references after the browsing phase completes
            if state["state"] == "answering" and not collected_references_for_replay:
                collected_references_for_replay = info["collected_references"][:] # Deep copy
            
        print(f"  Full episode finished. Total reward: {sum(full_episode_rewards):.2f}")
        agent.update_policy(full_episode_rewards, full_episode_actions_indices)


        # --- Phase 2: Generate Additional "Answering-Only" Replay Episodes ---
        if collected_references_for_replay:
            print(f"\n--- Initiating {num_answer_only_replays} Answering-Only Replay Episodes ---")
            print("  These episodes use references collected from the preceding full episode.")
            for replay_num in range(num_answer_only_replays):
                print(f"  Replay Episode {replay_num + 1}/{num_answer_only_replays}: Focusing on answer generation.")
                
                # Reset environment for replay, forcing it into the answering state
                replay_env = ChatbotEnvironment(knowledge_base) # Use a fresh environment instance
                replay_state = replay_env.reset(query) # Use the original query
                replay_state["state"] = "answering" # Explicitly set state to answering
                
                replay_rewards = []
                replay_actions_indices = []

                # Agent performs only the "answer" action in these replays
                action, action_idx = "answer", 1 # Force answer action
                
                # Pass the *fixed* references directly to the step function
                replay_next_state, replay_reward, replay_done, replay_info = replay_env.step(action, fixed_references=collected_references_for_replay)
                
                replay_rewards.append(replay_reward)
                replay_actions_indices.append(action_idx)

                print(f"  Replay episode finished. Total reward: {sum(replay_rewards):.2f}")
                agent.update_policy(replay_rewards, replay_actions_indices)
        else:
            print("  No references collected in the full episode, skipping answer-only replays for this turn.")

    print("\n--- Training Complete ---")
    print("\nFinal Agent Policy Weights (approximately [browse_skill, answer_skill]):", [f"{w:.3f}" for w in agent.policy_weights])
    print("Higher weight suggests better performance/propensity for that action.")

if __name__ == "__main__":
    main()
