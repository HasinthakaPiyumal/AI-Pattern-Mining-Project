import random
import collections

class CustomerSupportEnv:
    def __init__(self, knowledge_base_docs=None):
        self.knowledge_base = knowledge_base_docs if knowledge_base_docs else [
            "How to reset your password: Go to settings, click security, then reset password.",
            "Contact support for billing issues at support@example.com.",
            "Our product features include real-time chat and document sharing.",
            "Troubleshooting guide for login problems: Check internet connection, clear cache.",
            "To update your profile, navigate to your dashboard and select 'Edit Profile'."
        ]
        self.current_query = None
        self.collected_references = []
        self.partial_answer = []
        self.done = False
        self.max_answer_length = 5 # Simplified for demo
        self.step_count = 0

    def reset(self, query):
        self.current_query = query
        self.collected_references = []
        self.partial_answer = []
        self.done = False
        self.step_count = 0
        return self.get_state()

    def step(self, action, mode="full"):
        reward = 0
        self.step_count += 1
        
        # Action can be a browsing query or an answer token
        if mode == "full":
            if action.startswith("BROWSE:"):
                search_term = action.split(":")[1].strip().lower()
                found_refs = [doc for doc in self.knowledge_base if search_term in doc.lower()]
                self.collected_references.extend(found_refs)
                reward = 0.1 # Small reward for finding references
            elif action.startswith("ANSWER:"):
                answer_token = action.split(":")[1].strip()
                self.partial_answer.append(answer_token)
                reward = 0.5 # Reward for generating answer tokens
        elif mode == "answering_only":
            if action.startswith("ANSWER:"):
                answer_token = action.split(":")[1].strip()
                self.partial_answer.append(answer_token)
                reward = 1.0 # Higher reward for answer generation in this mode
            else:
                # Invalid action for answering_only mode
                reward = -0.5

        self.done = len(self.partial_answer) >= self.max_answer_length or self.step_count >= 10 # Simplified done condition

        if self.done and action.startswith("ANSWER:") and "good" in " ".join(self.partial_answer).lower():
             reward += 10 # Bonus for a 'good' answer

        return self.get_state(), reward, self.done, {}

    def get_state(self):
        return {
            "query": self.current_query,
            "references": list(set(self.collected_references)), # Unique references
            "partial_answer": " ".join(self.partial_answer),
            "mode": "full" if self.collected_references else "answering_only" # Indicative mode
        }

    def get_reward(self, generated_answer):
        # Placeholder for a more sophisticated reward function
        # In a real scenario, this would compare generated_answer to ground truth or use a model
        if "password" in self.current_query.lower() and "reset password" in generated_answer.lower():
            return 5
        elif "billing" in self.current_query.lower() and "support@example.com" in generated_answer.lower():
            return 5
        else:
            return -1 # Penalty for irrelevant answers

    def get_collected_references(self):
        return list(set(self.collected_references))

    def set_current_references(self, references):
        self.collected_references = list(set(references))
        self.partial_answer = [] # Reset partial answer for new answering_only episode
        self.done = False
        self.step_count = 0

    def is_done(self):
        return self.done

class RLAgent:
    def __init__(self):
        # Placeholder for actual neural networks (browsing and answer generation policies)
        # In a real application, these would be transformer models from `transformers` library
        # trained with `trl` or custom PPO/A2C with `torch`/`tensorflow`.
        self.browsing_policy = self._mock_policy
        self.answer_generation_policy = self._mock_policy

    def _mock_policy(self, state, available_actions):
        # A very simple mock policy: pick a random action
        if not available_actions:
            return None
        return random.choice(available_actions)

    def choose_action(self, state, mode="full"):
        query = state["query"]
        references = state["references"]
        partial_answer = state["partial_answer"]

        if mode == "full":
            if len(references) < 2 and random.random() < 0.7: # Prioritize browsing initially
                # Mock browsing actions
                possible_browsing_actions = [
                    f"BROWSE: {query.split()[0]}",
                    f"BROWSE: {random.choice(['password', 'billing', 'troubleshooting', 'features', 'profile'])}"
                ]
                return self.browsing_policy(state, possible_browsing_actions)
            else:
                # Mock answer generation actions
                possible_answer_tokens = [f"ANSWER: {token}" for token in ["The", "solution", "is", "here", "good"]] # Simplified tokens
                return self.answer_generation_policy(state, possible_answer_tokens)
        elif mode == "answering_only":
            # Only answer generation actions
            possible_answer_tokens = [f"ANSWER: {token}" for token in ["The", "solution", "is", "here", "good"]] # Simplified tokens
            return self.answer_generation_policy(state, possible_answer_tokens)
        return None

    def learn(self, experiences):
        # Placeholder for actual RL learning algorithm (e.g., PPO, A2C)
        # This would involve gradient updates to policy and value networks.
        # print(f"Agent learning from {len(experiences)} experiences...")
        pass # Actual learning logic would go here

Experience = collections.namedtuple("Experience", ["state", "action", "reward", "next_state", "done"])

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return []
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

def train_agent(num_episodes, n_answering_only_episodes, replay_buffer_capacity=10000, batch_size=32):
    env = CustomerSupportEnv()
    agent = RLAgent()
    replay_buffer = ReplayBuffer(replay_buffer_capacity)

    customer_queries = [
        "How can I reset my password?",
        "I have a question about my billing, who do I contact?",
        "What are the main features of your product?",
        "I'm having trouble logging in."
    ]

    print("Starting RL training with Reference Reuse...")

    for episode in range(num_episodes):
        current_query = random.choice(customer_queries)
        print(f"\n--- Episode {episode + 1}/{num_episodes} (Query: '{current_query}') ---")

        # --- Run a FULL episode (browsing + answering) ---
        state = env.reset(current_query)
        done_full_episode = False
        full_episode_rewards = []
        steps_full = 0
        
        print("Running FULL episode...")
        while not done_full_episode:
            action = agent.choose_action(state, "full")
            next_state, reward, done_full_episode, _ = env.step(action, "full")
            replay_buffer.add(Experience(state, action, reward, next_state, done_full_episode))
            state = next_state
            full_episode_rewards.append(reward)
            steps_full += 1
            if steps_full > 15: # Prevent infinite loops in conceptual env
                done_full_episode = True
        
        final_answer_full = env.get_state()["partial_answer"]
        final_reward_full = env.get_reward(final_answer_full) + sum(full_episode_rewards)
        print(f"FULL Episode complete. Total reward: {final_reward_full:.2f}, Answer: '{final_answer_full}'")
        collected_references_from_full_episode = env.get_collected_references()
        print(f"References collected: {collected_references_from_full_episode}")

        # --- Run N 'answering_only' episodes (Reference Reuse) ---
        if collected_references_from_full_episode:
            print(f"Running {n_answering_only_episodes} ANSWERING-ONLY episodes with collected references...")
            for i in range(n_answering_only_episodes):
                env.set_current_references(collected_references_from_full_episode)
                state_answering_only = env.reset(current_query) # Reset env state, keep references
                done_answering_only_episode = False
                answering_only_rewards = []
                steps_answering = 0
                
                while not done_answering_only_episode:
                    action = agent.choose_action(state_answering_only, "answering_only")
                    next_state_answering_only, reward, done_answering_only_episode, _ = env.step(action, "answering_only")
                    replay_buffer.add(Experience(state_answering_only, action, reward, next_state_answering_only, done_answering_only_episode))
                    state_answering_only = next_state_answering_only
                    answering_only_rewards.append(reward)
                    steps_answering += 1
                    if steps_answering > 10: # Prevent infinite loops in conceptual env
                        done_answering_only_episode = True

                final_answer_ao = env.get_state()["partial_answer"]
                final_reward_ao = env.get_reward(final_answer_ao) + sum(answering_only_rewards)
                print(f"  -> AO Episode {i+1}: Total reward: {final_reward_ao:.2f}, Answer: '{final_answer_ao}'")

        # --- Learning Step --- 
        if len(replay_buffer) >= batch_size:
            batch = replay_buffer.sample(batch_size)
            agent.learn(batch)
            print(f"Agent learned from a batch of {len(batch)} experiences.")
        else:
            print("Replay buffer not full enough for learning yet.")

    print("\nRL Training complete.")

if __name__ == "__main__":
    # Example usage
    train_agent(num_episodes=10, n_answering_only_episodes=3)
