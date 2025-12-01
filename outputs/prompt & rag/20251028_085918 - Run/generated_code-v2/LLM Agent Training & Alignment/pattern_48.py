import random

class KnowledgeBase:
    def __init__(self):
        self.articles = {
            "slow_internet": "Troubleshooting steps for slow internet: restart router, check cables, contact ISP.",
            "billing_issue": "Information on how to check your bill, common billing errors, and contacting the billing department.",
            "setup_guide": "Detailed guide for setting up new devices: connection instructions, software installation.",
            "account_recovery": "Steps to recover a forgotten password or username: security questions, email verification."
        }

    def search(self, query_keywords):
        found_references = []
        for title, content in self.articles.items():
            if any(keyword in title or keyword in content for keyword in query_keywords):
                found_references.append(content)
        return found_references

class CustomerHistory:
    def __init__(self):
        self.past_interactions = {
            "customer_A": ["previous slow internet issue, resolved by router restart."],
            "customer_B": ["inquired about billing last month, provided link to FAQs."]
        }

    def get_history(self, customer_id):
        return self.past_interactions.get(customer_id, [])

class CustomerInteractionEnv:
    def __init__(self, knowledge_base, customer_history):
        self.knowledge_base = knowledge_base
        self.customer_history = customer_history
        self.current_customer_id = None
        self.customer_query = None
        self.collected_references = []
        self.browsing_done = False
        self.solution_generated = False
        self.episode_steps = 0
        self.max_browsing_steps = 5

    def reset(self, customer_id, query):
        self.current_customer_id = customer_id
        self.customer_query = query
        self.collected_references = []
        self.browsing_done = False
        self.solution_generated = False
        self.episode_steps = 0
        return self._get_state()

    def _get_state(self):
        # Simplified state representation
        return {
            "query": self.customer_query,
            "references_collected_count": len(self.collected_references),
            "browsing_done": self.browsing_done,
            "solution_generated": self.solution_generated,
            "customer_history": self.customer_history.get_history(self.current_customer_id)
        }

    def step(self, action):
        reward = 0
        done = False
        info = {}
        self.episode_steps += 1

        if action == "browse" and not self.browsing_done:
            if self.episode_steps <= self.max_browsing_steps:
                # Simulate browsing - search KB and history
                kb_refs = self.knowledge_base.search(self.customer_query.lower().split())
                hist_refs = self.customer_history.get_history(self.current_customer_id)
                self.collected_references.extend(kb_refs)
                self.collected_references.extend(hist_refs)
                self.collected_references = list(set(self.collected_references)) # Remove duplicates
                reward -= 0.1 # Small penalty for browsing time
            else:
                self.browsing_done = True # Browsing limit reached
                info["message"] = "Browsing limit reached."
                
        elif action == "finish_browsing":
            self.browsing_done = True
            info["message"] = "Agent finished browsing explicitly."

        elif action == "generate_solution" and self.browsing_done and not self.solution_generated:
            # Simulate solution generation based on collected references
            if "slow internet" in self.customer_query.lower() and "restart router" in " ".join(self.collected_references).lower():
                reward += 10 # High reward for good solution
                info["solution_quality"] = "Excellent"
            elif self.collected_references:
                reward += 5 # Moderate reward if some references were used
                info["solution_quality"] = "Good"
            else:
                reward -= 5 # Penalty for generating solution without references
                info["solution_quality"] = "Poor"
            self.solution_generated = True
            done = True
            info["final_solution"] = f"Solution generated using {len(self.collected_references)} references."
        elif self.solution_generated:
            done = True # Already solved, end episode
            info["message"] = "Solution already generated."
        else:
            reward -= 1 # Penalty for invalid action or idle
            info["message"] = "Invalid action or not ready for solution generation."

        return self._get_state(), reward, done, info
    
    def get_collected_references(self):
        return self.collected_references

class RLAgent:
    def __init__(self):
        # In a real scenario, this would be a deep learning model (e.g., policy and value networks)
        # For this simulation, we'll use a simple heuristic.
        pass

    def act(self, state, phase="full"):
        if phase == "full":
            if not state["browsing_done"] and state["references_collected_count"] < 3 and state["query"] not in ["billing_issue", "account_recovery"]:
                return "browse"
            else:
                return "generate_solution"
        elif phase == "answering_only":
            return "generate_solution"
        return "idle"

    def learn(self, experience_batch):
        # In a real scenario, this would involve backpropagation and model updates.
        # For simulation, we'll just print that learning happened.
        # experience_batch contains tuples of (state, action, reward, next_state, done, references_used)
        print(f"Agent learned from {len(experience_batch)} experiences.")
        # Example of how references_used could be leveraged:
        # if "Excellent" in [exp[3].get("solution_quality") for exp in experience_batch]:
        #     print("Good solutions reinforced!")

class ExperienceReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, experience):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size) if len(self.buffer) >= batch_size else []

    def __len__(self):
        return len(self.buffer)

# Main simulation loop
if __name__ == "__main__":
    kb = KnowledgeBase()
    ch = CustomerHistory()
    env = CustomerInteractionEnv(kb, ch)
    agent = RLAgent()
    replay_buffer = ExperienceReplayBuffer(capacity=1000)

    num_full_episodes = 20
    num_answering_only_reuse = 5 # Number of additional answering-only episodes
    training_batch_size = 32

    print("Starting RL training with Sample-Efficient Reference Reuse...")

    for i_episode in range(num_full_episodes):
        customer_id = random.choice(["customer_A", "customer_B", "customer_C"])
        query = random.choice(["slow internet", "billing issue", "setup guide", "account recovery", "unrecognized charge"])
        
        print(f"\n--- Full Episode {i_episode + 1} --- (Customer: {customer_id}, Query: '{query}')")
        state = env.reset(customer_id, query)
        episode_experience = []
        done = False
        total_reward = 0
        collected_references_for_reuse = []

        # Phase 1: Full Multi-phase Episode (Browsing + Answering)
        print("  [Phase 1: Browsing and Answering]")
        browsing_done_in_full_episode = False
        while not done:
            action = agent.act(state, phase="full")
            next_state, reward, done, info = env.step(action)
            episode_experience.append((state, action, reward, next_state, done, env.get_collected_references()))
            total_reward += reward
            state = next_state
            if env.browsing_done and not browsing_done_in_full_episode:
                print(f"    Browsing finished. Collected {len(env.get_collected_references())} references.")
                collected_references_for_reuse = env.get_collected_references().copy()
                browsing_done_in_full_episode = True
            if done:
                print(f"    Full episode finished. Total Reward: {total_reward:.2f}, Info: {info.get('solution_quality', 'N/A')}")
                
        # Add experience from the full episode to replay buffer
        for exp in episode_experience:
            replay_buffer.push(exp)

        # Phase 2: Additional 'Answering-Only' Episodes with Reference Reuse
        if collected_references_for_reuse:
            print(f"  [Phase 2: Generating {num_answering_only_reuse} Answering-Only Episodes with Reference Reuse]")
            for _ in range(num_answering_only_reuse):
                # Reset environment, but provide pre-collected references
                reuse_env = CustomerInteractionEnv(kb, ch)
                reuse_env.reset(customer_id, query)
                reuse_env.collected_references = collected_references_for_reuse.copy() # Inject references
                reuse_env.browsing_done = True # Skip browsing phase

                reuse_episode_experience = []
                reuse_done = False
                reuse_total_reward = 0
                reuse_state = reuse_env._get_state() # Get initial state after injecting references

                while not reuse_done:
                    action = agent.act(reuse_state, phase="answering_only") # Agent acts in answering-only mode
                    next_state, reward, reuse_done, info = reuse_env.step(action)
                    reuse_episode_experience.append((reuse_state, action, reward, next_state, reuse_done, reuse_env.get_collected_references()))
                    reuse_total_reward += reward
                    reuse_state = next_state
                    if reuse_done:
                        # print(f"    Answering-only episode finished. Total Reward: {reuse_total_reward:.2f}, Info: {info.get('solution_quality', 'N/A')}")
                        pass # Suppress verbose output for reuse episodes
                
                # Add experience from answering-only episode to replay buffer
                for exp in reuse_episode_experience:
                    replay_buffer.push(exp)
        else:
            print("  No references collected in full episode, skipping answering-only reuse.")

        # Agent learns from a batch of experiences
        if len(replay_buffer) >= training_batch_size:
            batch = replay_buffer.sample(training_batch_size)
            agent.learn(batch)

    print("\nRL Training completed.")