import collections
import random

class MockKnowledgeBase:
    def __init__(self):
        self.faqs = {
            "shipping status": "Your order #12345 is currently in transit and expected to arrive by [Date].",
            "return policy": "You can return items within 30 days of purchase with the original receipt.",
            "technical support": "Please describe your technical issue in detail and we will connect you to an agent."
        }
        self.crm_data = {
            "customer_A": {"order_history": ["#12345", "#67890"], "contact": "email"},
            "customer_B": {"order_history": ["#11223"], "contact": "phone"}
        }

    def search_faq(self, query):
        for q, a in self.faqs.items():
            if query in q or q in query:
                return a
        return None

    def get_customer_data(self, customer_id):
        return self.crm_data.get(customer_id)

class ChatbotEnvironment:
    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base
        self.reset()

    def reset(self, initial_query=None, pre_gathered_references=None):
        self.phase = "info_gathering"
        self.current_query = initial_query if initial_query else random.choice(list(self.knowledge_base.faqs.keys()))
        self.gathered_references = pre_gathered_references if pre_gathered_references else []
        self.done = False
        self.turn = 0
        self.max_turns = 5 # Limit for info gathering or answer generation turns
        self.rewards_history = []
        if pre_gathered_references:
            self.phase = "answer_generation"
        return self._get_state()

    def _get_state(self):
        return {"phase": self.phase, "query": self.current_query, "references": self.gathered_references, "turn": self.turn}

    def step(self, action):
        reward = 0
        info = {}
        self.turn += 1

        if self.phase == "info_gathering":
            if action["type"] == "ask_clarifying":
                self.current_query = action["payload"]
                reward -= 0.1 # Small penalty for asking questions
            elif action["type"] == "search_kb":
                search_result = self.knowledge_base.search_faq(action["payload"])
                if search_result:
                    self.gathered_references.append(search_result)
                    reward += 0.5 # Reward for finding relevant info
                else:
                    reward -= 0.2 # Penalty for irrelevant search
            elif action["type"] == "transition_to_answer":
                self.phase = "answer_generation"
                reward += 0.1 # Small reward for moving to next phase
            else:
                reward -= 0.3 # Invalid action

            if self.turn >= self.max_turns or len(self.gathered_references) > 0:
                # Auto-transition or force transition if info gathered or turns exceed
                if self.phase == "info_gathering":
                    self.phase = "answer_generation"

        elif self.phase == "answer_generation":
            if action["type"] == "provide_answer":
                # Simple reward logic: higher reward for using references
                if any(ref in action["payload"] for ref in self.gathered_references):
                    reward += 10.0
                else:
                    reward += 2.0
                self.done = True
                info["final_answer"] = action["payload"]
            else:
                reward -= 1.0 # Invalid action in answer phase

        if self.done:
            self.rewards_history.append(reward)

        return self._get_state(), reward, self.done, info

class ExperienceReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done, references_collected):
        self.buffer.append((state, action, reward, next_state, done, references_collected))

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return []
        return random.sample(list(self.buffer), batch_size)

    def __len__(self):
        return len(self.buffer)

class RLChatbotAgent:
    def __init__(self, action_space_size, replay_buffer):
        self.action_space_size = action_space_size
        self.replay_buffer = replay_buffer
        self.policy_weights = {} # Placeholder for policy weights

    def choose_action(self, state):
        # Simplified action choice for demonstration
        if state["phase"] == "info_gathering":
            if not state["references"] and state["turn"] < 3:
                # Try to search or ask clarifying
                return {"type": "search_kb", "payload": state["query"]}
            else:
                return {"type": "transition_to_answer"}
        elif state["phase"] == "answer_generation":
            # Simple answer generation: combine references or give generic
            if state["references"]:
                return {"type": "provide_answer", "payload": "Based on the information: " + ". ".join(state["references"])}
            else:
                return {"type": "provide_answer", "payload": "I need more information to provide a specific answer."}
        return {"type": "invalid"}

    def learn(self, experiences):
        # In a real RL setup, this would update policy/value function
        # For this demo, we just acknowledge learning happened
        pass

class TrainerOrchestrator:
    def __init__(self, num_full_episodes, num_supplementary_per_full, replay_buffer_capacity=1000):
        self.knowledge_base = MockKnowledgeBase()
        self.environment = ChatbotEnvironment(self.knowledge_base)
        self.replay_buffer = ExperienceReplayBuffer(replay_buffer_capacity)
        self.agent = RLChatbotAgent(action_space_size=5, replay_buffer=self.replay_buffer) # Placeholder size
        self.num_full_episodes = num_full_episodes
        self.num_supplementary_per_full = num_supplementary_per_full

        self.total_rewards_full_episodes = []
        self.total_rewards_supplementary_episodes = []

    def _run_single_episode(self, initial_query=None, pre_gathered_references=None, is_supplementary=False):
        state = self.environment.reset(initial_query, pre_gathered_references)
        done = False
        episode_reward = 0
        references_collected_in_full_episode = []

        while not done:
            action = self.agent.choose_action(state)
            next_state, reward, done, info = self.environment.step(action)
            self.replay_buffer.add(state, action, reward, next_state, done, self.environment.gathered_references.copy())
            episode_reward += reward
            state = next_state

            if state["phase"] == "info_gathering":
                references_collected_in_full_episode = self.environment.gathered_references.copy()

        if is_supplementary:
            self.total_rewards_supplementary_episodes.append(episode_reward)
        else:
            self.total_rewards_full_episodes.append(episode_reward)

        return episode_reward, references_collected_in_full_episode

    def train(self):
        print("Starting RL Chatbot Training...")
        for i in range(self.num_full_episodes):
            print(f"\n--- Running Full Episode {i+1}/{self.num_full_episodes} ---")
            initial_query_for_full = random.choice(list(self.knowledge_base.faqs.keys()))
            full_episode_reward, references = self._run_single_episode(initial_query=initial_query_for_full, is_supplementary=False)
            print(f"Full episode {i+1} reward: {full_episode_reward:.2f}")
            print(f"References collected: {references}")

            # Generate supplementary answering-only episodes
            if references:
                print(f"Generating {self.num_supplementary_per_full} supplementary answering-only episodes...")
                for j in range(self.num_supplementary_per_full):
                    supp_episode_reward, _ = self._run_single_episode(initial_query=initial_query_for_full, pre_gathered_references=references, is_supplementary=True)
                    print(f"  Supplementary episode {j+1} reward: {supp_episode_reward:.2f}")

            # Agent learns from experiences
            if len(self.replay_buffer) >= 32: # A minimal batch size for learning
                batch = self.replay_buffer.sample(32)
                self.agent.learn(batch)

        print("\n--- Training Complete ---")
        print(f"Average reward for full episodes: {sum(self.total_rewards_full_episodes)/len(self.total_rewards_full_episodes):.2f}")
        if self.total_rewards_supplementary_episodes:
            print(f"Average reward for supplementary episodes: {sum(self.total_rewards_supplementary_episodes)/len(self.total_rewards_supplementary_episodes):.2f}")

if __name__ == "__main__":
    trainer = TrainerOrchestrator(num_full_episodes=5, num_supplementary_per_full=3)
    trainer.train()

    # Example of a single interaction outside training
    print("\n--- Demonstrating a single interaction after training ---")
    kb = MockKnowledgeBase()
    env = ChatbotEnvironment(kb)
    agent = RLChatbotAgent(action_space_size=5, replay_buffer=ExperienceReplayBuffer(10))

    initial_q = "shipping status"
    state = env.reset(initial_query=initial_q)
    print(f"Initial Query: {initial_q}")
    single_interaction_reward = 0
    single_interaction_done = False
    while not single_interaction_done:
        action = agent.choose_action(state)
        print(f"Agent action: {action}")
        state, reward, single_interaction_done, info = env.step(action)
        single_interaction_reward += reward
        print(f"  Current State: {state['phase']}, References: {state['references']}, Reward: {reward:.2f}")
        if "final_answer" in info:
            print(f"  Final Answer: {info['final_answer']}")
    print(f"Total reward for single interaction: {single_interaction_reward:.2f}")
