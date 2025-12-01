import numpy as np
from collections import deque
import random

class KnowledgeBase:
    def __init__(self):
        self.documents = {
            "doc_troubleshooting_network": "Troubleshooting network connectivity issues often involves checking your router, cables, and DNS settings. Make sure your IP configuration is correct.",
            "doc_software_install_guide": "To install the software, run the setup wizard and follow the on-screen prompts. Ensure you have administrator privileges.",
            "doc_account_recovery": "For account recovery, visit the password reset page and enter your registered email address. A recovery link will be sent.",
            "doc_performance_optimization": "Optimize software performance by clearing cache, updating drivers, and checking for background processes. Increase RAM if possible.",
            "doc_faq_common_errors": "Common errors include 'Access Denied' (check permissions) and 'File Not Found' (verify path)."
        }

    def search(self, query):
        found_references = []
        query_lower = query.lower()
        if "network" in query_lower or "connectivity" in query_lower:
            found_references.append("doc_troubleshooting_network")
        if "install" in query_lower or "setup" in query_lower:
            found_references.append("doc_software_install_guide")
        if "password" in query_lower or "account" in query_lower:
            found_references.append("doc_account_recovery")
        if "performance" in query_lower or "slow" in query_lower:
            found_references.append("doc_performance_optimization")
        if "error" in query_lower:
            found_references.append("doc_faq_common_errors")
        
        # Simulate content of references
        return [f"Ref: {doc_id} - {self.documents[doc_id][:50]}..." for doc_id in found_references]

class ChatbotEnvironment:
    BROWSING_PHASE = 0
    ANSWERING_PHASE = 1

    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base
        self.current_phase = self.BROWSING_PHASE
        self.gathered_references = []
        self.user_query = ""
        self.browsing_cost = -0.1 # Small negative reward for browsing steps
        self.answering_success_reward = 10.0
        self.answering_failure_reward = -5.0
        self.max_browsing_steps = 3
        self.max_answering_steps = 2
        self.current_steps_in_phase = 0
        self.state_space_size = 3 # Simplified: 0=browsing, 1=answering (no refs), 2=answering (with refs)
        self.action_space_size = 4 # 0=search, 1=next_question, 2=provide_solution_good, 3=provide_solution_bad

    def _get_state(self):
        if self.current_phase == self.BROWSING_PHASE:
            return 0 # Browsing state
        else:
            return 2 if self.gathered_references else 1 # Answering with refs vs. Answering no refs

    def reset(self, mode="full_episode", initial_references=None, user_query="How do I fix my network connection?"):
        self.user_query = user_query
        self.gathered_references = []
        self.current_steps_in_phase = 0
        self.done = False

        if mode == "full_episode":
            self.current_phase = self.BROWSING_PHASE
        elif mode == "answering_only" and initial_references is not None:
            self.current_phase = self.ANSWERING_PHASE
            self.gathered_references = initial_references
        else:
            raise ValueError("Invalid reset mode or missing initial_references for answering_only mode.")
        
        return self._get_state(), self.current_phase # observation, info

    def step(self, action):
        reward = 0
        self.done = False
        info = {"phase_transitioned": False}

        if self.current_phase == self.BROWSING_PHASE:
            if action == 0: # Simulate 'search'
                search_results = self.knowledge_base.search(self.user_query)
                self.gathered_references.extend(search_results)
                reward += self.browsing_cost * 0.5 # Slightly less costly search
            elif action == 1: # Simulate 'ask next question'
                reward += self.browsing_cost
            else: # Invalid action for browsing phase, or attempt to answer prematurely
                reward += self.browsing_cost * 2

            self.current_steps_in_phase += 1
            if self.current_steps_in_phase >= self.max_browsing_steps or len(self.gathered_references) > 0:
                self.current_phase = self.ANSWERING_PHASE
                self.current_steps_in_phase = 0
                info["phase_transitioned"] = True

        elif self.current_phase == self.ANSWERING_PHASE:
            if action == 2: # Simulate 'provide_solution_good'
                # Simple heuristic for good solution: depends on having references
                if self.gathered_references:
                    reward += self.answering_success_reward
                else:
                    reward += self.answering_failure_reward * 0.5 # Less severe if tried but no refs
                self.done = True
            elif action == 3: # Simulate 'provide_solution_bad'
                reward += self.answering_failure_reward
                self.done = True
            else: # Invalid action for answering phase, or attempt to browse again
                reward += self.answering_failure_reward * 0.1

            self.current_steps_in_phase += 1
            if self.current_steps_in_phase >= self.max_answering_steps and not self.done:
                # If agent didn't finish answering in time, it's a failure
                reward += self.answering_failure_reward
                self.done = True
        
        return self._get_state(), reward, self.done, info

    def get_gathered_references(self):
        return list(self.gathered_references)

class RLAgent:
    def __init__(self, state_space_size, action_space_size, learning_rate=0.1, discount_factor=0.95, exploration_rate=1.0, min_exploration_rate=0.01, exploration_decay_rate=0.995):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_min = min_exploration_rate
        self.epsilon_decay = exploration_decay_rate

        self.q_table = np.zeros((state_space_size, action_space_size))
        self.replay_buffer = deque(maxlen=2000) # Experience replay buffer

    def _get_action_mask(self, phase):
        mask = np.ones(self.action_space_size, dtype=bool)
        if phase == ChatbotEnvironment.BROWSING_PHASE:
            mask[[2, 3]] = False # Cannot provide solution in browsing phase
        elif phase == ChatbotEnvironment.ANSWERING_PHASE:
            mask[[0, 1]] = False # Cannot search/ask_question in answering phase
        return mask

    def choose_action(self, observation, phase):
        action_mask = self._get_action_mask(phase)
        valid_actions_indices = np.where(action_mask)[0]

        if np.random.rand() < self.epsilon:
            return np.random.choice(valid_actions_indices) # Explore
        else:
            # Exploit: choose best valid action
            q_values = self.q_table[observation, :]
            masked_q_values = q_values[valid_actions_indices]
            return valid_actions_indices[np.argmax(masked_q_values)]

    def learn(self, batch_size=32):
        if len(self.replay_buffer) < batch_size:
            return

        minibatch = random.sample(self.replay_buffer, batch_size)

        for state, action, reward, next_state, done in minibatch:
            if done:
                target = reward
            else:
                target = reward + self.gamma * np.max(self.q_table[next_state, :])
            
            self.q_table[state, action] = self.q_table[state, action] + self.lr * (target - self.q_table[state, action])

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# Training Orchestrator
if __name__ == "__main__":
    knowledge_base = KnowledgeBase()
    env = ChatbotEnvironment(knowledge_base)
    agent = RLAgent(env.state_space_size, env.action_space_size)

    num_full_episodes = 200
    num_answering_only_episodes_per_full = 5
    batch_size = 32
    update_q_freq = 4 # Learn every 4 steps

    print("Starting RL Chatbot Training with Reference Reuse...")

    total_rewards = []

    for episode in range(num_full_episodes):
        current_references = []
        episode_reward = 0
        step_count = 0

        # --- Full Episode Training ---
        obs, phase = env.reset(mode="full_episode", user_query=random.choice([
            "My internet is not working", "How to install new software?", 
            "Forgot my password, help!", "Why is my computer slow?", "I got an error message."
        ]))
        done = False

        while not done:
            action = agent.choose_action(obs, phase)
            next_obs, reward, done, info = env.step(action)
            agent.replay_buffer.append((obs, action, reward, next_obs, done))
            obs = next_obs
            phase = env.current_phase # Update phase if transitioned
            episode_reward += reward
            step_count += 1

            if step_count % update_q_freq == 0:
                agent.learn(batch_size)

            if done:
                current_references = env.get_gathered_references()
                break # End of full episode

        total_rewards.append(episode_reward)

        # --- Supplementary Answering-Only Episodes ---
        if current_references and episode % 10 == 0: # Periodically run supplementary training
            for _ in range(num_answering_only_episodes_per_full):
                answering_episode_reward = 0
                answering_step_count = 0
                
                # Reset environment directly to answering phase with collected references
                ans_obs, ans_phase = env.reset(mode="answering_only", initial_references=current_references)
                ans_done = False

                while not ans_done:
                    # Agent only chooses actions relevant to answering
                    ans_action = agent.choose_action(ans_obs, ans_phase)
                    ans_next_obs, ans_reward, ans_done, ans_info = env.step(ans_action)
                    agent.replay_buffer.append((ans_obs, ans_action, ans_reward, ans_next_obs, ans_done))
                    ans_obs = ans_next_obs
                    answering_episode_reward += ans_reward
                    answering_step_count += 1

                    if answering_step_count % update_q_freq == 0:
                        agent.learn(batch_size)

        if episode % 10 == 0:
            avg_reward = np.mean(total_rewards[-10:])
            print(f"Episode {episode}/{num_full_episodes}, Full Episode Reward: {episode_reward:.2f}, Avg 10-ep Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

    print("\nTraining complete!")
    print(f"Average reward over all full episodes: {np.mean(total_rewards):.2f}")