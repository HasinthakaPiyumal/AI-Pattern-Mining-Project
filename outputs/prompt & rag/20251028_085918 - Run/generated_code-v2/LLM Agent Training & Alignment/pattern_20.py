import tensorflow as tf
import numpy as np
import random

# --- 1. Policy Network (TensorFlow) ---
class PolicyNetwork(tf.keras.Model):
    def __init__(self, obs_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.logits = tf.keras.layers.Dense(action_dim)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.logits(x)

# --- 2. Environment Classes ---
class FullInteractionEnv:
    def __init__(self, num_solutions=5):
        self.current_state = None
        self.collected_references = []
        self.num_solutions = num_solutions
        self.problem_description = ""
        self.phase = "info_gathering" # "info_gathering" or "solution_recommendation"
        self.optimal_solution_idx = -1

    def _generate_problem(self):
        # Simulate a complex product troubleshooting problem
        problem_types = ["connectivity issue", "software bug", "hardware malfunction", "performance degradation"]
        self.problem_description = random.choice(problem_types) + " on product XYZ"
        self.optimal_solution_idx = random.randint(0, self.num_solutions - 1)

    def reset(self):
        self._generate_problem()
        self.current_state = np.random.rand(10).astype(np.float32) # Initial observation
        self.collected_references = []
        self.phase = "info_gathering"
        return self.current_state

    def step(self, action):
        reward = 0
        done = False

        if self.phase == "info_gathering":
            # Simulate information gathering based on action (e.g., browsing manuals)
            # Action could be an index to query a knowledge base
            simulated_info = f"Reference related to {self.problem_description} and action {action}"
            self.collected_references.append(simulated_info)
            reward += 0.1 # Small reward for gathering information

            if len(self.collected_references) >= 3: # After collecting enough info, transition to solution phase
                self.phase = "solution_recommendation"
                # Update state to reflect readiness for solution, maybe include digested info
                self.current_state = np.concatenate([self.current_state, np.array([0.5, 0.5])]).astype(np.float32)
            else:
                self.current_state = np.random.rand(10).astype(np.float32)

        elif self.phase == "solution_recommendation":
            # Action is the recommended solution index
            if action == self.optimal_solution_idx:
                reward += 10.0 # High reward for correct solution
            else:
                reward -= 1.0 # Penalty for incorrect solution
            done = True

        return self.current_state, reward, done, self.collected_references.copy()

    def get_references(self):
        return self.collected_references.copy()


class SolutionOnlyEnv:
    def __init__(self, fixed_references, num_solutions=5):
        self.fixed_references = fixed_references
        self.num_solutions = num_solutions
        self.optimal_solution_idx = -1 # This should ideally be determined by the references
        self._set_optimal_solution_from_references()

    def _set_optimal_solution_from_references(self):
        # A very simplistic way to derive an 'optimal solution' from references
        # In a real system, an NLP model would analyze references to find the best solution
        if "connectivity issue" in str(self.fixed_references):
            self.optimal_solution_idx = 0
        elif "software bug" in str(self.fixed_references):
            self.optimal_solution_idx = 1
        elif "hardware malfunction" in str(self.fixed_references):
            self.optimal_solution_idx = 2
        else:
            self.optimal_solution_idx = random.randint(0, self.num_solutions - 1)

    def reset(self):
        # State derived directly from fixed references
        # For simplicity, a random state, but in reality, would be an embedding of references
        state_from_references = np.random.rand(12).astype(np.float32) # Matches full env's solution phase state size
        return state_from_references

    def step(self, action):
        reward = 0
        done = True # Always a terminal step in solution-only env

        if action == self.optimal_solution_idx:
            reward += 10.0 # High reward for correct solution based on references
        else:
            reward -= 1.0 # Penalty for incorrect solution

        return self.reset(), reward, done, None # New state for next episode, no new references here

# --- 3. RL Agent --- 
class RLAgent:
    def __init__(self, obs_dim, action_dim, learning_rate=1e-3):
        self.policy_network = PolicyNetwork(obs_dim, action_dim)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    def choose_action(self, state):
        state = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
        logits = self.policy_network(state)
        action_probs = tf.nn.softmax(logits)
        action = tf.random.categorical(logits, 1)[0, 0].numpy()
        return action, action_probs[0, action]

    def update_policy(self, rewards, log_probs):
        # Simple policy gradient update
        discounted_rewards = []
        cumulative_reward = 0
        for reward in reversed(rewards):
            cumulative_reward = reward + 0.99 * cumulative_reward # Discount factor gamma=0.99
            discounted_rewards.append(cumulative_reward)
        discounted_rewards.reverse()

        loss = []
        for log_prob, r in zip(log_probs, discounted_rewards):
            loss.append(-log_prob * r)
        loss = tf.stack(loss)

        with tf.GradientTape() as tape:
            total_loss = tf.reduce_sum(loss)
        grads = tape.gradient(total_loss, self.policy_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.policy_network.trainable_variables))

# --- Main Training Loop --- 
if __name__ == "__main__":
    OBS_DIM_FULL_ENV = 10 # Initial state dimension
    OBS_DIM_SOLUTION_ENV = 12 # State dimension during solution phase (after info gathering)
    ACTION_DIM_FULL_ENV = 5 # Actions for info gathering (e.g., query types) + solutions
    ACTION_DIM_SOLUTION_ONLY = 5 # Actions are just solutions

    # Agent needs to handle states for both phases, let's make it flexible or train two agents
    # For simplicity, we'll use one agent and assume state dimensions are padded/handled.
    # A more robust approach would be a hierarchical agent or separate policy heads.
    agent = RLAgent(obs_dim=OBS_DIM_SOLUTION_ENV, action_dim=ACTION_DIM_SOLUTION_ONLY)
    
    full_env = FullInteractionEnv(num_solutions=ACTION_DIM_SOLUTION_ONLY)
    
    NUM_FULL_EPISODES = 100
    NUM_SOLUTION_ONLY_EPISODES_PER_FULL = 15 # As suggested by the pattern

    print("Starting RL training with Reference Reuse...")

    for episode in range(NUM_FULL_EPISODES):
        # --- Full Interaction Episode ---
        current_state = full_env.reset()
        episode_rewards = []
        episode_log_probs = []
        episode_references = []
        done = False

        print(f"\n--- Full Episode {episode + 1} --- (Problem: {full_env.problem_description})")
        while not done:
            # Adjust observation for the agent if needed (e.g., pad or embed)
            # For this simplified example, assume agent can handle current_state if it matches solution phase dim
            # We'll use the larger observation dimension for the agent and pad if necessary
            if full_env.phase == "info_gathering":
                padded_state = np.pad(current_state, (0, OBS_DIM_SOLUTION_ENV - OBS_DIM_FULL_ENV), 'constant').astype(np.float32)
            else:
                padded_state = current_state # Already 12-dim after info gathering

            action, log_prob = agent.choose_action(padded_state)
            next_state, reward, done, references_from_env = full_env.step(action)

            episode_rewards.append(reward)
            episode_log_probs.append(tf.math.log(log_prob))
            episode_references = references_from_env # Keep updating with the latest references
            current_state = next_state
        
        # Update policy based on full episode experience
        agent.update_policy(episode_rewards, episode_log_probs)
        print(f"Full episode finished. Total Reward: {sum(episode_rewards):.2f}. References collected: {len(episode_references)}")

        # --- Generate and Train on Solution-Only Episodes ---
        if episode_references: # Only proceed if references were collected
            for sol_only_idx in range(NUM_SOLUTION_ONLY_EPISODES_PER_FULL):
                sol_only_env = SolutionOnlyEnv(fixed_references=episode_references, num_solutions=ACTION_DIM_SOLUTION_ONLY)
                sol_only_state = sol_only_env.reset()
                sol_only_rewards = []
                sol_only_log_probs = []

                sol_action, sol_log_prob = agent.choose_action(sol_only_state)
                _, sol_reward, _, _ = sol_only_env.step(sol_action)

                sol_only_rewards.append(sol_reward)
                sol_only_log_probs.append(tf.math.log(sol_log_prob))

                agent.update_policy(sol_only_rewards, sol_only_log_probs)
                # print(f"  Solution-only episode {sol_only_idx + 1}. Reward: {sol_reward:.2f}")
        else:
            print("  No references collected in full episode, skipping solution-only episodes.")

    print("\nRL Training with Reference Reuse Complete.")
    # Example of using the trained agent
    final_state = full_env.reset()
    if full_env.phase == "info_gathering":
        padded_state = np.pad(final_state, (0, OBS_DIM_SOLUTION_ENV - OBS_DIM_FULL_ENV), 'constant').astype(np.float32)
    else:
        padded_state = final_state
    final_action, _ = agent.choose_action(padded_state)
    print(f"\nExample: Agent recommends solution {final_action} for problem: {full_env.problem_description}")
