import gym
from gym import spaces
import numpy as np
import random
from collections import deque

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

# --- 1. NLU Module (Simulated) ---
class NLUEmbedder:
    def __init__(self, embedding_dim=768):
        self.embedding_dim = embedding_dim

    def embed(self, text):
        # Simulate embedding using a random vector
        return np.random.rand(self.embedding_dim).astype(np.float32)

# --- 2. Knowledge Base (KB) / Reference Store (Simulated with basic retrieval) ---
class KnowledgeBase:
    def __init__(self, documents, embedder):
        self.documents = documents
        self.embedder = embedder
        self.document_embeddings = {doc: embedder.embed(doc) for doc in documents}

    def retrieve(self, query_embedding, top_n=3):
        # Simulate cosine similarity for retrieval
        query_embedding_norm = query_embedding / np.linalg.norm(query_embedding)
        similarities = []
        for doc, doc_emb in self.document_embeddings.items():
            doc_emb_norm = doc_emb / np.linalg.norm(doc_emb)
            similarity = np.dot(query_embedding_norm, doc_emb_norm)
            similarities.append((similarity, doc))
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in similarities[:top_n]]

# --- 3. NLG Module (Simulated) ---
class NLGGenerator:
    def generate_answer(self, query, references):
        # Simulate answer generation based on query and references
        if references:
            return f"Based on {references[0]} and your query '{query}', a simulated answer is provided."
        return f"I'm sorry, I couldn't find enough information for '{query}'."

# --- 4. Reward Function ---
class RewardCalculator:
    def calculate_reward(self, generated_answer, ideal_answer, browsing_cost, relevance_score):
        # Simulate answer quality (e.g., using a simple string match or semantic similarity score)
        answer_quality = 1.0 if "simulated answer" in generated_answer else 0.1

        conciseness_penalty = max(0, (len(generated_answer) - 100) * 0.01) # Penalize long answers
        
        # Combine rewards
        reward = (answer_quality * 5.0) + (relevance_score * 2.0) - browsing_cost - conciseness_penalty
        return max(-5.0, min(5.0, reward)) # Clip rewards

# --- 5. Customer Support Environment (gym.Env Compatible) ---
class CustomerSupportEnv(gym.Env):
    def __init__(self, kb, nlu_embedder, nlg_generator, reward_calculator, max_browsing_steps=5, embedding_dim=768):
        super(CustomerSupportEnv, self).__init__()

        self.kb = kb
        self.nlu_embedder = nlu_embedder
        self.nlg_generator = nlg_generator
        self.reward_calculator = reward_calculator
        self.max_browsing_steps = max_browsing_steps
        self.embedding_dim = embedding_dim

        self.action_space = spaces.Dict({
            "browse_action": spaces.Discrete(2), # 0: search, 1: answer
            "select_reference_idx": spaces.Discrete(3) # Index of retrieved reference to use for answer (0, 1, 2)
        })

        self.observation_space = spaces.Dict({
            "query_embedding": spaces.Box(low=-np.inf, high=np.inf, shape=(self.embedding_dim,), dtype=np.float32),
            "browsing_step": spaces.Box(low=0, high=self.max_browsing_steps, shape=(1,), dtype=np.float32),
            "retrieved_references_embedding": spaces.Box(low=-np.inf, high=np.inf, shape=(self.embedding_dim * 3,), dtype=np.float32) # Embeddings of 3 top references concatenated
        })

        self.current_query = None
        self.current_query_embedding = None
        self.browsing_steps_taken = 0
        self.retrieved_references = []
        self.full_episode_references = [] # Store references for reuse
        self.browsing_cost_acc = 0.0
        self.is_answering_only_episode = False
        self.pre_set_references = []

    def _get_observation(self):
        ref_embeddings = []
        for i in range(3):
            if i < len(self.retrieved_references):
                ref_embeddings.append(self.kb.embedder.embed(self.retrieved_references[i]))
            else:
                ref_embeddings.append(np.zeros(self.embedding_dim, dtype=np.float32))
        
        return {
            "query_embedding": self.current_query_embedding,
            "browsing_step": np.array([self.browsing_steps_taken], dtype=np.float32),
            "retrieved_references_embedding": np.concatenate(ref_embeddings)
        }

    def reset(self, user_query=None, is_answering_only=False, pre_set_references=None):
        self.current_query = user_query if user_query else "How do I reset my password?"
        self.current_query_embedding = self.nlu_embedder.embed(self.current_query)
        self.browsing_steps_taken = 0
        self.retrieved_references = []
        self.full_episode_references = []
        self.browsing_cost_acc = 0.0
        self.is_answering_only_episode = is_answering_only
        self.pre_set_references = pre_set_references if pre_set_references else []

        if self.is_answering_only_episode and self.pre_set_references:
            self.retrieved_references = self.pre_set_references # Skip browsing, use pre-set refs
            self.browsing_steps_taken = self.max_browsing_steps # Mark as if browsing is done

        return self._get_observation()

    def step(self, action):
        browse_action = action["browse_action"]
        select_reference_idx = action["select_reference_idx"]

        reward = 0.0
        done = False
        info = {}
        
        if self.is_answering_only_episode and browse_action == 0: # Cannot browse in answering-only mode
            reward -= 1.0 # Penalize invalid action
            browse_action = 1 # Force to answer phase

        if browse_action == 0 and not self.is_answering_only_episode: # Browsing phase
            if self.browsing_steps_taken < self.max_browsing_steps:
                self.browsing_steps_taken += 1
                self.browsing_cost_acc += 0.1
                
                # Simulate searching KB
                self.retrieved_references = self.kb.retrieve(self.current_query_embedding)
                self.full_episode_references = list(self.retrieved_references) # Store for reuse
                
                # Simple relevance score simulation
                relevance_score = 0.0
                if self.retrieved_references:
                    relevance_score = 1.0 # Assume finding references is good
                reward += self.reward_calculator.calculate_reward(
                    generated_answer="", ideal_answer="", browsing_cost=0.1, relevance_score=relevance_score
                )
            else:
                reward -= 0.5 # Penalize over-browsing
                done = True # End episode if too many browsing steps without answering

        else: # Answering phase (browse_action == 1 or forced in answering-only mode)
            selected_ref = None
            if self.retrieved_references and 0 <= select_reference_idx < len(self.retrieved_references):
                selected_ref = self.retrieved_references[select_reference_idx]
            
            # Simulate ideal answer
            ideal_answer = "Your password can be reset via the 'Forgot Password' link on the login page."

            generated_answer = self.nlg_generator.generate_answer(self.current_query, [selected_ref] if selected_ref else [])
            
            relevance_score = 1.0 if selected_ref else 0.0 # Reward for using a reference

            reward += self.reward_calculator.calculate_reward(
                generated_answer=generated_answer, ideal_answer=ideal_answer,
                browsing_cost=self.browsing_cost_acc, relevance_score=relevance_score
            )
            done = True
            info["generated_answer"] = generated_answer
            info["full_episode_references"] = self.full_episode_references

        obs = self._get_observation()
        return obs, reward, done, info

# --- 6. Reference Reuse Mechanism (Enhanced Replay Buffer) ---
class ReferenceReuseBuffer:
    def __init__(self, capacity=100):
        self.buffer = deque(maxlen=capacity)

    def store_episode(self, user_query, full_episode_references, final_reward, full_episode_duration):
        if full_episode_references:
            self.buffer.append({
                "user_query": user_query,
                "retrieved_references": full_episode_references,
                "final_reward": final_reward,
                "full_episode_duration": full_episode_duration
            })

    def sample_answering_only_episode(self):
        if not self.buffer:
            return None
        return random.choice(list(self.buffer))

    def __len__(self):
        return len(self.buffer)

# --- Training Loop Orchestration ---
if __name__ == "__main__":
    # Initialize Components
    embedding_dim = 768
    nlu_embedder = NLUEmbedder(embedding_dim=embedding_dim)
    
    kb_documents = [
        "Troubleshooting common login issues",
        "How to reset your password and account recovery options",
        "Updating personal information in your profile",
        "Contacting customer support for further assistance",
        "Billing and subscription management"
    ]
    kb = KnowledgeBase(kb_documents, nlu_embedder)
    nlg_generator = NLGGenerator()
    reward_calculator = RewardCalculator()

    env = CustomerSupportEnv(kb, nlu_embedder, nlg_generator, reward_calculator, embedding_dim=embedding_dim)

    # PPO Agent
    model = PPO("MultiInputPolicy", env, verbose=0, n_steps=1000, tensorboard_log="./ppo_chatbot_tensorboard/")

    # Reference Reuse Buffer
    ref_reuse_buffer = ReferenceReuseBuffer(capacity=50)

    total_timesteps = 100000
    num_full_episodes = 0
    num_answering_only_episodes = 0
    answering_only_frequency = 5 # Generate 5 answering-only episodes per full episode

    print("Starting RL Training with Reference Reuse...")

    obs = env.reset()
    for timestep in range(total_timesteps):
        if timestep % 1000 == 0: # Periodically train the agent
            print(f"Timestep: {timestep}/{total_timesteps}, Full Episodes: {num_full_episodes}, Answering-Only Episodes: {num_answering_only_episodes}")

        # --- 1. Run a FULL episode ---
        current_user_query = random.choice(["My account is locked", "How to update my email?", "Can I get a refund?"])
        obs = env.reset(user_query=current_user_query)
        done = False
        episode_reward = 0
        episode_duration = 0
        full_episode_references_collected = []

        while not done:
            action, _states = model.predict(obs, deterministic=False) # Agent decides browse or answer
            new_obs, reward, done, info = env.step(action)
            episode_reward += reward
            episode_duration += 1
            obs = new_obs
            
            if done and "full_episode_references" in info:
                full_episode_references_collected = info["full_episode_references"]

        # Store full episode experience in the reference reuse buffer
        if full_episode_references_collected:
            ref_reuse_buffer.store_episode(current_user_query, full_episode_references_collected, episode_reward, episode_duration)
        num_full_episodes += 1

        # --- 2. Generate and run ANswering-ONLY episodes ---
        for _ in range(answering_only_frequency):
            sampled_experience = ref_reuse_buffer.sample_answering_only_episode()
            if sampled_experience:
                num_answering_only_episodes += 1
                
                # Reset environment for answering-only, providing collected references
                obs_answering_only = env.reset(
                    user_query=sampled_experience["user_query"],
                    is_answering_only=True,
                    pre_set_references=sampled_experience["retrieved_references"]
                )
                
                done_answering_only = False
                answering_only_episode_reward = 0
                answering_only_episode_duration = 0

                while not done_answering_only:
                    # In answering-only, agent mostly tries to answer (action[0]=1)
                    # We ensure the action is 'answer' as browsing is bypassed
                    action_answering_only, _states_ao = model.predict(obs_answering_only, deterministic=False)
                    action_answering_only["browse_action"] = 1 # Force answer action

                    new_obs_ao, reward_ao, done_answering_only, info_ao = env.step(action_answering_only)
                    answering_only_episode_reward += reward_ao
                    answering_only_episode_duration += 1
                    obs_answering_only = new_obs_ao
            else:
                break # No experiences to sample

        # Agent learns from all collected experiences (both full and answering-only)
        # stable-baselines3's PPO 'learn' method handles its own experience buffer implicitly.
        # The key is that the 'env.step' calls generate observations, actions, rewards which PPO consumes.
        # By running more answering-only steps, we provide more 'answer-phase' specific experiences
        # to the underlying PPO buffer.
        if timestep % 500 == 0 and timestep > 0: # Periodically update the agent
             model.learn(total_timesteps=1000, reset_num_timesteps=False, log_interval=10)

    print("RL Training complete.")
    # Example of saving the model
    model.save("rl_chatbot_model")

    # --- Test the trained agent (simple inference) ---
    print("\nTesting the trained agent...")
    test_query = "I can't log in to my account."
    obs = env.reset(user_query=test_query)
    test_done = False
    test_references = []
    print(f"User Query: {test_query}")

    while not test_done:
        action, _states = model.predict(obs, deterministic=True)
        new_obs, reward, test_done, test_info = env.step(action)
        obs = new_obs
        if "full_episode_references" in test_info:
            test_references.extend(test_info["full_episode_references"])
        if "generated_answer" in test_info:
            print(f"Chatbot Answer: {test_info['generated_answer']}")
            print(f"References Used: {', '.join(test_references)}")

    test_query_2 = "How do I manage my billing?"
    obs = env.reset(user_query=test_query_2)
    test_done = False
    test_references = []
    print(f"\nUser Query: {test_query_2}")

    while not test_done:
        action, _states = model.predict(obs, deterministic=True)
        new_obs, reward, test_done, test_info = env.step(action)
        obs = new_obs
        if "full_episode_references" in test_info:
            test_references.extend(test_info["full_episode_references"])
        if "generated_answer" in test_info:
            print(f"Chatbot Answer: {test_info['generated_answer']}")
            print(f"References Used: {', '.join(test_references)}")
