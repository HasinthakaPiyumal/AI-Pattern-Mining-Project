import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig
from sentence_transformers import SentenceTransformer, util
import chromadb
import gym
from gym import spaces
import numpy as np
from tqdm import tqdm
import random

# 1. Knowledge Base (KB) Module
class KnowledgeBase:
    def __init__(self, collection_name="customer_support_kb"):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self._populate_kb()

    def _populate_kb(self):
        documents = [
            "How to reset your password? Go to settings -> security -> reset password.",
            "Troubleshooting login issues: Check your internet connection, clear browser cache, or contact support.",
            "Understanding subscription plans: We offer Basic, Premium, and Enterprise plans with varying features.",
            "Contacting customer support: You can reach us via live chat, email, or phone from 9 AM to 5 PM EST.",
            "Billing inquiries: All billing information can be found in your account dashboard under 'Billing'.",
            "Product features overview: Our product includes task management, team collaboration, and reporting tools."
        ]
        ids = [f"doc_{i}" for i in range(len(documents))]
        embeddings = self.embedding_model.encode(documents).tolist()
        self.collection.add(documents=documents, embeddings=embeddings, ids=ids)
        print(f"Knowledge Base populated with {len(documents)} documents.")

    def retrieve_references(self, query: str, top_k: int = 3) -> list:
        query_embedding = self.embedding_model.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents']
        )
        return results['documents'][0] if results['documents'] else []

# 2. RL Environment Module
class RLChatbotEnvironment(gym.Env):
    def __init__(self, kb: KnowledgeBase, tokenizer, llm_pipeline):
        super().__init__()
        self.kb = kb
        self.tokenizer = tokenizer
        self.llm_pipeline = llm_pipeline

        self.observation_space = spaces.Dict({
            "query": spaces.Text(256),  # Max query length
            "references": spaces.Text(512) # Max combined references length
        })
        # Action space is generating text tokens (simplified, as LLM handles this)
        self.action_space = spaces.Text(256) # Max answer length

        self.current_query = None
        self.current_references = []
        self.ideal_answers = {
            "How do I reset my password?": "To reset your password, navigate to settings, then security, and you'll find the option there.",
            "I can't log in": "If you're having trouble logging in, please check your internet connection, clear your browser cache, or contact our support team.",
            "Tell me about your plans": "We have three main subscription plans: Basic, Premium, and Enterprise, each with different features to suit your needs."
        }
        self.reward_model = SentenceTransformer("all-MiniLM-L6-v2")

    def reset(self, query: str, pre_browsed_references: list = None):
        self.current_query = query
        if pre_browsed_references:
            self.current_references = pre_browsed_references
            browsing_cost = 0 # No browsing cost for pre-browsed
        else:
            # Simulate browsing phase
            self.current_references = self.kb.retrieve_references(query)
            browsing_cost = 0.1 # Small penalty for browsing

        obs = self._get_observation()
        return obs, {"browsing_cost": browsing_cost}

    def step(self, generated_answer: str):
        reward = self._calculate_reward(generated_answer)
        # For simplicity, episode ends after one step (answer generation)
        done = True
        info = {}
        obs = self._get_observation() # State remains same for one-step answer
        return obs, reward, done, info

    def _get_observation(self):
        return {
            "query": self.current_query,
            "references": " ".join(self.current_references)
        }

    def _calculate_reward(self, generated_answer: str) -> float:
        query_lower = self.current_query.lower()
        target_answer = None
        for q_key, a_val in self.ideal_answers.items():
            if q_key.lower() in query_lower:
                target_answer = a_val
                break
        
        if target_answer:
            embeddings = self.reward_model.encode([generated_answer, target_answer], convert_to_tensor=True)
            similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
            # Scale similarity to be between 0 and 1, a good answer gets higher reward
            reward = similarity # Directly use similarity as reward
        else:
            # If no ideal answer, assign a moderate baseline reward based on generic helpfulness
            # This is a simplification; in a real system, a more robust metric would be needed.
            reward = random.uniform(0.1, 0.5) # Placeholder for generic good answer
        
        # Add a small penalty for very short or empty answers if references were available
        if not generated_answer.strip() and self.current_references: 
            reward -= 0.2
        elif len(generated_answer.split()) < 5 and self.current_references: # Penalize overly short answers
            reward -= 0.1

        return max(-1.0, min(1.0, reward)) # Clamp reward between -1 and 1


# 3. RL Agent Module & 4. Reference Reuse Mechanism & 5. Training Orchestrator
class ChatbotRLTrainer:
    def __init__(self, model_name="distilgpt2", learning_rate=1e-5, num_ppo_epochs=4, batch_size=4, mini_batch_size=1):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token # For models without explicit pad_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        self.kb = KnowledgeBase()
        # Pipeline for LLM to generate answers
        self.llm_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1, # Use GPU if available
            max_new_tokens=50,
            num_return_sequences=1,
            do_sample=True, # Enable sampling for more diverse answers
            top_k=50, 
            top_p=0.95,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id # Ensure padding is correct
        )

        self.env = RLChatbotEnvironment(self.kb, self.tokenizer, self.llm_pipeline)

        ppo_config = PPOConfig(
            learning_rate=learning_rate,
            num_ppo_epochs=num_ppo_epochs,
            batch_size=batch_size,
            mini_batch_size=mini_batch_size,
            seed=42,
        )

        self.ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=self.model,
            ref_model=None, # Use the same model as a reference for now, or a frozen copy
            tokenizer=self.tokenizer,
        )

        self.episode_history = [] # Stores (query, references, generated_answer, reward)

    def _generate_answer_from_llm(self, observation):
        prompt = f"Question: {observation['query']}\nReferences: {observation['references']}\nAnswer:"
        response = self.llm_pipeline(prompt)[0]['generated_text']
        # Extract only the generated answer part
        if "Answer:" in response:
            answer = response.split("Answer:", 1)[1].strip()
        else:
            answer = response.strip()
        return answer

    def train(self, total_episodes: int, ref_reuse_multiplier: int = 5, ref_reuse_frequency: int = 10):
        global_steps = 0
        for i in tqdm(range(total_episodes), desc="Training Episodes"):
            current_query = random.choice(list(self.env.ideal_answers.keys()))

            # --- Full Multi-phase Episode (Browsing + Answering) ---
            initial_obs, info = self.env.reset(current_query, pre_browsed_references=None)
            generated_answer = self._generate_answer_from_llm(initial_obs)
            _, reward, _, _ = self.env.step(generated_answer)

            self.episode_history.append({
                "query": current_query,
                "references": self.env.current_references,
                "generated_answer": generated_answer,
                "reward": reward,
                "full_episode": True
            })

            # Prepare for PPO update
            queries_tensor = self.tokenizer(f"Question: {initial_obs['query']}\nReferences: {initial_obs['references']}\nAnswer:", return_tensors="pt", truncation=True, padding=True).input_ids.to(self.model.device)
            responses_tensor = self.tokenizer(generated_answer, return_tensors="pt", truncation=True, padding=True).input_ids.to(self.model.device)
            rewards_tensor = torch.tensor([reward], device=self.model.device)

            ppo_logs = self.ppo_trainer.step([queries_tensor], [responses_tensor], [rewards_tensor])
            global_steps += 1

            print(f"\nFull Episode {i+1}/{total_episodes} - Query: {current_query[:50]}... ")
            print(f"  Generated Answer: {generated_answer[:70]}...")
            print(f"  Reward: {reward:.4f}, PPO Metrics: {ppo_logs['ppo/loss']:.4f}")

            # --- Reference Reuse Mechanism ---
            if (i + 1) % ref_reuse_frequency == 0 and len(self.episode_history) > 0:
                print(f"\n--- Generating {ref_reuse_multiplier} Answering-Only Episodes ---")
                for _ in range(ref_reuse_multiplier):
                    # Select a random past episode for reference reuse
                    past_episode_data = random.choice(self.episode_history)
                    reused_query = past_episode_data["query"]
                    reused_references = past_episode_data["references"]

                    # Answering-only episode (bypassing browsing)
                    initial_obs_reuse, _ = self.env.reset(reused_query, pre_browsed_references=reused_references)
                    generated_answer_reuse = self._generate_answer_from_llm(initial_obs_reuse)
                    _, reward_reuse, _, _ = self.env.step(generated_answer_reuse)

                    # Prepare for PPO update for reuse episodes
                    queries_reuse_tensor = self.tokenizer(f"Question: {initial_obs_reuse['query']}\nReferences: {initial_obs_reuse['references']}\nAnswer:", return_tensors="pt", truncation=True, padding=True).input_ids.to(self.model.device)
                    responses_reuse_tensor = self.tokenizer(generated_answer_reuse, return_tensors="pt", truncation=True, padding=True).input_ids.to(self.model.device)
                    rewards_reuse_tensor = torch.tensor([reward_reuse], device=self.model.device)
                    
                    ppo_logs_reuse = self.ppo_trainer.step([queries_reuse_tensor], [responses_reuse_tensor], [rewards_reuse_tensor])
                    global_steps += 1

                    print(f"  Reuse Episode - Query: {reused_query[:50]}...")
                    print(f"    Generated Answer: {generated_answer_reuse[:70]}...")
                    print(f"    Reward: {reward_reuse:.4f}, PPO Metrics: {ppo_logs_reuse['ppo/loss']:.4f}")

            if global_steps % 50 == 0: # Save model periodically
                self.model.save_pretrained(f"./chatbot_model_step_{global_steps}")
                self.tokenizer.save_pretrained(f"./chatbot_model_step_{global_steps}")
                print(f"Model saved at step {global_steps}")

        print("\nTraining complete!")
        self.model.save_pretrained("./final_chatbot_model")
        self.tokenizer.save_pretrained("./final_chatbot_model")
        print("Final model saved.")


# Main execution
if __name__ == "__main__":
    # Simulate a user interface interaction
    print("Initializing Chatbot Trainer...")
    trainer = ChatbotRLTrainer(model_name="distilgpt2")
    print("Starting training...")
    trainer.train(total_episodes=50, ref_reuse_multiplier=3, ref_reuse_frequency=5)

    print("\n--- Chatbot Ready for Inference (Simulated) ---")
    # Load the trained model for inference
    # For simplicity, we'll use the trainer's current model directly after training
    # In a real scenario, you would load from `./final_chatbot_model`

    def interact_with_chatbot(query: str):
        print(f"User Query: {query}")
        # Browsing phase
        references = trainer.kb.retrieve_references(query)
        print(f"  Retrieved References: {references}")
        
        # Answering phase
        # Construct a dummy observation for inference
        obs_inference = {"query": query, "references": " ".join(references)}
        answer = trainer._generate_answer_from_llm(obs_inference)
        print(f"  Chatbot Answer: {answer}")
        return answer

    # Example interactions
    interact_with_chatbot("I need help resetting my password.")
    interact_with_chatbot("What are the subscription options available?")
    interact_with_chatbot("I can't log into my account, what should I do?")
    interact_with_chatbot("Tell me a joke.") # Out of domain query

