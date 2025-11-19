import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from trl import SFTTrainer, PPOTrainer, PPOConfig
from datasets import Dataset
import pandas as pd
import random

# 1. LLMAgent Class (Core LLM and Response Generation)
class LLMAgent:
    def __init__(self, model_name="gpt2", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.device = device
        print(f"LLMAgent initialized with {model_name} on {self.device}")

    def generate_response(self, prompt, max_new_tokens=50, num_return_sequences=1):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.pad_token_id
        )
        return [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

# 2. RewardModel Class (Human Feedback Integration and Reward Modeling)
class RewardModel:
    def __init__(self, model_name="distilbert-base-uncased-sentiment", device="cuda" if torch.cuda.is_available() else "cpu"):
        # Using a sentiment analysis pipeline as a proxy for a learned reward model
        self.sentiment_pipeline = pipeline("sentiment-analysis", model=model_name, device=0 if device == "cuda" else -1)
        print(f"RewardModel initialized with {model_name} on {device}")

    def get_reward(self, response: str) -> float:
        # A simplistic reward: positive sentiment gets a higher reward
        result = self.sentiment_pipeline(response)[0]
        if result['label'] == 'POSITIVE':
            return result['score']  # Use confidence as reward
        elif result['label'] == 'NEGATIVE':
            return -result['score'] # Penalize negative sentiment
        else:
            return 0.0 # Neutral sentiment

# 3. DataCollector Class (Data Collection and Management)
class DataCollector:
    def __init__(self):
        self.human_demonstrations = [] # List of {'prompt': '...', 'response': '...'} dicts
        self.human_preferences = []    # List of {'response_a': '...', 'response_b': '...', 'preference': 'A' or 'B'} dicts
        self.reference_reuse_data = [] # List of {'context': '...', 'successful_action': '...'} dicts
        print("DataCollector initialized")

    def add_demonstration(self, prompt, response):
        self.human_demonstrations.append({'prompt': prompt, 'response': response})
        print(f"Added demonstration: '{prompt}' -> '{response[:30]}...' ")

    def add_preference(self, response_a, response_b, preference):
        # preference can be 'A' if response_a is better, 'B' if response_b is better
        self.human_preferences.append({'response_a': response_a, 'response_b': response_b, 'preference': preference})
        print(f"Added preference: A='{response_a[:30]}...', B='{response_b[:30]}...', Pref={preference}")

    def add_reference(self, context, successful_action):
        self.reference_reuse_data.append({'context': context, 'successful_action': successful_action})
        print(f"Added reference: Context='{context[:30]}...', Action='{successful_action[:30]}...' ")

# 4. Trainer Class (Training Orchestration)
class Trainer:
    def __init__(self, llm_agent: LLMAgent, reward_model: RewardModel, data_collector: DataCollector):
        self.llm_agent = llm_agent
        self.reward_model = reward_model
        self.data_collector = data_collector
        print("Trainer initialized")

    def sft_train(self):
        if not self.data_collector.human_demonstrations:
            print("No human demonstrations available for SFT training.")
            return
        print("Starting Behavior Cloning (SFT) training...")
        # In a real scenario, this would use trl.SFTTrainer
        # For demonstration, we'll just acknowledge the training.
        # Example: SFTTrainer(model=self.llm_agent.model, 
        #                      tokenizer=self.llm_agent.tokenizer,
        #                      train_dataset=Dataset.from_pandas(pd.DataFrame(self.data_collector.human_demonstrations)),
        #                      dataset_text_field="prompt",
        #                      ...).train()
        print(f"Simulated SFT training completed on {len(self.data_collector.human_demonstrations)} demonstrations.")

    def train_reward_model(self):
        if not self.data_collector.human_preferences:
            print("No human preferences available for Reward Model training.")
            return
        print("Starting Reward Model training...")
        # In a real scenario, this would train a separate model on preferences.
        # For demonstration, we'll just acknowledge the training.
        print(f"Simulated Reward Model training completed on {len(self.data_collector.human_preferences)} preferences.")

    def rlhf_train(self, prompt_dataset: Dataset):
        print("Starting Reinforcement Learning from Human Feedback (RLHF) training...")
        # In a real scenario, this would use trl.PPOTrainer
        # For demonstration, we'll just acknowledge the training.
        # ppo_config = PPOConfig(...)
        # ppo_trainer = PPOTrainer(
        #     config=ppo_config,
        #     model=self.llm_agent.model,
        #     ref_model=None, # Or a copy of the LLM for stability
        #     tokenizer=self.llm_agent.tokenizer,
        #     dataset=prompt_dataset,
        # )
        # ppo_trainer.train()
        print("Simulated RLHF training completed.")

    def rejection_sampling(self, prompt, num_candidates=5):
        print(f"Performing rejection sampling for prompt: '{prompt}'")
        candidate_responses = self.llm_agent.generate_response(prompt, num_return_sequences=num_candidates)
        scored_responses = []
        for res in candidate_responses:
            reward = self.reward_model.get_reward(res)
            scored_responses.append({'response': res, 'reward': reward})
            print(f"  Candidate: '{res[:50]}...', Reward: {reward:.2f}")
        
        # Select the response with the highest reward
        best_response = max(scored_responses, key=lambda x: x['reward'])
        print(f"Selected best response with reward {best_response['reward']:.2f}: '{best_response['response'][:70]}...'")
        return best_response['response']

# 5. Main Application Entry Point and Simulation
if __name__ == "__main__":
    print("Initializing Intelligent Customer Support Agent...")
    llm_agent = LLMAgent()
    reward_model = RewardModel()
    data_collector = DataCollector()
    trainer = Trainer(llm_agent, reward_model, data_collector)

    # --- Simulate Data Collection ---
    print("\n--- Simulating Data Collection ---")
    data_collector.add_demonstration(
        "My internet is not working.",
        "Please try restarting your router and modem. If that doesn't work, we can troubleshoot further."
    )
    data_collector.add_demonstration(
        "I want to change my subscription plan.",
        "Sure, I can help with that. Could you please confirm your account details?"
    )

    response1 = llm_agent.generate_response("How do I reset my password?")[0]
    response2 = llm_agent.generate_response("How do I reset my password?")[0] + " You should try to remember it better next time."
    data_collector.add_preference(response1, response2, 'A') # Assuming response1 is better

    data_collector.add_reference(
        "User asked about billing issue and agent provided refund instructions.",
        "Provided step-by-step refund process and confirmed amount."
    )

    # --- Simulate Training Phases ---
    print("\n--- Simulating Training Phases ---")
    trainer.sft_train() # Behavior Cloning
    trainer.train_reward_model() # Reward Model Training

    # Prepare a dummy dataset for RLHF (in a real scenario, this would be a prompt dataset)
    rlhf_prompts = pd.DataFrame([{"prompt": "What is my bill amount?"}, {"prompt": "I need technical support."}])
    trainer.rlhf_train(Dataset.from_pandas(rlhf_prompts)) # RLHF

    # --- Demonstrate Adaptive Response Generation ---
    print("\n--- Demonstrating Adaptive Response Generation ---")
    user_query = "I have an urgent issue with my service, can you help?"

    # Using standard LLM generation first
    print(f"\nLLM Agent (standard generation) for query: '{user_query}'")
    standard_response = llm_agent.generate_response(user_query)[0]
    print(f"Response: '{standard_response}'")
    print(f"Reward Score: {reward_model.get_reward(standard_response):.2f}")

    # Using Rejection Sampling with Reward Model
    print(f"\nLLM Agent (Rejection Sampling) for query: '{user_query}'")
    rejection_sampled_response = trainer.rejection_sampling(user_query, num_candidates=3)
    print(f"Final Rejection Sampled Response: '{rejection_sampled_response}'")
    print(f"Reward Score: {reward_model.get_reward(rejection_sampled_response):.2f}")

    print("\nSimulation complete.")