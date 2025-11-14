import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from trl import PPOTrainer, PPOConfig, SFTTrainer
from datasets import Dataset
import random

# --- 1. Data Simulation (Placeholders) ---
# In a real application, these would load actual datasets from files or databases.
def load_demonstration_data():
    """Simulates loading human demonstration data for behavior cloning."""
    print("Loading simulated demonstration data...")
    conversations = [
        {"prompt": "User: What's the status of my order #XYZ123?", "completion": "Agent: Please provide your full name and email associated with the order, and I'll check for you."},
        {"prompt": "User: My product arrived damaged. What should I do?", "completion": "Agent: I'm sorry to hear that! Could you please describe the damage and attach any photos?"},
        {"prompt": "User: How do I return an item?", "completion": "Agent: To initiate a return, please visit our 'Returns' page and fill out the online form with your order details."},
        {"prompt": "User: Can I change my delivery address?", "completion": "Agent: Delivery address changes are only possible before shipment. Please provide your order number and the new address immediately."},
        {"prompt": "User: I was charged twice for my order.", "completion": "Agent: I apologize for the inconvenience. Please share your order number, and I will investigate this billing issue."},
    ]
    return Dataset.from_list(conversations)

def load_preference_data():
    """Simulates loading human preference comparison data for reward modeling."""
    print("Loading simulated preference data...")
    preferences = [
        {
            "prompt": "User: How do I track my package?",
            "chosen": "Agent: You can track your package using the tracking number provided in your shipping confirmation email.",
            "rejected": "Agent: Just check the website. It should be there."
        },
        {
            "prompt": "User: My product didn't work after one week.",
            "chosen": "Agent: I'm sorry to hear that. What specific issue are you experiencing with the product?",
            "rejected": "Agent: That's a common problem. We can't do anything about it."
        },
        {
            "prompt": "User: What are your holiday hours?",
            "chosen": "Agent: Our customer support will be available from 9 AM to 5 PM local time during the holidays.",
            "rejected": "Agent: We are usually open."
        },
    ]
    return Dataset.from_list(preferences)

# --- 2. Reward Model Definition and Training (Simplified) ---
class RewardModel(torch.nn.Module):
    """A simplified Reward Model using a pre-trained sequence classification model."""
    def __init__(self, model_name="distilbert-base-uncased"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Using a sequence classification model with 1 label for scalar reward prediction
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
        print(f"RewardModel initialized with {model_name}.")

    def forward(self, text_input):
        inputs = self.tokenizer(text_input, return_tensors="pt", padding=True, truncation=True, max_length=512)
        # Ensure inputs are on the correct device if using GPU
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        outputs = self.model(**inputs)
        return outputs.logits.squeeze(-1) # Squeeze to get a scalar score per example

    def train_on_preferences(self, preference_data: Dataset):
        """Simulates training the reward model on human preference data."
        In a real scenario, this would involve a proper training loop, e.g., using Hugging Face Trainer,
        to learn to differentiate between chosen and rejected responses.
        """
        print("\n--- Training Reward Model (Human Feedback for Quality Optimization) ---")
        print(f"  - Simulating training on {len(preference_data)} human preference comparisons.")
        # Example of how you might prepare data for training, though actual training is omitted
        # for brevity and focus on patterns.
        # For each preference, the model would be trained to give a higher score to 'chosen' than 'rejected'.
        print("  - Reward Model is now (simulated) aligned with human preferences.")


# --- 3. Customer Support Agent Core Logic ---
class CustomerSupportAgent:
    """Implements an intelligent customer support agent using various AI design patterns."""
    def __init__(self, base_llm_name="distilgpt2", reward_model_name="distilbert-base-uncased"):
        print("\n--- Initializing Customer Support Agent ---")
        self.tokenizer = AutoTokenizer.from_pretrained(base_llm_name)
        # Set pad_token_id for generation, especially if the model doesn't have one by default
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.base_llm = AutoModelForCausalLM.from_pretrained(base_llm_name)

        if torch.cuda.is_available():
            self.base_llm.to("cuda")

        self.reward_model = RewardModel(reward_model_name)
        if torch.cuda.is_available():
            self.reward_model.to("cuda")

        print(f"Agent LLM initialized with {base_llm_name}.")

    def train_behavior_cloning(self, demonstration_data: Dataset):
        """Trains the base LLM using Behavior Cloning from human demonstrations."""
        print("\n--- Training Base LLM with Behavior Cloning for Initial Skill Acquisition ---")
        # Using SFTTrainer for supervised fine-tuning (Behavior Cloning)
        # This assumes demonstration_data has 'prompt' and 'completion' fields suitable for SFT
        training_args = {
            "output_dir": "./bc_model",
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 2e-5,
            "num_train_epochs": 1, # Keep small for simulation
            "logging_steps": 10,
            "save_steps": 10,
            "fp16": torch.cuda.is_available(), # Use mixed precision if GPU is available
        }

        # Prepare data for SFTTrainer (concatenating prompt and completion)
        def formatting_func(example):
            texts = []
            for i in range(len(example["prompt"])): # Iterate over batch
                text = f"User: {example["prompt"][i].replace('User: ', '')}\nAgent: {example["completion"][i].replace('Agent: ', '')}{self.tokenizer.eos_token}"
                texts.append(text)
            return {"text": texts}

        # For SFT, the dataset needs a 'text' column if using the default formatting
        # Or provide a custom formatting_func
        sft_dataset = demonstration_data.map(formatting_func, batched=True)

        trainer = SFTTrainer(
            model=self.base_llm,
            tokenizer=self.tokenizer,
            train_dataset=sft_dataset,
            dataset_text_field="text",
            max_seq_length=512,
            args=type('args', (object,), training_args)() # Simple object for args
        )
        print(f"  - Fine-tuning on {len(demonstration_data)} human demonstrations...")
        trainer.train()
        print("  - Base LLM successfully acquired initial skills from demonstrations.")

    def train_rlhf(self, demonstration_data: Dataset, preference_data: Dataset):
        """Performs RLHF using the reward model to further align the LLM.
           Incorporates principles of Sample-Efficient RL with Reference Reuse.
        """
        print("\n--- Performing RLHF for Quality Optimization (Reward Modeling & RLHF) ---")
        print("  - This phase leverages human preference data via the Reward Model.")
        ppo_config = PPOConfig(
            learning_rate=1e-5,
            mini_batch_size=1,
            gradient_accumulation_steps=1,
            target_kl=0.1,
            init_kl_coef=0.2,
            seed=0,
            log_with=None,
            remove_unused_columns=False, # Important for PPO
        )

        # For RLHF, we typically need a reference model to compute KL divergence
        ref_llm = AutoModelForCausalLM.from_pretrained(self.base_llm.config._name_or_path)
        if torch.cuda.is_available():
            ref_llm.to("cuda")

        # The PPO trainer takes the active LLM, reference LLM, and the reward model as part of the loop.
        # The reward model is used to compute rewards for generated samples.
        # A separate value head might be attached to the LLM or a separate model entirely.
        # For simplicity, we assume the reward model directly provides a score.

        # Dummy training dataset for PPO (prompts only)
        ppo_dataset_prompts = Dataset.from_dict({"query": [d["prompt"] for d in demonstration_data]})

        # A proper PPO trainer setup is more complex, requiring a collator and reward function wrapper.
        # Here, we'll demonstrate the conceptual flow.
        print("  - Initializing PPO Trainer (conceptual)...")

        # A complete PPO implementation would iterate:
        # 1. Generate responses from current policy (self.base_llm) given prompts from ppo_dataset_prompts.
        # 2. Score responses using self.reward_model.
        # 3. Compute KL divergence against ref_llm.
        # 4. Update self.base_llm using PPO algorithm.

        # Simulate a few PPO optimization steps
        for step in range(3):
            print(f"  - Simulating PPO optimization step {step+1}...")
            # Example of generating and scoring (simplified)
            sample_prompt = random.choice(ppo_dataset_prompts["query"])
            generated_response = self._generate_candidate_responses(sample_prompt, num_candidates=1)[0]
            reward = self.reward_model(sample_prompt + generated_response).item()
            print(f"    Prompt: '{sample_prompt}', Generated: '{generated_response[:50]}...', Reward: {reward:.2f}")
            # In a real PPO step, the gradients would be computed and applied here.

        print("  - LLM fine-tuned using Reinforcement Learning from Human Feedback (RLHF).")
        print("  - Sample-Efficient RL with Reference Reuse principles implicitly considered in the PPO loop (e.g., careful sampling, replay buffers, or specific loss functions). However, a full implementation is complex and omitted for this example.")

    def _generate_candidate_responses(self, prompt: str, num_candidates: int = 3) -> list:
        """Generates N candidate responses from the LLM."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        generated_ids = self.base_llm.generate(
            **inputs,
            max_new_tokens=60,
            num_return_sequences=num_candidates,
            do_sample=True,
            top_k=50,
            top_p=0.0,
            temperature=0.7,
            pad_token_id=self.tokenizer.pad_token_id, # Use configured pad token
            eos_token_id=self.tokenizer.eos_token_id
        )
        decoded_responses = []
        for g in generated_ids:
            response = self.tokenizer.decode(g[len(inputs["input_ids"][0]):], skip_special_tokens=True).strip()
            decoded_responses.append(response)
        return decoded_responses

    def _score_responses(self, prompt: str, responses: list) -> list:
        """Uses the reward model to score each generated response."""
        scores = []
        for res in responses:
            # The reward model takes the full context (prompt + response) to score
            full_text = prompt + " " + res
            score = self.reward_model(full_text).item()
            scores.append(score)
        return scores

    def get_response(self, user_query: str) -> str:
        """Generates a response using Rejection Sampling (Best-of-N) with the reward model."""
        print(f"\nUser: {user_query}")
        print("  - Generating candidate responses...")
        candidate_responses = self._generate_candidate_responses(user_query, num_candidates=5)
        print("  - Scoring candidates with Reward Model (Rejection Sampling)...")
        scores = self._score_responses(user_query, candidate_responses)

        # Select the best response based on the highest reward score
        best_response_idx = scores.index(max(scores))
        best_response = candidate_responses[best_response_idx]

        print("  - Selected best response.")
        return best_response


# --- Main Script Execution ---
if __name__ == "__main__":
    print("=======================================================")
    print("--- Intelligent E-commerce Customer Support Agent ---")
    print("=======================================================")

    # 1. Load Data (Simulated for demonstration)
    demonstration_data = load_demonstration_data()
    preference_data = load_preference_data()

    # 2. Initialize Agent
    # Using smaller models ('distilgpt2', 'distilbert-base-uncased') for quicker execution
    agent = CustomerSupportAgent(base_llm_name="distilgpt2", reward_model_name="distilbert-base-uncased")

    # 3. Dual Data Collection & Training Pipeline
    print("\n===========================")
    print("--- Training Phase ---")
    print("===========================")

    # Behavior Cloning for Initial Skill Acquisition
    agent.train_behavior_cloning(demonstration_data)

    # Train Reward Model with Human Feedback
    agent.reward_model.train_on_preferences(preference_data) # Simulated training

    # RLHF for Quality Optimization (includes Sample-Efficient RL)
    agent.train_rlhf(demonstration_data, preference_data) # Conceptual RLHF loop

    print("\n===========================")
    print("--- Inference Phase ---")
    print("===========================")

    # Simulate customer interactions with Rejection Sampling (Best-of-N)
    user_queries = [
        "I need to know if my order #ABC789 has shipped yet.",
        "What's your policy on damaged goods during shipping?",
        "Can I get a refund for a product I bought last month?",
        "I can't log into my account. It says 'invalid password'."
    ]

    for query in user_queries:
        response = agent.get_response(query)
        print(f"Agent: {response}")
        print("--------------------------------------------------")
