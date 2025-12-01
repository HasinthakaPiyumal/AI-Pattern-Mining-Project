import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, pipeline
from datasets import Dataset
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from fastapi import FastAPI
import uvicorn
import random
import numpy as np


class Config:
    SFT_MODEL_NAME = "gpt2"
    RM_MODEL_NAME = "bert-base-uncased"
    MODEL_SAVE_PATH = "./models"
    DATA_SAVE_PATH = "./data"
    NUM_PREFERENCE_SAMPLES = 100
    RM_TRAIN_EPOCHS = 3
    RLHF_TRAIN_STEPS = 10


class DataCollector:
    def simulate_human_preferences(self, num_samples=Config.NUM_PREFERENCE_SAMPLES):
        preference_data = []
        for i in range(num_samples):
            query = f"customer query {i + 1}"
            response_A = f"This is a helpful response A to query {i + 1}."
            response_B = f"This is a less helpful response B to query {i + 1}."
            preferred_response = random.choice([response_A, response_B])
            
            if i % 3 == 0: # Simulate some human-like responses
                response_B = f"This is an even better, more empathetic response B to query {i + 1}."
                preferred_response = response_B

            preference_data.append({"query": query, "response_A": response_A, "response_B": response_B, "preferred_response": preferred_response})
        print(f"Simulated {num_samples} human preference samples.")
        return preference_data


class RewardModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(Config.RM_MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(Config.RM_MODEL_NAME, num_labels=2)
        self.pipeline = None

    def _preprocess_function(self, examples):
        # For training, we'll simplify: just tokenize the preferred response as positive, other as negative.
        # In a real RM, you'd train on pairs and predict preference probability.
        tokenized_inputs = []
        labels = []

        for i in range(len(examples["query"])):
            query = examples["query"][i]
            response_A = examples["response_A"][i]
            response_B = examples["response_B"][i]
            preferred = examples["preferred_response"][i]

            # Simulate training logic for binary classification (preferred vs not preferred)
            if preferred == response_A:
                positive_text = f"{query} [SEP] {response_A}"
                negative_text = f"{query} [SEP] {response_B}"
                tokenized_inputs.append(self.tokenizer(positive_text, truncation=True, padding="max_length", max_length=128))
                labels.append(1)  # Preferred
                tokenized_inputs.append(self.tokenizer(negative_text, truncation=True, padding="max_length", max_length=128))
                labels.append(0)  # Not preferred
            else:
                positive_text = f"{query} [SEP] {response_B}"
                negative_text = f"{query} [SEP] {response_A}"
                tokenized_inputs.append(self.tokenizer(positive_text, truncation=True, padding="max_length", max_length=128))
                labels.append(1)  # Preferred
                tokenized_inputs.append(self.tokenizer(negative_text, truncation=True, padding="max_length", max_length=128))
                labels.append(0)  # Not preferred

        flattened_input_ids = [item["input_ids"] for item in tokenized_inputs]
        flattened_attention_mask = [item["attention_mask"] for item in tokenized_inputs]
        
        return {
            "input_ids": flattened_input_ids,
            "attention_mask": flattened_attention_mask,
            "labels": labels
        }

    def train(self, preference_data):
        print("Training Reward Model...")
        # Convert list of dicts to Hugging Face Dataset
        dataset = Dataset.from_list(preference_data)
        tokenized_dataset = dataset.map(self._preprocess_function, batched=True, remove_columns=dataset.column_names)
        tokenized_dataset.set_format("torch")

        # This is a highly simplified training loop. In reality, you'd use Trainer or custom loop.
        # Here, we just pretend to train.
        print(f"Simulating RM training for {Config.RM_TRAIN_EPOCHS} epochs.")
        for epoch in range(Config.RM_TRAIN_EPOCHS):
            print(f"RM Epoch {epoch+1}/{Config.RM_TRAIN_EPOCHS}")
            # Dummy optimization step
            # optimizer.step(); scheduler.step(); model.zero_grad()
        
        # In a real scenario, you'd save the trained model weights
        self.model.save_pretrained(f"{Config.MODEL_SAVE_PATH}/reward_model")
        self.tokenizer.save_pretrained(f"{Config.MODEL_SAVE_PATH}/reward_model")
        self.pipeline = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer, device=0 if torch.cuda.is_available() else -1)
        print("Reward Model training simulated and saved.")

    def predict_reward(self, query, response):
        if not self.pipeline:
            self.pipeline = pipeline("sentiment-analysis", model=f"{Config.MODEL_SAVE_PATH}/reward_model", tokenizer=Config.RM_MODEL_NAME, device=0 if torch.cuda.is_available() else -1)
        
        # The RM pipeline is designed for classification (e.g., pos/neg). 
        # For a reward model, we'd typically want a scalar score. 
        # Here, we'll map the classification output to a simple reward.
        text = f"{query} [SEP] {response}"
        # Dummy reward: higher score for preferred responses based on label '1'
        result = self.pipeline(text)[0]
        # Assuming 'LABEL_1' means preferred (high reward), 'LABEL_0' means not preferred (low reward)
        if result['label'] == 'LABEL_1':
            return result['score']  # Use confidence as reward
        else:
            return 1 - result['score'] # Invert confidence for the other label


class ChatbotRLHFTrainer:
    def __init__(self, reward_model):
        self.tokenizer = AutoTokenizer.from_pretrained(Config.SFT_MODEL_NAME)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

        self.sft_model = AutoModelForCausalLM.from_pretrained(Config.SFT_MODEL_NAME)
        self.sft_model.resize_token_embeddings(len(self.tokenizer))

        self.reward_model = reward_model
        self.rlhf_model = None

    def load_sft_model(self, model_path=None):
        # In a real scenario, you'd load a fully trained SFT model
        print(f"Loading initial SFT model from {Config.SFT_MODEL_NAME}")
        # For this simulation, self.sft_model is already loaded

    def train_rlhf(self, preference_data):
        print("Starting RLHF training...")

        # Prepare the model for PPO
        self.rlhf_model = AutoModelForCausalLMWithValueHead.from_pretrained(self.sft_model)
        self.rlhf_model.config.pad_token_id = self.tokenizer.pad_token_id

        ppo_config = PPOConfig(
            learning_rate=1e-5,
            mini_batch_size=1,
            batch_size=1,
            gradient_accumulation_steps=1,
            ppo_epochs=1,
            init_kl_coef=0.01,
            target_kl=0.1,
            log_with=None,
            seed=0,
        )

        ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=self.rlhf_model,
            ref_model=None, # In a real scenario, this would be the SFT model for KL divergence
            tokenizer=self.tokenizer,
        )

        # Simulate a small dataset for PPO training
        queries = [d["query"] for d in preference_data[:Config.RLHF_TRAIN_STEPS]]
        query_tensors = [self.tokenizer(q, return_tensors="pt").input_ids.squeeze() for q in queries]

        for step in range(Config.RLHF_TRAIN_STEPS):
            print(f"RLHF Training Step {step+1}/{Config.RLHF_TRAIN_STEPS}")
            query_tensor = query_tensors[step].to(self.rlhf_model.device)

            # Generate response from current policy
            generation_kwargs = {
                "min_new_tokens": 5,
                "max_new_tokens": 20,
                "temperature": 0.7,
                "top_k": 0.0,
                "top_p": 1.0,
                "do_sample": True,
                "pad_token_id": self.tokenizer.pad_token_id
            }
            response_tensor = ppo_trainer.generate(query_tensor, **generation_kwargs)
            response_text = self.tokenizer.decode(response_tensor[0][len(query_tensor):], skip_special_tokens=True)
            
            # Get reward from RM
            reward = self.reward_model.predict_reward(queries[step], response_text)
            print(f"  Generated response: '{response_text}' | Reward: {reward:.2f}")
            rewards = torch.tensor([reward], device=self.rlhf_model.device)

            # PPO step (simplified: just log for this example)
            # In a real scenario, this would involve computing advantages, value loss, policy loss
            # For a proper PPO step: ppo_trainer.step([query_tensor], [response_tensor], rewards)
            # For this example, we'll just simulate the update.
            print("  Simulating PPO optimization step...")

        self.rlhf_model.save_pretrained(f"{Config.MODEL_SAVE_PATH}/rlhf_chatbot")
        self.tokenizer.save_pretrained(f"{Config.MODEL_SAVE_PATH}/rlhf_chatbot")
        print("RLHF training simulated and chatbot model saved.")


class ChatbotAPI:
    def __init__(self):
        self.app = FastAPI()
        self.tokenizer = None
        self.model = None
        self._load_rlhf_model()
        self._setup_routes()

    def _load_rlhf_model(self):
        print("Loading RLHF-tuned chatbot model for inference...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(f"{Config.MODEL_SAVE_PATH}/rlhf_chatbot")
            if self.tokenizer.pad_token is None:
                self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            self.model = AutoModelForCausalLM.from_pretrained(f"{Config.MODEL_SAVE_PATH}/rlhf_chatbot")
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            print("RLHF chatbot model loaded successfully.")
        except Exception as e:
            print(f"Could not load RLHF model, using base SFT model for demonstration: {e}")
            self.tokenizer = AutoTokenizer.from_pretrained(Config.SFT_MODEL_NAME)
            if self.tokenizer.pad_token is None:
                self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            self.model = AutoModelForCausalLM.from_pretrained(Config.SFT_MODEL_NAME)
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def _setup_routes(self):
        @self.app.post("/chat")
        async def chat(query: str):
            inputs = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
            
            generation_kwargs = {
                "min_new_tokens": 10,
                "max_new_tokens": 50,
                "temperature": 0.7,
                "top_k": 50,
                "top_p": 0.95,
                "do_sample": True,
                "pad_token_id": self.tokenizer.pad_token_id
            }
            
            outputs = self.model.generate(inputs["input_ids"], **generation_kwargs)
            response = self.tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
            return {"response": response.strip()}

    def run(self):
        print("Starting Chatbot API...")
        uvicorn.run(self.app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    # Ensure directories exist
    import os
    os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(Config.DATA_SAVE_PATH, exist_ok=True)

    # 1. Data Collection
    data_collector = DataCollector()
    preference_data = data_collector.simulate_human_preferences()

    # 2. Reward Model Training
    reward_model = RewardModel()
    reward_model.train(preference_data)

    # 3. Chatbot RLHF Fine-tuning
    # In a real scenario, you would first perform SFT (Supervised Fine-Tuning) on your base LLM
    # before passing it to the RLHF trainer. Here, we directly use the base LLM as SFT start.
    rlhf_trainer = ChatbotRLHFTrainer(reward_model)
    rlhf_trainer.load_sft_model() # Placeholder for actual SFT model loading
    rlhf_trainer.train_rlhf(preference_data)

    # 4. Chatbot Inference/Deployment
    chatbot_api = ChatbotAPI()
    chatbot_api.run()
