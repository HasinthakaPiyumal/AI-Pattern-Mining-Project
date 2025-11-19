
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from trl import SFTTrainer, PPOConfig, PPOTrainer, RewardTrainer
from datasets import Dataset
import random

# --- Configuration --- #
MODEL_NAME = "distilgpt2" # A smaller model for demonstration purposes
REWARD_MODEL_NAME = "distilbert-base-uncased" # For the reward model

class LLMCustomerSupportAssistant:
    def __init__(self, model_name=MODEL_NAME, reward_model_name=REWARD_MODEL_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.llm = AutoModelForCausalLM.from_pretrained(model_name)
        self.reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
        self.reward_model = None # Will be trained later
        self.ppo_trainer = None

        self.demonstrations = []
        self.comparison_data = []
        self.successful_interactions = []

    def _generate_synthetic_data(self):
        # Simulate expert demonstrations for Behavior Cloning
        self.demonstrations = [
            {"prompt": "My order #12345 is late.", "completion": "I apologize for the delay. Let me check the status of your order #12345. It appears there was a shipping anomaly. We've initiated an inquiry and will update you within 24 hours."}, 
            {"prompt": "How do I return an item?", "completion": "To return an item, please visit our 'Returns' page on the website, enter your order details, and follow the instructions to generate a return label. You have 30 days from delivery to return most items."},
            {"prompt": "I received a damaged product.", "completion": "I'm very sorry to hear that! Please provide your order number and a photo of the damaged item. We'll arrange for a replacement to be sent out immediately and provide a return label for the damaged product."},
            {"prompt": "Can I change my shipping address?", "completion": "Unfortunately, once an order is placed, we cannot guarantee changes to the shipping address. Please contact us immediately with your order number, and we'll see if it's still possible before shipment."},
            {"prompt": "What's your refund policy?", "completion": "Our refund policy states that once we receive and inspect your returned item, your refund will be processed within 5-7 business days to your original payment method."}
        ]

        # Simulate human comparison data for Reward Model training
        # (prompt, chosen_response, rejected_response)
        self.comparison_data = [
            {"prompt": "My order is lost.",
             "chosen": "I'm sorry to hear that. Please provide your order number so I can investigate.",
             "rejected": "That's bad. What should I do?"},
            {"prompt": "I need help with a product.",
             "chosen": "Could you please tell me which product you're referring to and what issue you're facing?",
             "rejected": "Products are sometimes tricky."},
            {"prompt": "When will my new laptop arrive?",
             "chosen": "Please provide your order number, and I will check the estimated delivery date for your laptop.",
             "rejected": "It will arrive when it arrives."},
            {"prompt": "How do I track my package?",
             "chosen": "You can track your package by clicking the tracking link in your shipping confirmation email or by entering your order number on our 'Track Order' page.",
             "rejected": "Go to the post office website."},
            {"prompt": "Can I cancel my subscription?",
             "chosen": "To cancel your subscription, please log in to your account, navigate to 'My Subscriptions', and follow the cancellation steps.",
             "rejected": "Just stop using it."}
        ]

    def train_behavior_cloning(self):
        print("\n--- Starting Behavior Cloning Training ---")
        self._generate_synthetic_data() # Ensure data is available
        bc_dataset = Dataset.from_list([{"text": d["prompt"] + " " + d["completion"]} for d in self.demonstrations])

        trainer = SFTTrainer(
            model=self.llm,
            tokenizer=self.tokenizer,
            train_dataset=bc_dataset,
            dataset_text_field="text",
            max_seq_length=128,
            args=transformers.TrainingArguments(
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                warmup_steps=2,
                learning_rate=2e-4,
                num_train_epochs=1,
                logging_steps=1,
                output_dir="./bc_training",
                optim="adamw_torch",
            ),
        )
        trainer.train()
        print("--- Behavior Cloning Training Complete ---")

    def train_reward_model(self):
        print("\n--- Starting Reward Model Training ---")
        # For RewardTrainer, we need a specific dataset format: {'prompt', 'chosen', 'rejected'}
        rm_dataset = Dataset.from_list(self.comparison_data)

        # Load a base model for the reward model
        # A simple classification head can be added on top of a pre-trained language model
        class CustomRewardModel(torch.nn.Module):
            def __init__(self, base_model_name, num_labels=1):
                super().__init__()
                self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
                # In a real scenario, you'd use AutoModelForSequenceClassification or a custom head
                # For demonstration, we'll just use the base LM to process inputs.
                # TRL's RewardTrainer expects a model with a 'score' output.
                # We'll adapt by using a simple linear layer for scoring.
                self.score_head = torch.nn.Linear(self.base_model.config.hidden_size, num_labels)

            def forward(self, input_ids, attention_mask=None):
                outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                # Take the last hidden state of the last token as input to the scoring head
                last_hidden_state = outputs.hidden_states[-1]
                # Assuming the last token's hidden state represents the sequence
                score = self.score_head(last_hidden_state[:, -1, :])
                return score

        # For a more robust Reward Model, one would typically use a specialized model
        # like AutoModelForSequenceClassification with a single output neuron, or a custom architecture.
        # Given the `trl.RewardTrainer` expects a `forward` method that outputs scores,
        # we simulate a simple scoring mechanism on top of a causal LM for this example.
        # In a real scenario, `distilbert-base-uncased` would be used with a classification head.
        
        # Let's use a simpler dummy reward model for `trl.RewardTrainer` compatibility
        # as defining a proper `AutoModelForSequenceClassification` requires more setup
        # and is out of scope for a quick demo.
        # Instead, we will simulate the reward model's function when needed.
        
        print("Reward model training simulated. In a real application, a RewardTrainer would be used here.")
        
        # For the purpose of this demo, we'll create a dummy 'reward_model' object
        # that can be called to give scores, simulating a trained model.
        # In practice, `trl.RewardTrainer` would train a specific model for this.
        class DummyRewardModel:
            def __init__(self, tokenizer):
                self.tokenizer = tokenizer
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis", 
                    model="distilbert-base-uncased", 
                    tokenizer="distilbert-base-uncased", 
                    truncation=True, 
                    max_length=128
                )

            def __call__(self, texts):
                # Simulate scoring: higher score for 'POSITIVE' sentiment, lower for 'NEGATIVE'
                scores = []
                for text in texts:
                    result = self.sentiment_pipeline(text)[0]
                    # We'll map POSITIVE to ~1.0, NEGATIVE to ~-1.0
                    score = result['score'] if result['label'] == 'POSITIVE' else (1 - result['score']) * -1
                    scores.append(score)
                return torch.tensor(scores)

        self.reward_model = DummyRewardModel(self.tokenizer)
        print("--- Reward Model (simulated) Ready ---")


    def train_rlhf(self):
        print("\n--- Starting RLHF Training ---")
        if not self.reward_model:
            self.train_reward_model()

        # PPO config
        ppo_config = PPOConfig(
            learning_rate=1e-5,
            batch_size=2,
            forward_batch_size=2,
            ppo_epochs=1,
            gradient_accumulation_steps=1,
            target_kl=0.1,
            log_with=None,
            output_dir="./ppo_training",
        )

        # Dummy PPO dataset (in a real scenario, this would come from interactions)
        ppo_dataset = Dataset.from_list([{"query": d["prompt"]} for d in self.demonstrations])

        def collator(data):
            return dict((key, [d[key] for d in data]) for key in data[0])

        # PPO Trainer
        self.ppo_trainer = PPOTrainer(
            ppo_config,
            self.llm,
            ref_model=None, # Use the initial LLM as the reference model
            tokenizer=self.tokenizer,
            dataset=ppo_dataset,
            data_collator=collator,
        )

        # Simulate the RLHF loop (simplified)
        for epoch in range(1):
            for batch in self.ppo_trainer.dataloader:
                query_tensors = [self.tokenizer(q, return_tensors="pt").input_ids[0] for q in batch["query"]]

                # Generate responses
                response_tensors = []
                for query in query_tensors:
                    gen_len = random.randint(20, 40)
                    response = self.ppo_trainer.generate(query, max_new_tokens=gen_len,
                                                        num_return_sequences=1,
                                                        do_sample=True, top_k=50, top_p=0.95)[0]
                    response_tensors.append(response[len(query):]) # Get only the generated part

                # Get rewards from the reward model
                texts = [self.tokenizer.decode(torch.cat([q, r])) for q, r in zip(query_tensors, response_tensors)]
                rewards = self.reward_model(texts) # Use the dummy reward model

                # Train with PPO
                stats = self.ppo_trainer.step(query_tensors, response_tensors, rewards)
                self.ppo_trainer.log_stats(stats, batch, rewards)

        print("--- RLHF Training Complete ---")

    def generate_response(self, prompt, model_to_use=None):
        if model_to_use is None:
            model_to_use = self.llm

        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=128)
        outputs = model_to_use.generate(
            **inputs,
            max_new_tokens=50,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from the response
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        return response

    def generate_response_with_rejection_sampling(self, prompt, num_samples=3):
        print(f"Generating response for '{prompt}' with rejection sampling...")
        if not self.reward_model:
            print("Reward model not trained. Falling back to standard generation.")
            return self.generate_response(prompt)

        best_response = ""
        highest_score = -float('inf')

        candidate_responses = []
        for _ in range(num_samples):
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=128)
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=50,
                num_return_sequences=1,
                do_sample=True, 
                top_k=50, 
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id
            )
            response_full = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response_full[len(prompt):].strip()
            candidate_responses.append(response)

        # Score candidates with the reward model
        scores = self.reward_model([prompt + " " + res for res in candidate_responses])

        for i, score in enumerate(scores):
            if score > highest_score:
                highest_score = score
                best_response = candidate_responses[i]
        
        print(f"Selected best response (score: {highest_score:.2f}): {best_response}")
        return best_response

    def save_successful_interaction(self, prompt, response, feedback_score):
        print(f"Saving successful interaction for reuse: Prompt='{prompt[:30]}...', Score={feedback_score}")
        self.successful_interactions.append({
            "prompt": prompt,
            "response": response,
            "feedback_score": feedback_score
        })
        # In a real system, this would persist to a database or file

    def load_successful_interactions(self):
        print(f"Loaded {len(self.successful_interactions)} successful interactions.")
        return self.successful_interactions

    def simulate_dual_data_collection(self):
        print("\n--- Simulating Dual Data Collection ---")
        # This method is more conceptual within the code, as data is generated upfront
        # In a real scenario, this would involve active data collection pipelines.
        # Demonstrations are used for BC.
        # Comparisons (prompt, chosen, rejected) are used for RM.
        print(f"Collected {len(self.demonstrations)} demonstrations for skill-building.")
        print(f"Collected {len(self.comparison_data)} comparisons for preference alignment.")
        print("--- Dual Data Collection Simulation Complete ---")


    def run_full_training_pipeline(self):
        print("\n--- Running Full LLM Assistant Training Pipeline ---")
        self.simulate_dual_data_collection()
        self.train_behavior_cloning()
        self.train_reward_model()
        self.train_rlhf() # Uses the trained reward model
        print("--- Full Training Pipeline Complete ---")

    def simulate_customer_interaction(self, prompt):
        print(f"\n--- Customer Interaction Simulation ---")
        print(f"Customer: {prompt}")
        # Use the PPO-trained model if available, otherwise the BC-trained one.
        # And apply rejection sampling.
        response = self.generate_response_with_rejection_sampling(prompt)
        print(f"Assistant: {response}")
        # Simulate human feedback for future sample-efficient RL
        feedback = random.uniform(0.5, 1.0) # Assume positive feedback for this demo
        self.save_successful_interaction(prompt, response, feedback)
        print(f"(Simulated human feedback: {feedback:.2f})")
        print("--- Interaction End ---")

# --- Main Execution --- #
if __name__ == "__main__":
    # Suppress warnings from transformers library for cleaner output
    transformers.logging.set_verbosity_error()

    assistant = LLMCustomerSupportAssistant()

    # Run the full training pipeline
    assistant.run_full_training_pipeline()

    # Simulate some customer interactions
    print("\n*** Simulating Customer Support Interactions ***")
    assistant.simulate_customer_interaction("My parcel hasn't arrived yet, order number is #98765.")
    assistant.simulate_customer_interaction("I want to know about your warranty policy.")
    assistant.simulate_customer_interaction("The item I received is not what I ordered. It's a blue shirt instead of a red one.")
    assistant.simulate_customer_interaction("How can I contact a human agent?")

    # Demonstrate sample-efficient RL with reference reuse (conceptually)
    print("\n--- Reviewing Sample-Efficient RL Data ---")
    reused_data = assistant.load_successful_interactions()
    if reused_data:
        print(f"Example of a successful interaction for reuse: {reused_data[0]}")
    else:
        print("No successful interactions collected yet for reuse.")

    print("\nAI Customer Support Assistant demo completed.")
