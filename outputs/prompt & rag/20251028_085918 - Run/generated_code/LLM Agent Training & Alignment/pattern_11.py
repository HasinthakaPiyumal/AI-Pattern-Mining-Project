class MockTokenizer:
    """A mock tokenizer to simulate tokenization without external libraries."""
    def encode(self, text):
        return [ord(c) for c in text] # Simple mock encoding
    def decode(self, tokens):
        return "".join([chr(t) for t in tokens]) # Simple mock decoding

class MockModel:
    """A mock large language model to simulate an LLM."""
    def __init__(self, name="base"):
        self.name = name
        self.weights = {} # Placeholder for weights
    def save_pretrained(self, path):
        print(f"Mock: Saving model {self.name} to {path}")
    def generate(self, input_ids, num_return_sequences=1, max_length=50):
        # Mock generation: just repeats input or simple placeholder logic
        generated_sequence = []
        if input_ids and input_ids[0]:
            # Simulate generating new tokens based on input
            base_token = input_ids[0][0]
            for _ in range(max_length):
                generated_sequence.append(base_token + (len(generated_sequence) % 5)) # Simple pattern
        else:
            # Generate some default tokens if no input
            generated_sequence = [72, 101, 108, 108, 111, 32, 87, 111, 114, 108, 100]

        return [generated_sequence[:max_length]] * num_return_sequences

class MockSFTTrainer:
    """A mock trainer for Supervised Fine-Tuning (Behavior Cloning)."""
    def __init__(self, model, tokenizer, train_dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
    def train(self):
        print("Mock: Performing Supervised Fine-Tuning (Behavior Cloning)...")
        for i, example in enumerate(self.train_dataset):
            if i >= 2: break # Process a few examples conceptually
            print(f"Mock: Processing SFT example: {example['query']} -> {example['response']}")
        print("Mock: SFT training complete.")
        return self.model

class MockRewardModel(MockModel):
    """A mock reward model to simulate evaluating response quality."""
    def __init__(self, base_model, name="reward"):
        super().__init__(name=name)
        self.base_model = base_model # Reward model often built on base LLM

    def __call__(self, query_tokens, response_tokens):
        # Mock reward: longer responses get higher scores, simple heuristic
        # In a real scenario, this would be a neural network prediction
        score = sum(response_tokens) / (len(response_tokens) + 1e-6) if response_tokens else 0.0
        return score

    def train(self, preference_data):
        print("Mock: Training Reward Model...")
        for i, example in enumerate(preference_data):
            if i >= 2: break # Process a few examples conceptually
            print(f"Mock: Processing RM preference: {example['query']} | {example['response_a']} vs {example['response_b']}")
        print("Mock: Reward Model training complete.")

class MockPPOTrainer:
    """A mock trainer for Reinforcement Learning from Human Feedback (PPO)."""
    def __init__(self, model, ref_model, tokenizer, reward_model, dataset):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.reward_model = reward_model
        self.dataset = dataset
    def train(self):
        print("Mock: Performing RLHF (PPO training)...")
        for i, example in enumerate(self.dataset):
            if i >= 2: break # Simulate a few RLHF steps
            print(f"Mock: RLHF iteration for query: {example['query']}")
            # In a real scenario: generate response, get reward from RM, compute loss, update policy
        print("Mock: RLHF training complete.")
        return self.model

# --- Data Simulation Functions ---
def load_demonstration_data():
    """Simulates loading human demonstration data."""
    print("Mock: Loading demonstration data...")
    return [
        {"query": "My internet is not working.", "response": "Please check your router and modem connections."},
        {"query": "How do I reset my password?", "response": "You can reset your password on the login page by clicking 'Forgot Password'."},
        {"query": "I need help with billing.", "response": "Could you please provide your account number for billing assistance?"},
    ]

def load_preference_data():
    """Simulates loading human preference comparison data."""
    print("Mock: Loading human preference data...")
    return [
        {"query": "Slow internet speed.", "response_a": "Have you tried restarting your router?", "response_b": "Your internet is slow. Try restarting the modem and router. If that doesn't work, call our support.", "preferred": "response_b"},
        {"query": "Account locked.", "response_a": "Please contact us.", "response_b": "For security reasons, please visit our password reset page or call our dedicated support line for account unlocking.", "preferred": "response_b"},
    ]

# --- Core Pipeline Functions ---

def train_sft_model(model_name="mock-llm-base"):
    """Simulates the Behavior Cloning (SFT) phase."""
    print("\n--- Step 1: Behavior Cloning (SFT) ---")
    tokenizer = MockTokenizer()
    base_llm = MockModel(name=model_name)
    demonstration_data = load_demonstration_data()

    sft_trainer = MockSFTTrainer(base_llm, tokenizer, demonstration_data)
    sft_model = sft_trainer.train()
    sft_model.save_pretrained("./sft_model")
    return sft_model, tokenizer

def train_reward_model(sft_model, tokenizer):
    """Simulates the Reward Model training phase."""
    print("\n--- Step 2: Reward Model Training ---")
    reward_model = MockRewardModel(sft_model, name="mock-reward-model")
    preference_data = load_preference_data()

    reward_model.train(preference_data)
    reward_model.save_pretrained("./reward_model")
    return reward_model

def train_rlhf_model(sft_model, tokenizer, reward_model):
    """Simulates the Reinforcement Learning from Human Feedback (RLHF) phase."""
    print("\n--- Step 3: RLHF Fine-tuning ---")
    # A reference model (a copy of the SFT model before RLHF) is typically used in PPO
    ref_model = MockModel(name="mock-ref-model")
    # In a real scenario, this dataset would consist of prompts to generate responses for RLHF
    rlhf_dataset = [
        {"query": "My printer is not connecting.", "context": "home office"},
        {"query": "I need to upgrade my service plan.", "context": "account management"}
    ]

    ppo_trainer = MockPPOTrainer(sft_model, ref_model, tokenizer, reward_model, rlhf_dataset)
    rlhf_model = ppo_trainer.train()
    rlhf_model.save_pretrained("./rlhf_model")
    return rlhf_model

def generate_and_select_response(llm_model, reward_model, tokenizer, query, num_candidates=3):
    """Simulates inference with rejection sampling (Best-of-N)."""
    print(f"\n--- Step 4: Inference with Rejection Sampling for query: '{query}' ---")
    input_tokens = tokenizer.encode(query)
    candidate_responses = []
    
    print(f"Generating {num_candidates} candidate responses...")
    for i in range(num_candidates):
        # In a real scenario, the LLM would generate distinct responses based on some stochasticity
        generated_tokens = llm_model.generate([input_tokens], num_return_sequences=1, max_length=len(input_tokens) + 20)[0]
        response_text = tokenizer.decode(generated_tokens)
        candidate_responses.append({"text": response_text, "tokens": generated_tokens})
        print(f"  Candidate {i+1}: '{response_text}'")

    best_response = None
    highest_score = -float('inf')

    print("Evaluating candidates using Reward Model...")
    for candidate in candidate_responses:
        score = reward_model(input_tokens, candidate["tokens"])
        print(f"    Candidate: '{candidate['text']}' | Score: {score:.2f}")
        if score > highest_score:
            highest_score = score
            best_response = candidate["text"]
    
    print(f"  Selected Best Response (Score: {highest_score:.2f}): '{best_response}'")
    return best_response

def main():
    """Orchestrates the entire customer support agent pipeline."""
    print("--- Starting Intelligent Customer Support Agent Pipeline ---")

    # 1. Behavior Cloning (SFT) for initial skill acquisition
    sft_model, tokenizer = train_sft_model()

    # 2. Reward Model Training using human preferences
    reward_model = train_reward_model(sft_model, tokenizer)

    # 3. RLHF Fine-tuning to align with human preferences
    rlhf_model = train_rlhf_model(sft_model, tokenizer, reward_model)

    # 4. Inference with Rejection Sampling for high-quality responses
    customer_query = "My account is locked and I can't log in."
    final_response = generate_and_select_response(rlhf_model, reward_model, tokenizer, customer_query)

    print("\n--- Deployment & API (Conceptual) ---")
    print("In a real application, the rlhf_model and reward_model would be deployed via frameworks like FastAPI/BentoML.")
    print("Example API Endpoint: /ask_agent")
    print("Request (JSON): {'query': '...'}")
    print("Response (JSON): {'answer': '...'}")

    print("\n--- Dual Data Collection & Continuous Improvement (Conceptual) ---")
    print("This pipeline is designed for continuous improvement. New demonstration and preference data")
    print("would be collected from live interactions to periodically retrain and update models, ensuring")
    print("the agent remains robust, capable, and human-aligned.")

    print("\n--- Pipeline Complete ---")

if __name__ == "__main__":
    main()