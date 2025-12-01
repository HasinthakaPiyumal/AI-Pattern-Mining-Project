import torch
from chatbot_model import ChatbotModel
from reward_model import RewardModel
from data_collector import DataCollector
from rlhf_trainer import RLHFTrainer

# Configuration
MODEL_NAME = "gpt2"  # Or any other suitable small model for demonstration
REWARD_MODEL_NAME = "bert-base-uncased" # Or a smaller model
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
NUM_TRAINING_STEPS_RM = 100 # Simulated steps for Reward Model
NUM_TRAINING_STEPS_RLHF = 50 # Simulated steps for RLHF

def main():
    print("Initializing Chatbot Model...")
    chatbot = ChatbotModel(model_name=MODEL_NAME)

    print("Initializing Reward Model...")
    reward_model = RewardModel(model_name=REWARD_MODEL_NAME)

    print("Simulating Human Feedback Data Collection...")
    data_collector = DataCollector(chatbot)
    # Collect a batch of simulated human preference data
    # For simplicity, we simulate pairs of (prompt, chosen_response, rejected_response)
    # where chosen is preferred over rejected
    simulated_feedback_data = data_collector.collect_feedback_data(num_samples=100)

    print("Training Reward Model...")
    # In a real scenario, you'd have a DataLoader and more robust training loop
    # Here, we'll simulate a simple training process
    for step in range(NUM_TRAINING_STEPS_RM):
        # Sample a batch from simulated_feedback_data
        # For demonstration, we'll just use a small subset conceptually
        if not simulated_feedback_data:
            break
        sample = simulated_feedback_data[step % len(simulated_feedback_data)]
        prompt, chosen_response, rejected_response = sample['prompt'], sample['chosen'], sample['rejected']
        
        # Simulate reward model training step
        # The reward model would take these and calculate a loss based on preference
        # For this example, we'll just call a dummy train_step
        reward_model.train_step(prompt, chosen_response, rejected_response)
        if (step + 1) % 20 == 0:
            print(f"  Reward Model training step {step + 1}/{NUM_TRAINING_STEPS_RM}")
    print("Reward Model training complete (simulated).")

    print("Starting RLHF Fine-tuning of Chatbot Model...")
    rlhf_trainer = RLHFTrainer(chatbot, reward_model, learning_rate=LEARNING_RATE)
    
    for step in range(NUM_TRAINING_STEPS_RLHF):
        # In a real RLHF setup, this involves generating responses, getting rewards, and updating
        # For this simulation, we'll call a dummy training step.
        rlhf_trainer.train_step(prompt_batch=["How can I help you?", "What's my order status?"])
        if (step + 1) % 10 == 0:
            print(f"  RLHF training step {step + 1}/{NUM_TRAINING_STEPS_RLHF}")
    print("RLHF Fine-tuning complete (simulated).")

    print("\n--- Demonstrating Fine-tuned Chatbot ---")
    test_prompts = [
        "Tell me about your return policy.",
        "My order hasn't arrived, what should I do?",
        "Can I change my shipping address?"
    ]

    for prompt in test_prompts:
        print(f"Customer: {prompt}")
        response = chatbot.generate_response(prompt)
        print(f"Chatbot: {response}")
        print("-" * 30)

if __name__ == "__main__":
    main()