import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

class RewardModel(nn.Module):
    def __init__(self, pretrained_model_name="bert-base-uncased"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        # Use a sequence classification model to act as a reward model.
        # The output logit will be interpreted as the reward score.
        self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name, num_labels=1)

    def forward(self, input_texts):
        # Tokenize the input texts
        inputs = self.tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True)
        # Get the output logits (which represent the reward score)
        outputs = self.model(**inputs)
        # The reward is the single logit output
        return outputs.logits.squeeze(-1)

    def save_pretrained(self, save_directory):
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path):
        model_instance = cls(pretrained_model_name=pretrained_model_name_or_path)
        # The AutoModelForSequenceClassification will load weights correctly
        return model_instance

def conceptually_train_reward_model(reward_model, preference_data):
    """
    A conceptual function to illustrate how a Reward Model would be trained.
    In a real-world scenario, this would involve a proper training loop
    with an optimizer, loss function (e.g., pairwise ranking loss),
    and actual human preference data.

    Args:
        reward_model: An instance of the RewardModel class.
        preference_data: A list of tuples, where each tuple contains
                         (chosen_response, rejected_response) based on human preference.
    """
    print("\n--- Conceptually Training Reward Model ---")
    print(f"Training Reward Model on {len(preference_data)} conceptual preference pairs.")

    # In a real scenario, you'd iterate over epochs, batches, and use a ranking loss.
    # For demonstration, we'll just simulate the process.
    for i, (chosen, rejected) in enumerate(preference_data):
        # Simulate forward pass to get scores
        chosen_score = reward_model(chosen).item()
        rejected_score = reward_model(rejected).item()

        # In a real setup, we would calculate a loss like:
        # loss = -torch.log(torch.sigmoid(chosen_score - rejected_score))
        # and then perform backpropagation and optimization.

        print(f"Pair {i+1}: Chosen='{chosen[:30]}...', Rejected='{rejected[:30]}...'")
        print(f"  Simulated scores: Chosen={chosen_score:.2f}, Rejected={rejected_score:.2f}")
        # Simulate a small update to show the concept
        if chosen_score < rejected_score: # If model prefers wrong, conceptual 'update'
            print("  (Conceptual update: Model would be adjusted to prefer 'chosen' more)")
        else:
            print("  (Conceptual: Model already prefers 'chosen' or scores are close)")

    print("--- Reward Model Training (Conceptual) Complete ---")
    print("Reward Model ready for use in RLHF process.")

if __name__ == "__main__":
    # Example Usage and Conceptual Training
    print("Initializing Reward Model...")
    reward_model = RewardModel("distilbert-base-uncased") # Using a smaller BERT for faster demo
    print("Reward Model Initialized.")

    # Simulate human preference data
    # (chosen_response, rejected_response)
    simulated_preference_data = [
        ("The quick brown fox jumps over the lazy dog.", "Fox jumps over dog."),
        ("A detailed explanation of quantum physics is provided here.", "Quantum physics explanation."),
        ("Customer support can be reached via phone at 1-800-XXX-XXXX or email support@example.com.", "Call support."),
        ("The product features a long-lasting battery life and a high-resolution display, making it ideal for multimedia consumption.", "Battery is good."),
        ("To troubleshoot your device, please ensure it is powered on and connected to a stable internet source. Then, try restarting it.", "Restart device.")
    ]

    conceptually_train_reward_model(reward_model, simulated_preference_data)

    # Demonstrate scoring a new response
    test_responses = [
        "This is a very good and helpful answer for the user.",
        "Bad answer.",
        "The product warranty covers manufacturing defects for a period of one year from the date of purchase."
    ]

    print("\n--- Demonstrating Reward Model Scoring ---")
    for response in test_responses:
        score = reward_model(response).item()
        print(f"Response: '{response[:50]}...'")
        print(f"  Reward Score: {score:.4f}")