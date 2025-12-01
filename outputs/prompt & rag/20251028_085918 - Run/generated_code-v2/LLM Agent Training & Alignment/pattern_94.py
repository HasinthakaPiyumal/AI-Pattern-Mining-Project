import random

class RewardModel:
    """A conceptual Reward Model that learns to predict human preferences."""

    def __init__(self, model_name="SimulatedRewardModel"):
        self.model_name = model_name
        self.learned_preferences = {}
        print(f"Reward Model \'{self.model_name}\' initialized.")

    def train(self, feedback_data: list[dict]):
        """Simulates training the reward model on collected human feedback.
        In a real scenario, this would involve a machine learning training loop
        (e.g., using a neural network to predict preference scores).
        """
        print(f"\nTraining Reward Model \'{self.model_name}\' with {len(feedback_data)} feedback entries.")
        for entry in feedback_data:
            query = entry["query"]
            preferred_response_index = entry["preferred_response_index"]
            # In a real model, features would be extracted from responses for training.
            # Here, we'll conceptually store the preferred index for simplicity.
            if query not in self.learned_preferences:
                self.learned_preferences[query] = {}
            self.learned_preferences[query][tuple(entry["responses"])] = preferred_response_index
        print(f"Reward Model training simulated. Learned preferences for {len(self.learned_preferences)} queries.")

    def predict_rewards(self, query: str, responses: list[str]) -> list[float]:
        """Simulates predicting a reward score for each response based on learned preferences.
        Higher scores indicate higher predicted human preference.
        """
        print(f"\nPredicting rewards for query: \'{query}\'")
        predicted_scores = [0.0] * len(responses)

        # In a real model, the RM would analyze the responses and output scores.
        # Here, we'll use our simulated learned preferences.
        if query in self.learned_preferences and tuple(responses) in self.learned_preferences[query]:
            preferred_index = self.learned_preferences[query][tuple(responses)]
            # Assign a higher 'reward' to the historically preferred response
            for i in range(len(responses)):
                if i == preferred_index:
                    predicted_scores[i] = random.uniform(0.8, 1.0) # High reward
                else:
                    predicted_scores[i] = random.uniform(0.1, 0.7) # Lower reward
        else:
            # If no specific preference learned for this exact set, assign random scores
            print("  (No specific learned preference for this set of responses, assigning random scores)")
            predicted_scores = [random.uniform(0.1, 1.0) for _ in responses]

        print(f"  Predicted rewards: {predicted_scores}")
        return predicted_scores