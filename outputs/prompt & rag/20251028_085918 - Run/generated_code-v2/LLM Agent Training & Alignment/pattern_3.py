from human_feedback_simulator import HumanFeedbackSimulator

class RewardModel:
    """
    A placeholder Reward Model that assigns a score to an explanation.
    In a real-world RLHF setup, this would be a neural network (e.g., a fine-tuned BERT model)
    trained on human preference data (comparisons of explanations).
    For this demonstration, it uses a simulated score from the HumanFeedbackSimulator.
    """
    def __init__(self):
        self.feedback_simulator = HumanFeedbackSimulator() # Using the simulator to get scores for demo
        print("Initialized RewardModel using HumanFeedbackSimulator for score generation.")

    def predict_score(self, explanation: str) -> float:
        """
        Predicts a reward score for a given explanation.
        Higher scores indicate better explanations according to the (simulated) human preference.
        """
        # In a real RM, this would be a forward pass through a trained neural network.
        # For this demo, we use the simulated human preference score.
        score = self.feedback_simulator.get_preference_score(explanation)
        return score

# Example Usage (for testing):
# if __name__ == "__main__":
#     rm = RewardModel()
#     test_explanation = "This is a very clear and accurate explanation about hypertension."
#     score = rm.predict_score(test_explanation)
#     print(f"Reward score for explanation: {score:.4f}")
#
#     another_explanation = "Hypertension is when your blood pressure is high."
#     score2 = rm.predict_score(another_explanation)
#     print(f"Reward score for another explanation: {score2:.4f}")