class MedicalSentimentAnalyzer:
    def __init__(self):
        # Define the verbalizer map: LLM output patterns mapped to standardized sentiment labels
        self.verbalizer_map = {
            "positive": ["excellent experience", "very satisfied", "great care", "highly recommend", "no complaints", "satisfied", "good service", "positive feedback"],
            "neutral": ["no strong feelings", "average experience", "acceptable", "neither good nor bad", "neutral response", "okay"],
            "negative": ["poor service", "unsatisfied", "bad experience", "not happy", "disappointed", "terrible", "negative feedback"]
        }

    def _simulate_llm_output(self, feedback_text: str) -> str:
        """
        Simulates an LLM's raw output based on patient feedback.
        In a real application, this would be an actual LLM call.
        For this demonstration, it's a simple keyword-based simulation.
        """
        feedback_text_lower = feedback_text.lower()

        for sentiment, keywords in self.verbalizer_map.items():
            for keyword in keywords:
                if keyword in feedback_text_lower:
                    # Simulate the LLM providing an output that needs verbalization
                    if sentiment == "positive":
                        return "The patient expressed strong satisfaction with the service."
                    elif sentiment == "negative":
                        return "There were clear indications of dissatisfaction."
                    elif sentiment == "neutral":
                        return "The feedback was largely indifferent or non-committal."
        return "The patient's sentiment was not clearly discernible."

    def analyze_sentiment(self, feedback_text: str) -> str:
        """
        Analyzes patient feedback to classify sentiment using the Verbalizer pattern.
        """
        # Step 1: Simulate LLM processing the feedback
        llm_raw_output = self._simulate_llm_output(feedback_text)
        print(f"Raw LLM Output for '{feedback_text}': {llm_raw_output}")

        # Step 2: Apply the Verbalizer to map LLM's raw output to a standardized label
        normalized_sentiment = "Unknown"
        llm_raw_output_lower = llm_raw_output.lower()

        for sentiment_label, verbalizations in self.verbalizer_map.items():
            for verbalization_keyword in verbalizations:
                if verbalization_keyword in llm_raw_output_lower:
                    normalized_sentiment = sentiment_label.capitalize() # e.g., "Positive", "Negative"
                    break # Found a match, no need to check further verbalizations for this sentiment
            if normalized_sentiment != "Unknown":
                break # Found a match for a sentiment label, no need to check other sentiments

        return normalized_sentiment

# --- Demonstration --- 
if __name__ == "__main__":
    analyzer = MedicalSentimentAnalyzer()

    patient_feedback_examples = [
        "The nurse provided excellent care and was very attentive. I highly recommend them!",
        "My experience was just average. Nothing really stood out, good or bad.",
        "I'm very disappointed with the long wait times and the doctor seemed rushed.",
        "The staff was quite professional and the facilities were clean. No complaints.",
        "It was an okay visit, nothing special to report.",
        "Absolutely terrible service, I will not be returning.",
        "The service was mostly good, but the billing process was a bit confusing."
    ]

    for i, feedback in enumerate(patient_feedback_examples):
        print(f"\n--- Patient Feedback {i+1} ---")
        sentiment = analyzer.analyze_sentiment(feedback)
        print(f"Classified Sentiment: {sentiment}")
        print("----------------------")
