class Verbalizer:
    def __init__(self, label_to_output_map: dict):
        self.label_to_output_map = label_to_output_map
        self.output_to_label_map = self._build_output_to_label_map()

    def _build_output_to_label_map(self):
        inverted_map = {}
        for label, outputs in self.label_to_output_map.items():
            for output_phrase in outputs:
                inverted_map[output_phrase.lower()] = label
        return inverted_map

    def output_to_label(self, llm_output: str) -> str:
        llm_output_lower = llm_output.lower()
        for output_phrase, label in self.output_to_label_map.items():
            if output_phrase in llm_output_lower:
                return label
        return "Unknown"

    def label_to_output(self, label: str) -> str:
        if label in self.label_to_output_map:
            # Return the first phrase as a representative output
            return self.label_to_output_map[label][0]
        return ""

class SentimentAnalyzer:
    def __init__(self, verbalizer_instance: Verbalizer):
        self.verbalizer = verbalizer_instance

    def _simulate_llm_response(self, ticket_text: str) -> str:
        ticket_text_lower = ticket_text.lower()
        if "refund" in ticket_text_lower or "cancel" in ticket_text_lower or "broken" in ticket_text_lower:
            return "This is negative sentiment, I need a quick resolution."
        elif "thank you" in ticket_text_lower or "great" in ticket_text_lower or "excellent" in ticket_text_lower:
            return "Everything is positive, good job."
        elif "question" in ticket_text_lower or "how to" in ticket_text_lower:
            return "I have a neutral inquiry."
        elif "urgent" in ticket_text_lower or "critical" in ticket_text_lower or "immediately" in ticket_text_lower:
            return "This is an urgent matter that needs immediate attention."
        else:
            return "I am processing the request."

    def analyze_ticket(self, ticket_text: str) -> dict:
        simulated_llm_output = self._simulate_llm_response(ticket_text)
        identified_sentiment_label = self.verbalizer.output_to_label(simulated_llm_output)
        return {
            "ticket_text": ticket_text,
            "simulated_llm_output": simulated_llm_output,
            "identified_sentiment_label": identified_sentiment_label
        }

# Example Flow:
if __name__ == "__main__":
    # 1. Define the label_to_output_map
    label_to_output_map = {
        "Positive": ["positive", "good job", "everything is good", "excellent"],
        "Negative": ["negative", "bad experience", "needs resolution", "broken"],
        "Neutral": ["neutral inquiry", "processing request", "information request"],
        "Urgent": ["urgent matter", "immediate attention", "critical issue"]
    }

    # 2. Instantiate Verbalizer with the map
    verbalizer = Verbalizer(label_to_output_map)

    # 3. Instantiate SentimentAnalyzer with the Verbalizer instance
    sentiment_analyzer = SentimentAnalyzer(verbalizer)

    # 4. Provide customer support ticket texts to analyze_ticket
    ticket_texts = [
        "My internet is completely broken and I need a refund immediately.",
        "Thank you for your excellent support, everything is working great now.",
        "I have a question about my billing cycle.",
        "This is a critical issue that needs immediate attention, my service is down!",
        "I just want to know how to update my profile."
    ]

    # 5. Print the analysis results
    print("--- Sentiment Analysis Results ---")
    for text in ticket_texts:
        analysis_result = sentiment_analyzer.analyze_ticket(text)
        print(f"Ticket: {analysis_result['ticket_text']}")
        print(f"  Simulated LLM Output: {analysis_result['simulated_llm_output']}")
        print(f"  Identified Sentiment: {analysis_result['identified_sentiment_label']}")
        print("-" * 30)

    # Demonstrate label_to_output
    print("\n--- Verbalizer label_to_output Demo ---")
    print(f"Positive label to output: {verbalizer.label_to_output('Positive')}")
    print(f"Urgent label to output: {verbalizer.label_to_output('Urgent')}")
    print(f"NonExistent label to output: {verbalizer.label_to_output('NonExistent')}")
