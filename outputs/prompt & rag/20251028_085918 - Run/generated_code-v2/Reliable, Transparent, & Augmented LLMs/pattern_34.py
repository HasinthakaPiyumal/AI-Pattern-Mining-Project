import re

def simulate_llm_sentiment(ticket_text: str) -> str:
    if "issue resolved quickly and efficiently!" in ticket_text:
        return "The customer expressed great satisfaction: issue resolved quickly and efficiently!"
    elif "I am extremely frustrated with this problem." in ticket_text:
        return "Customer is very upset: I am extremely frustrated with this problem."
    elif "Immediate assistance required!" in ticket_text:
        return "Urgent matter detected: Immediate assistance required!"
    elif "Can you help me with this?" in ticket_text:
        return "A general inquiry: Can you help me with this?"
    return "Neutral sentiment observed."

class SentimentVerbalizer:
    def __init__(self):
        self.output_to_label_map = [
            (re.compile(r".*satisfaction.*|.*resolved quickly.*", re.IGNORECASE), "Positive"),
            (re.compile(r".*upset.*|.*frustrated.*|.*problem.*", re.IGNORECASE), "Negative"),
            (re.compile(r".*urgent.*|.*immediate assistance.*", re.IGNORECASE), "Urgent"),
            (re.compile(r".*general inquiry.*|.*help me.*", re.IGNORECASE), "Neutral"),
        ]
        self.label_to_output_map = {
            "Positive": "The customer provided positive feedback and seems satisfied.",
            "Negative": "The customer expressed dissatisfaction and requires further attention.",
            "Neutral": "The customer made a neutral statement or a general inquiry.",
            "Urgent": "This is an urgent request and needs immediate action.",
            "Unknown": "The sentiment of the customer\'s message is unclear."
        }

    def label_from_llm_output(self, llm_output_text: str) -> str:
        for pattern, label in self.output_to_label_map:
            if pattern.search(llm_output_text):
                return label
        return "Unknown"

    def llm_output_from_label(self, sentiment_label: str) -> str:
        return self.label_to_output_map.get(sentiment_label, self.label_to_output_map["Unknown"])

def main():
    verbalizer = SentimentVerbalizer()

    customer_tickets = [
        "Thank you, the issue resolved quickly and efficiently!",
        "I am extremely frustrated with this problem.",
        "Can you help me with this? My internet is not working.",
        "Immediate assistance required! My service is completely down.",
        "Just checking on my order status."
    ]

    print("--- Sentiment Analysis for Customer Support Tickets ---")
    print("\n")

    for i, ticket in enumerate(customer_tickets):
        print(f"Ticket {i+1}: {ticket}")
        llm_output = simulate_llm_sentiment(ticket)
        sentiment_label = verbalizer.label_from_llm_output(llm_output)
        response_phrase = verbalizer.llm_output_from_label(sentiment_label)

        print(f"  Simulated LLM Output: {llm_output}")
        print(f"  Standardized Sentiment Label: {sentiment_label}")
        print(f"  Generated LLM-like Response Phrase: {response_phrase}")
        print("\n")

if __name__ == "__main__":
    main()