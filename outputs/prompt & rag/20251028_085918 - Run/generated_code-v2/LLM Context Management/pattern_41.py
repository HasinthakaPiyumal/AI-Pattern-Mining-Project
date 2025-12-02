import random
from transformers import AutoTokenizer

class QueryClassifier:
    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # In a real scenario, you would load your fine-tuned classification model here.
        # For this example, we'll simulate classification based on keywords.
        self.complexity_keywords = {
            "straightforward": ["reset password", "return policy", "account login", "shipping cost", "faq", "how to get"],
            "moderate": ["troubleshoot", "configure", "integrate", "compatibility", "update software", "billing dispute", "problem with"],
            "complex": ["unresolved issue", "custom solution", "escalate", "technical deep dive", "data migration", "system failure", "critical error"]
        }
        print(f"QueryClassifier initialized with tokenizer for {model_name}. Classification will be simulated based on keywords.")

    def classify_query(self, query: str) -> str:
        query_lower = query.lower()
        for complexity, keywords in self.complexity_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return complexity
        # If no specific keywords are found, default to 'moderate' or use a more sophisticated fallback.
        # For simulation, let's randomly assign if no strong keyword matches.
        return random.choice(["straightforward", "moderate", "complex"])

class Chatbot:
    def __init__(self, classifier: QueryClassifier):
        self.classifier = classifier
        print("Chatbot initialized with QueryClassifier.")

    def handle_query(self, query: str) -> str:
        print(f"\nCustomer Query: \"{query}\" ")
        complexity = self.classifier.classify_query(query)
        print(f"Classified as: \'{complexity}\'")

        if complexity == "straightforward":
            return self._handle_straightforward(query)
        elif complexity == "moderate":
            return self._handle_moderate(query)
        elif complexity == "complex":
            return self._handle_complex(query)
        else:
            return "Error: Unknown query complexity."

    def _handle_straightforward(self, query: str) -> str:
        # Simulate direct FAQ answer or simple information retrieval
        if "reset password" in query.lower():
            return "To reset your password, please visit our website and click on the 'Forgot Password' link on the login page."
        elif "return policy" in query.lower():
            return "Our standard return policy allows returns within 30 days of purchase with a valid receipt. Special conditions may apply to certain items."
        elif "shipping cost" in query.lower():
            return "Shipping costs depend on your location and the chosen delivery speed. You can see the exact cost at checkout."
        return "Here is a straightforward answer to your question. Please refer to our FAQ for common topics."

    def _handle_moderate(self, query: str) -> str:
        # Simulate knowledge base search or guided troubleshooting
        return f"Searching our comprehensive knowledge base for relevant articles regarding \'{query}\'. You might find solutions in 'Troubleshooting Guide' or 'Configuration Best Practices'."

    def _handle_complex(self, query: str) -> str:
        # Escalate to a human agent with relevant context
        return "Your query requires specialized assistance. I am escalating this to a human support agent who will review your case and contact you shortly to provide a personalized solution."

if __name__ == "__main__":
    # Initialize the classifier and chatbot
    classifier = QueryClassifier()
    chatbot = Chatbot(classifier)

    # Test with different customer queries
    queries = [
        "How do I reset my password?",
        "What is your return policy?",
        "I need to troubleshoot my software installation problem.",
        "How do I configure API integration with a third-party service?",
        "I have an unresolved issue with my recent order that needs immediate attention.",
        "Can you help me with a custom solution for enterprise reporting for my specific business needs?",
        "My account login is not working, and I can't access my profile.",
        "I have a billing dispute regarding my last invoice and need it reviewed.",
        "How to get started with your product?",
        "There is a critical error in my system after the recent update."
    ]

    for q in queries:
        response = chatbot.handle_query(q)
        print(f"Chatbot Response: {response}")
        print("-" * 50)
