class QueryClassifier:
    """
    A simplified query complexity classifier.
    For demonstration, it uses basic heuristics (query length and keywords)
    instead of a trained ML model.
    """

    def __init__(self):
        # In a real system, this would load a pre-trained small LLM or ML model
        # For this demo, no model loading is needed.
        pass

    def classify(self, query: str) -> str:
        """
        Classifies the incoming query into complexity levels: 
        'straightforward', 'moderate', or 'complex'.
        """
        query_lower = query.lower()

        # Straightforward queries: short, simple greetings, direct product lookup
        straightforward_keywords = ["hello", "hi", "thank you", "what is", "price of", "return policy", "order status"]
        if any(kw in query_lower for kw in straightforward_keywords) and len(query.split()) < 10:
            return "straightforward"
        
        # Complex queries: involve comparisons, multiple conditions, troubleshooting, advice
        complex_keywords = ["compare", "best for", "troubleshoot", "recommendation", "why is", "how to fix", "multiple items"]
        if any(kw in query_lower for kw in complex_keywords) or len(query.split()) > 20:
            return "complex"

        # Moderate queries: default for anything not clearly straightforward or complex
        return "moderate"

# Example Usage (for testing the classifier in isolation)
if __name__ == "__main__":
    classifier = QueryClassifier()
    
    queries = [
        "Hello, what is my order status?", # Straightforward
        "How do I return a product?",       # Straightforward
        "I need help with my recent purchase, order number 12345.", # Moderate
        "What's the difference between product A and product B, and which one is better for gaming and video editing?", # Complex
        "My device is not turning on after the latest update, how can I troubleshoot it?", # Complex
        "Can you recommend a laptop for a student on a budget?", # Complex
        "Where is my package?",             # Straightforward
        "I have a billing question.",       # Moderate
        "My account seems to be locked, what should I do?"
    ]

    for q in queries:
        complexity = classifier.classify(q)
        print(f"Query: \"{q}\" -> Complexity: {complexity}")
