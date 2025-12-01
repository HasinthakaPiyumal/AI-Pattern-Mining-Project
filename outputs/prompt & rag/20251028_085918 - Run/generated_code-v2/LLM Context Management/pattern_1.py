class QueryComplexityClassifier:
    def __init__(self):
        # In a real-world scenario, this would be a loaded, pre-trained smaller LLM
        # For demonstration, we'll use a simple rule-based approach.
        print("QueryComplexityClassifier initialized. (Using rule-based classification for demonstration)")

    def classify(self, query: str) -> str:
        """
        Classifies an incoming query into 'straightforward', 'moderate', or 'complex'.
        """
        query_lower = query.lower()

        # Simple rule-based classification for demonstration
        if "what is" in query_lower and ("return policy" in query_lower or "shipping cost" in query_lower):
            return "straightforward" # e.g., "What is your return policy?"
        elif "how to" in query_lower and ("assemble" in query_lower or "troubleshoot" in query_lower):
            return "moderate" # e.g., "How to assemble my new desk?"
        elif "compare" in query_lower or "recommend a product for" in query_lower or "explain why" in query_lower or "diagnostic" in query_lower:
            return "complex" # e.g., "Compare the XYZ laptop with ABC laptop considering gaming performance and battery life."
        elif len(query.split()) < 5:
            return "straightforward" # Very short queries tend to be straightforward
        elif len(query.split()) > 15 and ("problem" in query_lower or "issue" in query_lower or "not working" in query_lower):
            return "complex" # Longer queries with problem indicators
        else:
            return "moderate" # Default to moderate if no strong indicators

