from enum import Enum

class QueryComplexity(Enum):
    SIMPLE = "simple"
    COMPLEX = "complex"

class QueryClassifier:
    """
    Classifies incoming user queries into 'simple' or 'complex'
    to adapt LLM processing strategies.
    """
    def __init__(self):
        # In a real-world scenario, this could be a fine-tuned small LLM,
        # a rule-based system, or a machine learning model trained on query features.
        pass

    def classify(self, query: str) -> QueryComplexity:
        """
        Classifies the given query based on predefined rules or a trained model.
        For demonstration, a simple rule-based classification is used.
        """
        query_lower = query.lower()

        complex_keywords = [
            "diagnose", "differential diagnosis", "treatment plan", 
            "prognosis", "drug interactions", "interpret lab results",
            "clinical guidelines", "pathophysiology", "mechanism of action"
        ]
        
        simple_keywords = [
            "hello", "hi", "what is", "how to", "define", "meaning of", 
            "symptoms of", "causes of", "side effects of"
        ]

        # Check for complex keywords first
        if any(keyword in query_lower for keyword in complex_keywords):
            return QueryComplexity.COMPLEX

        # Check for simple keywords
        if any(keyword in query_lower for keyword in simple_keywords):
            return QueryComplexity.SIMPLE

        # Fallback for queries that don't match explicit simple/complex keywords
        # A more sophisticated model would handle nuanced classification
        if len(query.split()) > 8 and '?' in query: # Heuristic for potentially complex questions
            return QueryComplexity.COMPLEX

        return QueryComplexity.SIMPLE # Default to simple if no clear indication

    def __repr__(self):
        return "QueryClassifier()"
