
from config import INTENT_KEYWORDS

class IntentRecognizer:
    """Recognizes user intent based on keywords. 
       In a full implementation, this would involve a fine-tuned foundation model.
    """

    def recognize_intent(self, query: str) -> dict:
        """Identifies the most probable intent from a user query."""
        query_lower = query.lower()
        
        # Simple keyword-based matching for demonstration
        for intent, keywords in INTENT_KEYWORDS.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return {"intent": intent, "confidence": 0.8} # Assign a mock confidence

        return {"intent": "unknown_intent", "confidence": 0.1}

    def clarify_intent(self, query: str, possible_intents: list) -> str:
        """Generates a clarification prompt for ambiguous queries.
           This method is a placeholder for more advanced NLU capabilities.
        """
        if not possible_intents:
            return "I couldn't quite understand that. Can you tell me more?"
        
        options = ", ".join([intent.replace("_", " ") for intent in possible_intents])
        return f"It seems like your request regarding '{query}' could be about {options}. Can you clarify?"
