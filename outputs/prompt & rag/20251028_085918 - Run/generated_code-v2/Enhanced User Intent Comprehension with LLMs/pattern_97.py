class IntentClassifier:
    def __init__(self):
        self.intent_keywords = {
            "Track Order": ["where is", "my order", "tracking", "delivery", "status", "late"],
            "Initiate Return": ["return", "send back", "faulty", "wrong item", "damaged"],
            "Change Shipping Address": ["change address", "update delivery", "wrong address", "move order"],
            "Product Inquiry": ["about product", "details", "specifications", "features", "size", "color"],
            "Complaint": ["complain", "unhappy", "problem", "issue", "dissatisfied", "bad service"]
        }
        self.greetings = ["hello", "hi", "hey", "good morning", "good afternoon"]

    def classify_intent(self, query: str):
        normalized_query = query.lower()

        # Check for greetings first
        if any(greet in normalized_query for greet in self.greetings):
            return {"intent": "Greeting", "confidence": 1.0, "is_ambiguous": False, "clarification": None}

        scores = {intent: 0 for intent in self.intent_keywords}
        for intent, keywords in self.intent_keywords.items():
            for keyword in keywords:
                if keyword in normalized_query:
                    scores[intent] += 1
        
        # Filter out intents with zero scores unless it's a very short query
        active_scores = {intent: score for intent, score in scores.items() if score > 0}

        if not active_scores:
            return {"intent": "Unknown", "confidence": 0.0, "is_ambiguous": False, "clarification": None}

        max_score = max(active_scores.values())
        top_intents = [intent for intent, score in active_scores.items() if score == max_score]

        if len(top_intents) == 1:
            return {"intent": top_intents[0], "confidence": max_score / len(normalized_query.split()), "is_ambiguous": False, "clarification": None}
        else:
            # Ambiguous intent
            clarifications = {
                "Track Order": "Are you trying to track an existing order or inquire about shipping options?",
                "Initiate Return": "Do you want to start a new return or check the status of a past return?",
                "Complaint": "Are you complaining about a product, delivery, or customer service?"
            }
            # Provide a general clarification for ambiguous cases
            suggested_clarification = "It seems your query could relate to a few things. Can you please provide more details? "
            # More specific clarifications if possible based on the top_intents
            if "Track Order" in top_intents and "Complaint" in top_intents:
                suggested_clarification = clarifications.get("Track Order") or suggested_clarification # Prioritize tracking if 'late' is a common overlap

            return {"intent": "Ambiguous", "confidence": max_score / len(normalized_query.split()), "is_ambiguous": True, "clarification": suggested_clarification, "options": top_intents}
