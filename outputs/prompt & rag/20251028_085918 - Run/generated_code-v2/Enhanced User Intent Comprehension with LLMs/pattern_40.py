
import random

class FoundationModel:
    """
    Simulates a fine-tuned foundation model for intent recognition.
    In a real scenario, this would load a pre-trained model (e.g., from Hugging Face transformers)
    and perform actual inference.
    """
    def __init__(self):
        self.intents = [
            "track_order", "cancel_order", "return_item", "product_information",
            "payment_issue", "delivery_issue", "technical_support", "greeting",
            "unknown_intent"
        ]
        # Simulate some keyword-based "understanding" and "ambiguity"
        self._keyword_map = {
            "order": {"track_order": 0.6, "delivery_issue": 0.3, "cancel_order": 0.1},
            "delivery": {"delivery_issue": 0.8, "track_order": 0.1},
            "where": {"track_order": 0.7, "unknown_intent": 0.2},
            "cancel": {"cancel_order": 0.9},
            "return": {"return_item": 0.9},
            "product": {"product_information": 0.8},
            "item": {"product_information": 0.5, "return_item": 0.3},
            "pay": {"payment_issue": 0.8},
            "problem": {"technical_support": 0.4, "unknown_intent": 0.5},
            "hello": {"greeting": 0.9},
            "hi": {"greeting": 0.9},
            "my order": {"track_order": 0.5, "delivery_issue": 0.3, "cancel_order": 0.2}
        }

    def predict(self, text: str) -> dict:
        """
        Simulates predicting intent probabilities for a given text.
        Returns a dictionary of intent -> probability.
        """
        text = text.lower()
        probabilities = {intent: 0.0 for intent in self.intents}
        matched = False

        for keyword, intent_probs in self._keyword_map.items():
            if keyword in text:
                for intent, prob in intent_probs.items():
                    probabilities[intent] += prob
                matched = True

        if not matched:
            probabilities["unknown_intent"] = 0.9 # High probability for truly unknown

        # Normalize probabilities (crude simulation)
        total_prob = sum(probabilities.values())
        if total_prob == 0:
             probabilities["unknown_intent"] = 1.0
        else:
            probabilities = {k: v / total_prob for k, v in probabilities.items()}

        # Add some noise to make it less deterministic
        for intent in probabilities:
            probabilities[intent] = max(0.0, min(1.0, probabilities[intent] + random.uniform(-0.05, 0.05)))

        # Re-normalize after noise
        total_prob_after_noise = sum(probabilities.values())
        if total_prob_after_noise > 0:
            probabilities = {k: v / total_prob_after_noise for k, v in probabilities.items()}
        else: # Fallback if all become zero
            probabilities = {intent: 1/len(self.intents) for intent in self.intents}


        return probabilities

class IntentRecognizer:
    """
    Uses a FoundationModel to identify the most likely intent and manage ambiguity.
    """
    def __init__(self, confidence_threshold: float = 0.7, ambiguity_threshold: float = 0.15):
        self.model = FoundationModel()
        self.confidence_threshold = confidence_threshold
        self.ambiguity_threshold = ambiguity_threshold # If top two intents are within this diff, it's ambiguous

    def recognize(self, text: str) -> tuple[str, float, bool]:
        """
        Recognizes intent, returns (intent, confidence, needs_clarification).
        """
        probabilities = self.model.predict(text)
        sorted_intents = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)

        top_intent, top_confidence = sorted_intents[0]
        
        needs_clarification = False
        if len(sorted_intents) > 1:
            second_intent, second_confidence = sorted_intents[1]
            if top_confidence - second_confidence < self.ambiguity_threshold and top_confidence > 0.3: # If top two are too close and not super low confidence
                needs_clarification = True

        if top_confidence < self.confidence_threshold:
            return "unknown_intent", top_confidence, False # Too low confidence, treat as unknown

        return top_intent, top_confidence, needs_clarification
