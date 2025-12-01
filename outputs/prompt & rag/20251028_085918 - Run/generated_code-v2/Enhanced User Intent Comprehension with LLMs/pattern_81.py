from transformers import pipeline

class IntentRecognizer:
    def __init__(self, candidate_labels=None):
        if candidate_labels is None:
            self.candidate_labels = [
                'billing_inquiry', 'technical_support', 'product_information',
                'account_management', 'general_query', 'complaint', 'feature_request',
                'cancel_service'
            ]
        else:
            self.candidate_labels = candidate_labels

        # Using a zero-shot classification pipeline for intent understanding.
        # This leverages a foundation model (BART large MNLI) to classify text
        # into custom labels without explicit fine-tuning on those labels.
        # This model will be downloaded the first time it's used.
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        print("IntentRecognizer initialized with zero-shot classification model.")

    def recognize_intent(self, text):
        if not text.strip():
            return {"intent": "empty_query", "confidence": 1.0}

        try:
            result = self.classifier(text, self.candidate_labels, multi_label=False)
            top_intent = result['labels'][0]
            confidence = result['scores'][0]
            return {"intent": top_intent, "confidence": confidence}
        except Exception as e:
            print(f"Error during intent recognition: {e}")
            return {"intent": "error", "confidence": 0.0}