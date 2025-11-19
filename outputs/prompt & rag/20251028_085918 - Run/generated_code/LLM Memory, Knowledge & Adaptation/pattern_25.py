import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
import joblib

class QueryComplexityClassifier:
    def __init__(self, model_path="query_complexity_model.joblib"):
        self.model_path = model_path
        self.model = self._load_or_train_model()

    def _load_or_train_model(self):
        try:
            return joblib.load(self.model_path)
        except FileNotFoundError:
            print("Model not found. Training a new one...")
            return self._train_model()

    def _train_model(self):
        # Simplified training data generation for demonstration
        # In a real scenario, this would come from a larger, diverse dataset.
        simple_queries = [
            "What is a headache?",
            "How to treat a cold?",
            "Symptoms of flu",
            "What is blood pressure?",
            "Can I take paracetamol for fever?"
        ]
        complex_queries = [
            "Explain the pathophysiology of type 2 diabetes and its management strategies including pharmacological and non-pharmacological interventions.",
            "Differentiate between various autoimmune encephalitis syndromes, discussing diagnostic criteria, prognosis, and treatment protocols.",
            "Analyze the genetic predispositions and environmental factors contributing to the development of Crohn's disease, and outline the current therapeutic landscape.",
            "Discuss the implications of a positive ANA test in the absence of clinical symptoms for systemic lupus erythematosus, considering differential diagnoses and follow-up strategies.",
            "Review the current understanding of prion diseases, including their molecular mechanisms, diagnostic challenges, and experimental therapeutic approaches."
        ]
        
        X = simple_queries + complex_queries
        y = ["simple"] * len(simple_queries) + ["complex"] * len(complex_queries)

        model = make_pipeline(TfidfVectorizer(), SVC(kernel='linear', probability=True))
        model.fit(X, y)
        joblib.dump(model, self.model_path)
        return model

    def classify(self, query: str) -> str:
        prediction = self.model.predict([query])[0]
        proba = self.model.predict_proba([query])[0]
        confidence = max(proba)
        print(f"Query: '{query}' classified as '{prediction}' with confidence {confidence:.2f}")
        return prediction

# Example Usage:
if __name__ == "__main__":
    classifier = QueryComplexityClassifier()
    
    print("\n--- Testing Classifier ---")
    classifier.classify("What are the common symptoms of a stroke?")
    classifier.classify("Discuss the neurological pathways involved in proprioception and the clinical manifestations of their disruption.")
    classifier.classify("How much water should I drink daily?")
    classifier.classify("Explain the latest advancements in CRISPR-Cas9 technology for genetic disease therapy, including ethical considerations.")
    classifier.classify("Is my fever serious?")

    # Simulate automatic training data generation (conceptually)
    # In a real system, feedback loops or synthetic data generation pipelines
    # would continuously generate and label data to improve the classifier.
    print("\n--- Automatic Classifier Training Data Generation (Conceptual) ---")
    print("This component would involve processes to continually gather new queries,")
    print("potentially label them (human-in-the-loop or weak supervision), and retrain the classifier.")
    print("For instance, if an LLM struggles with a query, it could be flagged for review and added to training data.")
    print("Or, rules based on keyword density, sentence length, or named entity recognition could generate 'weak' labels.")

