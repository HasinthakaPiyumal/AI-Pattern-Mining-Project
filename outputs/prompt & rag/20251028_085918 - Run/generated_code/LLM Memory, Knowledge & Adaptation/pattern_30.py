from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
import numpy as np

class QueryComplexityClassifier:
    def __init__(self):
        # A simple pipeline for text classification
        self.model = make_pipeline(TfidfVectorizer(), SVC(kernel='linear'))
        self.labels = ['simple', 'medium', 'complex']

    def train_classifier(self, queries, complexities):
        """
        Trains the query complexity classifier.
        queries: List of query strings.
        complexities: List of corresponding complexity labels ('simple', 'medium', 'complex').
        """
        print("Training Query Complexity Classifier...")
        self.model.fit(queries, complexities)
        print("Classifier trained.")

    def predict_complexity(self, query):
        """
        Predicts the complexity of a given query.
        """
        if not hasattr(self.model, 'classes_'):
            raise RuntimeError("Classifier not trained. Please call train_classifier first.")
        return self.model.predict([query])[0]

    @staticmethod
    def generate_training_data():
        """
        Generates synthetic training data for query complexity classification.
        In a real application, this would involve more sophisticated methods
        like rule-based systems, expert labeling, or crowd-sourcing.
        """
        print("Generating synthetic training data...")
        sample_queries = [
            "What is the dosage for ibuprofen?", # Simple
            "Symptoms of common cold vs flu?", # Medium
            "Differential diagnosis for chronic fatigue and myalgia in a 45-year-old female with a history of lupus?", # Complex
            "What is paracetamol?", # Simple
            "Explain the mechanism of action of ACE inhibitors.", # Medium
            "Discuss the latest guidelines for managing type 2 diabetes with comorbid cardiovascular disease, including pharmacological and lifestyle interventions.", # Complex
            "How often should I take aspirin for a headache?", # Simple
            "Compare and contrast MRSA and VRSA infections.", # Medium
            "Outline the diagnostic criteria and management strategies for antiphospholipid syndrome, considering pregnancy complications.", # Complex
        ]
        sample_complexities = [
            'simple', 'medium', 'complex',
            'simple', 'medium', 'complex',
            'simple', 'medium', 'complex',
        ]
        print(f"Generated {len(sample_queries)} samples.")
        return sample_queries, sample_complexities

# Example Usage:
if __name__ == "__main__":
    classifier = QueryComplexityClassifier()
    
    # Generate and train
    queries, complexities = QueryComplexityClassifier.generate_training_data()
    classifier.train_classifier(queries, complexities)

    # Test predictions
    print(f"\nPrediction for 'What is a fever?': {classifier.predict_complexity('What is a fever?')}")
    print(f"Prediction for 'How do diuretics work?': {classifier.predict_complexity('How do diuretics work?')}")
    print(f"Prediction for 'Analyze the efficacy of novel immunotherapies for metastatic melanoma, considering patient stratification biomarkers.': {classifier.predict_complexity('Analyze the efficacy of novel immunotherapies for metastatic melanoma, considering patient stratification biomarkers.')}")
