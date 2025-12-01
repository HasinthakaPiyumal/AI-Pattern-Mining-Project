import pandas as pd
import random
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class DataGenerator:
    def __init__(self):
        self.simple_queries = [
            "What is your return policy?",
            "How can I track my order?",
            "What are your operating hours?",
            "How do I reset my password?",
            "Do you offer free shipping?"
        ]
        self.moderate_queries = [
            "I want to exchange an item, what's the process?",
            "My product arrived damaged, what should I do?",
            "Can I combine multiple discount codes?",
            "What is the warranty on this electronics product?",
            "How do I configure my account settings for email notifications?"
        ]
        self.complex_queries = [
            "I'm experiencing an intermittent error with the API integration, specifically with the authentication token expiration, can you help me debug this?",
            "I need assistance with a custom order that requires specific material sourcing and a tailored delivery schedule to multiple international locations, what are my options?",
            "Can you provide a detailed comparison between your premium subscription tiers, including SLA differences, data retention policies, and compliance certifications?",
            "I'm trying to migrate my entire data infrastructure from a legacy system to your cloud platform, what are the best practices and potential pitfalls I should be aware of?",
            "Explain the legal implications of using your software in a highly regulated industry, specifically regarding data privacy and intellectual property rights in different jurisdictions."
        ]

    def generate_synthetic_queries(self, num_queries_per_type=50):
        all_queries = []
        for _ in range(num_queries_per_type):
            all_queries.append(random.choice(self.simple_queries))
            all_queries.append(random.choice(self.moderate_queries))
            all_queries.append(random.choice(self.complex_queries))
        random.shuffle(all_queries)
        return all_queries

    def simulate_strategy_success(self, query_text, strategy_type):
        query_len = len(query_text.split())

        if strategy_type == "simple_faq":
            if query_len < 10 and any(keyword in query_text.lower() for keyword in ["return", "track", "hours", "password", "shipping"]):
                return True, "simple"
            return False, None
        elif strategy_type == "single_step_rag":
            if 8 <= query_len < 25 and not any(keyword in query_text.lower() for keyword in ["debug", "migrate", "legal", "compare", "custom order"]):
                return True, "moderate"
            return False, None
        elif strategy_type == "multi_step_rag":
            if query_len >= 20 and any(keyword in query_text.lower() for keyword in ["api integration", "custom order", "subscription tiers", "data infrastructure", "legal implications"]):
                return True, "complex"
            return False, None
        return False, None

    def create_training_dataset(self, num_queries=150):
        queries = self.generate_synthetic_queries(num_queries // 3)
        labeled_data = []

        for query in queries:
            label = None
            # Strategy 1: Model Prediction Outcomes (prioritize simpler models)
            for strategy in ["simple_faq", "single_step_rag", "multi_step_rag"]:
                success, complexity_label = self.simulate_strategy_success(query, strategy)
                if success:
                    label = complexity_label
                    break

            # Strategy 2: Inherent Dataset Biases (fallback)
            if label is None:
                query_len = len(query.split())
                if query_len < 10:
                    label = "simple"
                elif 10 <= query_len < 20:
                    label = "moderate"
                else:
                    label = "complex"

            labeled_data.append({"query": query, "complexity_label": label})

        return pd.DataFrame(labeled_data)

class ClassifierTrainer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.classifier = LogisticRegression(max_iter=1000)

    def train_query_complexity_classifier(self, data_df):
        X = data_df["query"]
        y = data_df["complexity_label"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        X_train_vec = self.vectorizer.fit_transform(X_train)
        self.classifier.fit(X_train_vec, y_train)

        X_test_vec = self.vectorizer.transform(X_test)
        y_pred = self.classifier.predict(X_test_vec)

        report = classification_report(y_test, y_pred, output_dict=True)

        joblib.dump(self.vectorizer, "tfidf_vectorizer.joblib")
        joblib.dump(self.classifier, "logistic_regression_classifier.joblib")

        return self.vectorizer, self.classifier, report

    def load_classifier(self, vectorizer_path="tfidf_vectorizer.joblib", classifier_path="logistic_regression_classifier.joblib"):
        self.vectorizer = joblib.load(vectorizer_path)
        self.classifier = joblib.load(classifier_path)
        return self.vectorizer, self.classifier

    def predict_complexity(self, query):
        query_vec = self.vectorizer.transform([query])
        return self.classifier.predict(query_vec)[0]

def main():
    data_gen = DataGenerator()
    classifier_trainer = ClassifierTrainer()

    print("Generating training data...")
    training_data = data_gen.create_training_dataset(num_queries=300)
    print(f"Generated {len(training_data)} samples.")
    print("Sample data:\n", training_data.head())

    print("Training query complexity classifier...")
    vectorizer, classifier, report = classifier_trainer.train_query_complexity_classifier(training_data)
    print("Classifier training complete.")
    print("Classification Report:\n", pd.DataFrame(report).transpose())

    print("\nDemonstrating adaptive query routing...")
    test_queries = [
        "What is your refund policy?",
        "I need help understanding my bill statement.",
        "My API call is failing with a 500 error after a recent update, what could be the issue?",
        "How do I change my profile picture?",
        "What are the technical specifications for integrating your platform with an existing CRM system?"
    ]

    for query in test_queries:
        predicted_complexity = classifier_trainer.predict_complexity(query)
        print(f"Query: \"{query}\" -> Predicted Complexity: {predicted_complexity}")

if __name__ == "__main__":
    main()