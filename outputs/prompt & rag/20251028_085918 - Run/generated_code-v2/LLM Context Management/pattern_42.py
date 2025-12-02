import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from collections import Counter

# 1. Automatic Training Data Generator (data_generator.py logic)
class DataGenerator:
    def __init__(self):
        pass

    def _no_retrieval_llm_success(self, query_text, true_complexity):
        return true_complexity == "simple" or random.random() < 0.2 # Small chance for others

    def _single_step_rag_success(self, query_text, true_complexity):
        return true_complexity in ["simple", "moderate"] or random.random() < 0.3 # Small chance for complex

    def _multi_step_rag_success(self, query_text, true_complexity):
        return true_complexity in ["simple", "moderate", "complex"]

    def label_queries_by_model_outcomes(self, queries):
        labeled_queries = []
        unlabeled_queries = []
        for query in queries:
            label = None
            if self._no_retrieval_llm_success(query["text"], query["true_complexity"]):
                label = "simple"
            elif self._single_step_rag_success(query["text"], query["true_complexity"]) and label is None:
                label = "moderate"
            elif self._multi_step_rag_success(query["text"], query["true_complexity"]) and label is None:
                label = "complex"
            
            if label:
                labeled_queries.append({"text": query["text"], "complexity": label})
            else:
                unlabeled_queries.append(query)
        return labeled_queries, unlabeled_queries

    def label_queries_by_dataset_biases(self, unlabeled_queries, benchmark_datasets_meta):
        labeled_by_bias = []
        remaining_unlabeled = []
        for query in unlabeled_queries:
            label_assigned = False
            for ds_name, meta in benchmark_datasets_meta.items():
                # Simulate if query originated from this dataset
                if f"ds_{ds_name}" in query.get("id", ""):
                    if meta["bias"] == "single-hop":
                        labeled_by_bias.append({"text": query["text"], "complexity": "moderate"})
                        label_assigned = True
                        break
                    elif meta["bias"] == "multi-hop":
                        labeled_by_bias.append({"text": query["text"], "complexity": "complex"})
                        label_assigned = True
                        break
            if not label_assigned:
                remaining_unlabeled.append(query)
        return labeled_by_bias, remaining_unlabeled

    def generate_training_data(self, raw_customer_queries, benchmark_datasets_meta):
        # Strategy 1: Model Prediction Outcomes
        labeled_by_model, unlabeled_after_model = self.label_queries_by_model_outcomes(raw_customer_queries)
        
        # Strategy 2: Inherent Dataset Biases
        labeled_by_bias, _ = self.label_queries_by_dataset_biases(unlabeled_after_model, benchmark_datasets_meta)
        
        final_labeled_data = labeled_by_model + labeled_by_bias
        
        X = [item["text"] for item in final_labeled_data]
        y = [item["complexity"] for item in final_labeled_data]
        return X, y

# 2. Query Complexity Classifier (classifier.py logic)
class QueryComplexityClassifier:
    def __init__(self):
        self.vectorizer = None
        self.classifier = None

    def train_query_complexity_classifier(self, X_train_text, y_train_labels):
        self.vectorizer = TfidfVectorizer()
        X_train_vectorized = self.vectorizer.fit_transform(X_train_text)
        self.classifier = LogisticRegression(max_iter=1000) # Increased max_iter for convergence
        self.classifier.fit(X_train_vectorized, y_train_labels)
        return self.vectorizer, self.classifier

    def predict_query_complexity(self, vectorizer, classifier, query_text):
        query_vectorized = vectorizer.transform([query_text])
        predicted_label = classifier.predict(query_vectorized)[0]
        return predicted_label

# 3. Main Application and Routing Logic (main.py logic)
class CustomerSupportRouter:
    def __init__(self):
        self.data_generator = DataGenerator()
        self.classifier_model = QueryComplexityClassifier()
        self.vectorizer = None
        self.classifier = None

    def simulate_customer_queries(self):
        return [
            {"id": "q1", "text": "How do I reset my password?", "true_complexity": "simple"},
            {"id": "q2", "text": "Where is my order #12345?", "true_complexity": "simple"},
            {"id": "q3_ds_single-hop", "text": "What is the return policy for electronics bought last month?", "true_complexity": "moderate"},
            {"id": "q4_ds_multi-hop", "text": "Compare the features of the new laptop model with the previous one, considering upgrade options and warranty details.", "true_complexity": "complex"},
            {"id": "q5", "text": "I want to change my shipping address.", "true_complexity": "simple"},
            {"id": "q6_ds_single-hop", "text": "Can I use multiple discount codes on a single purchase?", "true_complexity": "moderate"},
            {"id": "q7_ds_multi-hop", "text": "My product arrived damaged. What are the steps to initiate a replacement and how long will it take?", "true_complexity": "complex"},
            {"id": "q8", "text": "What payment methods do you accept?", "true_complexity": "simple"},
            {"id": "q9", "text": "How do I track my delivery?", "true_complexity": "simple"},
            {"id": "q10", "text": "I need information about a specific product's specifications and compatibility with other accessories.", "true_complexity": "moderate"}
        ]

    def simulate_benchmark_datasets_meta(self):
        return {
            "ecommerce_single_hop": {"bias": "single-hop", "description": "Questions answerable with one piece of information."},
            "ecommerce_multi_hop": {"bias": "multi-hop", "description": "Questions requiring combining multiple pieces of information."},
        }

    def route_query_to_resource(self, predicted_complexity):
        if predicted_complexity == "simple":
            return "Chatbot"
        elif predicted_complexity == "moderate":
            return "Advanced Bot"
        elif predicted_complexity == "complex":
            return "Human Agent"
        else:
            return "Unknown Route"

    def run(self):
        print("--- Starting Customer Support Router Simulation ---")

        # Simulate initial customer queries and benchmark datasets
        raw_customer_queries = self.simulate_customer_queries()
        benchmark_datasets_meta = self.simulate_benchmark_datasets_meta()
        print(f"\nSimulated {len(raw_customer_queries)} raw customer queries.")

        # Generate training data
        X_train, y_train = self.data_generator.generate_training_data(raw_customer_queries, benchmark_datasets_meta)
        print(f"Generated {len(X_train)} training data samples.")
        print(f"Label distribution in generated data: {Counter(y_train)}")

        # Train the query complexity classifier
        self.vectorizer, self.classifier = self.classifier_model.train_query_complexity_classifier(X_train, y_train)
        print("Query complexity classifier trained successfully.")

        # Simulate new incoming customer queries and route them
        new_incoming_queries = [
            "I need help with a refund for an item purchased last week.",
            "What are the technical specifications for the product 'XYZ' and its compatibility with third-party accessories?",
            "How do I change my account email?",
            "I want to understand the difference between standard and premium shipping options, including delivery times and costs.",
            "My order hasn't arrived. Can you check its status?"
        ]
        print("\n--- Routing New Incoming Queries ---")
        for i, query_text in enumerate(new_incoming_queries):
            predicted_complexity = self.classifier_model.predict_query_complexity(self.vectorizer, self.classifier, query_text)
            routing_decision = self.route_query_to_resource(predicted_complexity)
            print(f"Query {i+1}: '{query_text}'\nPredicted Complexity: {predicted_complexity}, Routed to: {routing_decision}\n")

        print("--- Simulation Complete ---")

if __name__ == "__main__":
    router = CustomerSupportRouter()
    router.run()
