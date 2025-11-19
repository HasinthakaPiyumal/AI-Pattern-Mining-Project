from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import random

class QueryComplexityClassifier:
    """
    Classifies query complexity (e.g., 'simple_faq', 'order_issue', 'technical_problem').
    Also includes a method for generating synthetic training data.
    """
    def __init__(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', SVC(kernel='linear'))
        ])
        self.labels = ['simple_faq', 'order_issue', 'technical_problem', 'return_request', 'product_info']

    def generate_synthetic_data(self, num_samples=100):
        """
        Generates synthetic training data for the classifier.
        In a real-world scenario, this would involve LLMs or human annotation.
        """
        sample_queries = {
            'simple_faq': [
                "What are your shipping costs?", "How do I track my order?",
                "Do you offer international shipping?", "What is your return policy?",
                "How can I reset my password?"
            ],
            'order_issue': [
                "My order #12345 hasn't arrived yet.", "I received a wrong item in my order.",
                "My order was cancelled without my knowledge.", "I want to change my shipping address for order #67890.",
                "There's a problem with the payment for my recent purchase."
            ],
            'technical_problem': [
                "I can't log into my account.", "The website is not loading correctly.",
                "My discount code isn't working at checkout.", "I'm having trouble uploading product images.",
                "The search function on your site is broken."
            ],
            'return_request': [
                "I want to return item XYZ from order #11223.", "How do I initiate a return?",
                "Can I get a refund for a damaged product?", "I need to exchange a product.",
                "What's the process for returning a gift?"
            ],
            'product_info': [
                "Tell me more about product ABC.", "What are the specifications of Model X?",
                "Is product PQR available in blue?", "What's the warranty for this laptop?",
                "Do you have any reviews for item 123?"
            ]
        }

        texts = []
        labels = []
        for _ in range(num_samples):
            label = random.choice(self.labels)
            text = random.choice(sample_queries[label])
            texts.append(text)
            labels.append(label)
        return texts, labels

    def train(self, texts, labels):
        """
        Trains the query complexity classifier.
        Args:
            texts (list): A list of query strings.
            labels (list): A list of corresponding complexity labels.
        """
        print("Training Query Complexity Classifier...")
        self.pipeline.fit(texts, labels)
        print("Classifier trained successfully.")

    def predict(self, query):
        """
        Predicts the complexity label for a given query.
        Args:
            query (str): The input query string.
        Returns:
            str: The predicted complexity label.
        """
        if not hasattr(self.pipeline, 'classes_'):
            raise RuntimeError("Classifier not trained. Please call .train() first.")
        return self.pipeline.predict([query])[0]

# Example Usage (for testing/demonstration)
if __name__ == "__main__":
    classifier = QueryComplexityClassifier()

    # Generate and train on synthetic data
    texts, labels = classifier.generate_synthetic_data(num_samples=200)
    classifier.train(texts, labels)

    # Test predictions
    print(f"\nPrediction for 'Where is my order 98765?' : {classifier.predict('Where is my order 98765?')}")
    print(f"Prediction for 'What is your refund policy?' : {classifier.predict('What is your refund policy?')}")
    print(f"Prediction for 'My account is locked out.' : {classifier.predict('My account is locked out.')}")
    print(f"Prediction for 'I want to return this shirt.' : {classifier.predict('I want to return this shirt.')}")
    print(f"Prediction for 'Do you sell product XYZ?' : {classifier.predict('Do you sell product XYZ?')}")
