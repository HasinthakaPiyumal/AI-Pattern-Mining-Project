import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Training Data Generation Module

# Simulated RAG Models
def simulate_no_rag(query):
    # Simple queries can be answered without retrieval
    simple_keywords = ["hello", "hi", "thank you", "order status", "return policy", "shipping cost"]
    is_simple = any(keyword in query.lower() for keyword in simple_keywords)
    if is_simple:
        return True, "Simple"
    return False, None

def simulate_single_step_rag(query):
    # Moderate queries might need single-step retrieval (e.g., product details)
    moderate_keywords = ["product details", "price of", "availability of", "specs for"]
    is_moderate = any(keyword in query.lower() for keyword in moderate_keywords)
    if is_moderate:
        return True, "Moderate"
    return False, None

def simulate_multi_step_rag(query):
    # Complex queries might need multi-step retrieval (e.g., comparison, troubleshooting)
    complex_keywords = ["compare", "troubleshoot", "how to set up", "compatibility with"]
    is_complex = any(keyword in query.lower() for keyword in complex_keywords)
    if is_complex:
        return True, "Complex"
    return False, None

def generate_data_from_model_outcomes(unlabeled_queries):
    labeled_data = []
    for query in unlabeled_queries:
        label = None
        # Prioritize simpler models
        if simulate_no_rag(query)[0]:
            label = "Simple"
        elif simulate_single_step_rag(query)[0]:
            label = "Moderate"
        elif simulate_multi_step_rag(query)[0]:
            label = "Complex"
        
        if label:
            labeled_data.append((query, label))
    return labeled_data

# Strategy 2 (Inherent Dataset Biases)
def generate_data_from_biases(num_samples=100):
    biased_data = []
    # Seed queries/patterns for different complexity levels
    simple_seeds = [
        "What is my order status?", "How can I return an item?", "What are the shipping costs?",
        "Hello, I have a question.", "Thank you for your help.", "Can I cancel my order?"
    ]
    moderate_seeds = [
        "Tell me about the features of product X.", "What is the price of the new smartphone?",
        "Is product Y available in red?", "What are the specifications of this laptop?",
        "How long is the warranty for product Z?"
    ]
    complex_seeds = [
        "Compare product A with product B, focusing on performance and battery life.",
        "I need help troubleshooting my smart home device connection issues.",
        "What are the steps to set up my new wireless printer with my old computer?",
        "Is this gaming console compatible with last year's VR headset and which games support both?",
        "Explain the difference between your premium and standard subscription plans in terms of features and billing cycles."
    ]

    for _ in range(num_samples // 3):
        biased_data.append((random.choice(simple_seeds), "Simple"))
        biased_data.append((random.choice(moderate_seeds), "Moderate"))
        biased_data.append((random.choice(complex_seeds), "Complex"))
    return biased_data

def generate_training_data(unlabeled_queries, num_bias_samples=100):
    data_from_models = generate_data_from_model_outcomes(unlabeled_queries)
    data_from_biases = generate_data_from_biases(num_bias_samples)
    
    # Combine and remove duplicates (if any query was labeled by both methods)
    combined_data = {query: label for query, label in (data_from_models + data_from_biases)}
    return list(combined_data.items())

# 2. Query Complexity Classifier Module
class QueryComplexityClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.classifier = LogisticRegression(max_iter=1000)

    def train(self, queries, labels):
        X = self.vectorizer.fit_transform(queries)
        self.classifier.fit(X, labels)

    def predict(self, query):
        X = self.vectorizer.transform([query])
        return self.classifier.predict(X)[0]

# 3. Adaptive Routing Module
def route_query(complexity_label, query):
    if complexity_label == "Simple":
        return "Routing to FAQ/No RAG. Response: {}".format(simulate_faq_response(query))
    elif complexity_label == "Moderate":
        return "Routing to Single-step RAG. Response: {}".format(simulate_single_step_rag_response(query))
    elif complexity_label == "Complex":
        return "Routing to Multi-step RAG. Response: {}".format(simulate_multi_step_rag_response(query))
    else:
        return "Routing to Human Agent. Reason: Unclassified or unusually complex query."

# Simulated LLM Strategy Responses
def simulate_faq_response(query):
    if "order status" in query.lower():
        return "Your order #12345 is currently being processed and is expected to ship within 2 business days."
    elif "return policy" in query.lower():
        return "Our return policy allows returns within 30 days of purchase with the original receipt."
    return "Please refer to our FAQ section for more information."

def simulate_single_step_rag_response(query):
    if "product features" in query.lower() or "specs for" in query.lower():
        return "Based on our database, the product X features include: [Feature 1, Feature 2, Feature 3]."
    return "Gathering information from relevant product documentation..."

def simulate_multi_step_rag_response(query):
    if "compare" in query.lower():
        return "Performing a detailed comparison of products A and B based on your criteria..."
    elif "troubleshoot" in query.lower():
        return "Initiating a multi-step troubleshooting guide for your device..."
    return "Accessing multiple knowledge bases and advanced reasoning for your complex inquiry..."

# 4. Overall Workflow
if __name__ == "__main__":
    # Example Unlabeled Queries (simulating incoming customer queries before training)
    unlabeled_customer_queries = [
        "Hi, I want to know my order status.",
        "What is your return policy?",
        "Can you tell me the specifications of the new XYZ laptop model?",
        "How do I troubleshoot the Wi-Fi connection on my smart TV?",
        "I'd like to compare the battery life of the Pro Max phone and the Ultra Plus phone.",
        "What are the shipping costs for international orders?",
        "My product arrived damaged, what should I do?",
        "Do you offer gift wrapping services?",
        "How can I reset my password?",
        "What are the environmental initiatives of your company?"
    ]

    print("--- Generating Training Data ---")
    training_data = generate_training_data(unlabeled_customer_queries, num_bias_samples=50)
    training_queries = [item[0] for item in training_data]
    training_labels = [item[1] for item in training_data]
    
    print(f"Generated {len(training_data)} training samples.")
    # print("Sample training data:")
    # for i in range(min(5, len(training_data))):
    #     print(f"  Query: '{training_data[i][0]}' -> Label: {training_data[i][1]}")

    # Split data for training and evaluation (optional, but good practice)
    if len(training_queries) > 1:
        X_train, X_test, y_train, y_test = train_test_split(training_queries, training_labels, test_size=0.2, random_state=42, stratify=training_labels)
    else: # Handle cases with very few samples for simplicity in this example
        X_train, y_train = training_queries, training_labels
        X_test, y_test = [], []

    print("--- Training Query Complexity Classifier ---")
    classifier = QueryComplexityClassifier()
    if X_train:
        classifier.train(X_train, y_train)
        print("Classifier trained successfully.")

        if X_test:
            y_pred = [classifier.predict(q) for q in X_test]
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Classifier accuracy on test set: {accuracy:.2f}")
    else:
        print("Not enough training data to train the classifier.")

    print("\n--- Processing New Customer Queries ---")
    new_queries = [
        "I need to know the dimensions of the X7 TV.",
        "My charging cable is not working with my phone, what should I do?",
        "What is the status of my recent order, ID 98765?",
        "Can you provide a detailed comparison of your eco-friendly packaging options?"
    ]

    for i, query in enumerate(new_queries):
        print(f"\nCustomer Query {i+1}: '{query}'")
        if X_train:
            predicted_complexity = classifier.predict(query)
            print(f"Predicted Complexity: {predicted_complexity}")
            routing_action = route_query(predicted_complexity, query)
            print(f"Routing Action: {routing_action}")
        else:
            print("Cannot classify, classifier not trained.")
            print(f"Routing Action: {route_query(None, query)}") # Route to human if no classifier