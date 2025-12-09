import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
import joblib
import os

# --- 1. Automatic Training Data Generation Pipeline (data_generator.py logic) ---

def no_retrieval_llm(query):
    if any(keyword in query.lower() for keyword in ["return policy", "shipping cost", "account login"]):
        return "Answered by No Retrieval LLM: Basic info provided.", True
    return None, False

def single_step_rag_llm(query):
    if any(keyword in query.lower() for keyword in ["track order", "product details", "payment options"]):
        return "Answered by Single-Step RAG LLM: Specific product/order info.", True
    return None, False

def multi_step_rag_llm(query):
    if any(keyword in query.lower() for keyword in ["troubleshoot", "warranty claim", "technical issue", "how to assemble"]):
        return "Answered by Multi-Step RAG LLM: Complex issue resolution.", True
    return None, False

def generate_training_data(queries):
    labeled_data = []
    for query in queries:
        label = None

        response, success = no_retrieval_llm(query)
        if success:
            label = "simple"
        else:
            response, success = single_step_rag_llm(query)
            if success:
                label = "moderate"
            else:
                response, success = multi_step_rag_llm(query)
                if success:
                    label = "complex"
        
        # Strategy 2: Inherent Dataset Biases - Fallback
        if label is None:
            if any(keyword in query.lower() for keyword in ["what is", "where is"]):
                label = "simple"
            elif any(keyword in query.lower() for keyword in ["when will", "my order"]):
                label = "moderate"
            elif any(keyword in query.lower() for keyword in ["fix", "problem with"]):
                label = "complex"
            else:
                label = random.choice(["simple", "moderate", "complex"]) # Assign randomly if no heuristic matches

        labeled_data.append({"query": query, "complexity_label": label})

    return pd.DataFrame(labeled_data)

if __name__ == "__main__":
    # Simulate raw customer queries
    sample_queries = [
        "What is your return policy?",
        "How can I track my order #12345?",
        "I have a problem with my new blender, it's not turning on.",
        "When will my delivery arrive?",
        "Can I change my account password?",
        "What are the product details for item XYZ?",
        "I need help troubleshooting my smart speaker connection.",
        "Where is the nearest store?",
        "How do I claim my warranty for laptop ABC?",
        "Tell me about your shipping costs.",
        "My headphones are not pairing, how to fix it?",
        "What payment methods do you accept?"
    ]

    # Generate training data
    training_data_df = generate_training_data(sample_queries)
    training_data_df.to_csv("training_data.csv", index=False)
    print("Generated training_data.csv:")
    print(training_data_df.head())

    # --- 2. Query Complexity Classifier (query_complexity_classifier.py logic) ---
    
    # Load the generated training data
    training_data = pd.read_csv("training_data.csv")

    # Initialize Sentence Transformer model
    model_name = "all-MiniLM-L6-v2"
    try:
        embedding_model = SentenceTransformer(model_name)
    except Exception as e:
        print(f"Could not load SentenceTransformer model {model_name}. Please ensure you have an internet connection or the model is cached. Error: {e}")
        print("Attempting to use a placeholder for embeddings. This will not work correctly without the actual model.")
        class PlaceholderEmbeddingModel:
            def encode(self, texts, convert_to_tensor=False):
                return [[0.0] * 384 for _ in texts] # Dummy embeddings
        embedding_model = PlaceholderEmbeddingModel()

    # Generate embeddings
    X = embedding_model.encode(training_data["query"].tolist(), convert_to_tensor=False)

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(training_data["complexity_label"])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train a classifier
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X_train, y_train)

    # Evaluate
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nClassifier Accuracy: {accuracy:.2f}")

    # Save the trained model and label encoder
    joblib.dump(classifier, "qcc_model.pkl")
    joblib.dump(label_encoder, "label_encoder.pkl")
    print("Trained classifier and label encoder saved.")

    # --- 3. Adaptive LLM System (adaptive_llm_system.py logic) ---

    # Load the trained models
    loaded_classifier = joblib.load("qcc_model.pkl")
    loaded_label_encoder = joblib.load("label_encoder.pkl")
    
    # Reuse the embedding_model loaded earlier if available, otherwise attempt to load again or use placeholder
    try:
        if 'embedding_model' not in locals() or isinstance(embedding_model, PlaceholderEmbeddingModel):
            embedding_model = SentenceTransformer(model_name)
    except Exception:
        print("Using placeholder embedding model for adaptive LLM system due to loading failure.")

    def adaptive_llm_response(query):
        # Get query embedding
        query_embedding = embedding_model.encode([query], convert_to_tensor=False)

        # Predict complexity
        predicted_label_encoded = loaded_classifier.predict(query_embedding)[0]
        predicted_complexity = loaded_label_encoder.inverse_transform([predicted_label_encoded])[0]

        print(f"\nQuery: '{query}'")
        print(f"Predicted Complexity: {predicted_complexity}")

        # Route to appropriate LLM strategy
        if predicted_complexity == "simple":
            response, _ = no_retrieval_llm(query)
            return response if response else "Simple query handled by No Retrieval LLM fallback."
        elif predicted_complexity == "moderate":
            response, _ = single_step_rag_llm(query)
            return response if response else "Moderate query handled by Single-Step RAG LLM fallback."
        elif predicted_complexity == "complex":
            response, _ = multi_step_rag_llm(query)
            return response if response else "Complex query handled by Multi-Step RAG LLM fallback."
        else:
            return "Could not determine query complexity or route request."

    # Demonstrate with new queries
    new_queries = [
        "Where can I find my order history?",
        "My washing machine is making a loud noise, what should I do?",
        "What is the price of product X?",
        "How do I set up my new router?"
    ]

    for query in new_queries:
        response = adaptive_llm_response(query)
        print(f"System Response: {response}")

    # Clean up generated files
    try:
        os.remove("training_data.csv")
        os.remove("qcc_model.pkl")
        os.remove("label_encoder.pkl")
        print("\nCleaned up generated files: training_data.csv, qcc_model.pkl, label_encoder.pkl")
    except OSError as e:
        print(f"Error cleaning up files: {e}")
