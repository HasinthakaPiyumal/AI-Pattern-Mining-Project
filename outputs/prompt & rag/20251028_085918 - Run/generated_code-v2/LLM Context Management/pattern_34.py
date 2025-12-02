import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

def simulate_no_retrieval(query: str) -> bool:
    return any(keyword in query.lower() for keyword in ["hello", "hi", "greetings", "thanks", "thank you"])

def simulate_single_step_rag(query: str) -> bool:
    return any(keyword in query.lower() for keyword in ["order status", "shipping cost", "return policy", "product details"])

def simulate_multi_step_rag(query: str) -> bool:
    return any(keyword in query.lower() for keyword in ["troubleshoot", "compare", "compatibility", "how to fix", "setup instructions"])

def generate_training_data(raw_queries: list[dict]) -> list[dict]:
    labeled_data = []

    for item in raw_queries:
        query = item["query"]
        label = "unlabeled"

        if simulate_no_retrieval(query):
            label = "low"
        elif simulate_single_step_rag(query):
            label = "moderate"
        elif simulate_multi_step_rag(query):
            label = "high"
        
        labeled_data.append({"query": query, "complexity_label": label, "source": item.get("source", "unknown")})
    
    final_labeled_data = []
    for item in labeled_data:
        if item["complexity_label"] == "unlabeled":
            source = item["source"]
            if source == "FAQ_log":
                item["complexity_label"] = "low"
            elif source == "product_page_search":
                item["complexity_label"] = "moderate"
            elif source == "troubleshooting_session":
                item["complexity_label"] = "high"
            else:
                item["complexity_label"] = random.choice(["low", "moderate", "high"])
        final_labeled_data.append({"query": item["query"], "complexity_label": item["complexity_label"]})

    return final_labeled_data

class QueryComplexityClassifier:

    def __init__(self):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(stop_words="english", max_features=1000)),
            ("classifier", RandomForestClassifier(random_state=42))
        ])

    def train(self, data: list[dict]):
        if not data:
            print("No data provided for training the classifier.")
            return
            
        queries = [item["query"] for item in data]
        labels = [item["complexity_label"] for item in data]
        self.pipeline.fit(queries, labels)
        print(f"Classifier trained on {len(data)} samples.")

    def predict(self, query: str) -> str:
        if not hasattr(self, 'pipeline') or not self.pipeline.named_steps['classifier'].classes_.size > 0:
            print("Classifier not trained. Returning 'unknown'.")
            return "unknown"
        return self.pipeline.predict([query])[0]

def get_rag_strategy(complexity_label: str) -> str:
    if complexity_label == "low":
        return "No Retrieval / Direct LLM Response (e.g., simple greeting or very common FAQ)"
    elif complexity_label == "moderate":
        return "Single-Step RAG (e.g., FAQ lookup, simple knowledge base query)"
    elif complexity_label == "high":
        return "Multi-Step RAG (e.g., complex knowledge base search, troubleshooting guide, multi-turn dialogue)"
    else:
        return "Default RAG Strategy (unknown complexity)"

def answer_query_adaptively(query: str, classifier: QueryComplexityClassifier) -> str:
    print(f"\n--- Processing Query: '{query}' ---")
    predicted_complexity = classifier.predict(query)
    selected_strategy = get_rag_strategy(predicted_complexity)
    
    print(f"Predicted Complexity: {predicted_complexity}")
    print(f"Selected RAG Strategy: {selected_strategy}")
    
    if predicted_complexity == "low":
        response = "Hello! How can I help you today?"
    elif predicted_complexity == "moderate":
        response = f"Searching our FAQs for '{query}'. Here's what I found... (simulated FAQ answer)"
    elif predicted_complexity == "high":
        response = f"Initiating a multi-step knowledge base search and diagnostic process for '{query}'. Please bear with me... (simulated complex resolution)"
    else:
        response = "I'm sorry, I'm having trouble understanding. Can you please rephrase?"
        
    return response

def main():
    print("Starting Intelligent Customer Support Assistant setup...")

    raw_customer_queries = [
        {"query": "Hi, I need help.", "source": "chat_initiation"},
        {"query": "What is my order status?", "source": "FAQ_log"},
        {"query": "How much does shipping cost to New York?", "source": "FAQ_log"},
        {"query": "My device is not turning on, what should I do?", "source": "troubleshooting_session"},
        {"query": "Can you compare the features of model A and model B smartwatches?", "source": "product_page_search"},
        {"query": "I want to return a product. What's the process?", "source": "FAQ_log"},
        {"query": "The Bluetooth connection keeps dropping on my headphones.", "source": "troubleshooting_session"},
        {"query": "What are your business hours?", "source": "FAQ_log"},
        {"query": "I need detailed instructions for setting up my new router.", "source": "troubleshooting_session"},
        {"query": "What's the difference between OLED and QLED TVs?", "source": "product_page_search"},
        {"query": "Thank you!", "source": "chat_end"},
        {"query": "Where is my parcel?", "source": "FAQ_log"},
        {"query": "I'm having trouble with my account login.", "source": "troubleshooting_session"},
        {"query": "Are pets allowed in your stores?", "source": "FAQ_log"},
        {"query": "Explain how the warranty works for refurbished items.", "source": "product_page_search"}
    ]

    print("\nGenerating training data...")
    training_data = generate_training_data(raw_customer_queries)
    print("Generated Training Data Samples:")
    for i, sample in enumerate(training_data[:5]):
        print(f"  {i+1}. Query: '{sample['query']}' -> Label: '{sample['complexity_label']}'")
    print(f"... Total {len(training_data)} samples generated.")

    print("\nInitializing and training Query Complexity Classifier...")
    classifier = QueryComplexityClassifier()
    classifier.train(training_data)

    print("\nDemonstrating Adaptive Customer Support Assistant:")
    test_queries = [
        "Hello, assistant.",
        "What is the status of my recent order?",
        "My gaming laptop is overheating constantly, what diagnostics can I run?",
        "Can I get a refund for a digital purchase?",
        "How do I connect my smart home device to Wi-Fi if it's not showing up?",
        "Thanks for your help!"
    ]

    for query in test_queries:
        response = answer_query_adaptively(query, classifier)
        print(f"Assistant Response: {response}")

    print("\nIntelligent Customer Support Assistant demonstration finished.")

if __name__ == "__main__":
    main()