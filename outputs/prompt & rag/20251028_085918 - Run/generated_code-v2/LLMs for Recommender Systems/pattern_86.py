from model_loader import load_embedding_model
from recommender import ProductRecommender

def main():
    print("Starting Smart Product Recommender for E-commerce...")

    # 1. Load the embedding model
    # Simulating a distilled/optimized LLM for content interpretation
    try:
        embedding_model = load_embedding_model("all-MiniLM-L6-v2") 
    except Exception as e:
        print(f"Failed to load embedding model. Exiting. Error: {e}")
        return

    # 2. Initialize the Product Recommender
    recommender = ProductRecommender(embedding_model)

    # 3. Add example product data
    # In a real-world scenario, this would come from a database or product catalog
    products_data = [
        {"id": "P001", "description": "High-performance gaming laptop with RTX 3080 and 32GB RAM."},
        {"id": "P002", "description": "Lightweight ultrabook for productivity, 13-inch display, 8GB RAM."},
        {"id": "P003", "description": "Ergonomic mechanical keyboard with silent switches for coding."},
        {"id": "P004", "description": "Wireless noise-cancelling headphones with superb audio quality."},
        {"id": "P005", "description": "4K Smart TV, 65-inch, HDR support, voice control."},
        {"id": "P006", "description": "Portable Bluetooth speaker with long battery life and deep bass."},
        {"id": "P007", "description": "Gaming mouse with customizable RGB lighting and high DPI sensor."},
        {"id": "P008", "description": "Professional studio monitor headphones for audio mixing."},
        {"id": "P009", "description": "Budget-friendly smartphone with a great camera and all-day battery."},
        {"id": "P010", "description": "Fitness tracker smartwatch with heart rate monitoring and GPS."}
    ]
    print("Adding products to the recommender...")
    recommender.add_products(products_data)

    # 4. Get recommendations based on user queries
    queries = [
        "Looking for a new laptop for gaming.",
        "I need headphones for listening to music quietly.",
        "Show me some smart devices for home entertainment.",
        "What kind of keyboard is good for programming?",
        "Affordable phone with good camera."
    ]

    for query in queries:
        print(f"\n--- User Query: {query} ---")
        recommendations = recommender.get_recommendations(query, top_k=3)
        if recommendations:
            print("Recommended Products:")
            for rec in recommendations:
                print(f"  - ID: {rec["id"]}, Description: \"{rec["description"][:70]}...\", Score (Similarity): {rec["score"]:.4f}")
        else:
            print("No recommendations found.")

    print("\nSmart Product Recommender finished.")

if __name__ == "__main__":
    main()