from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 1. Product Data Representation
products_data = [
    {
        "id": "P001",
        "name": "Smartwatch Series 7",
        "description": "Advanced smartwatch with health tracking, GPS, and water resistance. Features a vibrant Retina display.",
        "reviews": [
            "Great watch for fitness, battery life is decent.",
            "Love the design and seamless integration with my phone.",
            "A bit expensive but worth it for the features."
        ]
    },
    {
        "id": "P002",
        "name": "Noise-Cancelling Headphones Pro",
        "description": "Over-ear headphones with industry-leading noise cancellation, immersive sound, and comfortable earcups.",
        "reviews": [
            "The sound quality is amazing!",
            "Perfect for travel, completely blocks out airplane noise.",
            "Comfortable for long listening sessions."
        ]
    },
    {
        "id": "P003",
        "name": "Ultra HD 4K Monitor 27-inch",
        "description": "27-inch 4K UHD monitor with HDR support, ideal for graphic designers and gamers. Features multiple input ports.",
        "reviews": [
            "Stunning clarity and color accuracy.",
            "Great for coding and video editing.",
            "Wish it had a higher refresh rate for gaming."
        ]
    },
    {
        "id": "P004",
        "name": "Ergonomic Office Chair",
        "description": "Adjustable ergonomic chair designed for maximum comfort and posture support during long working hours. Breathable mesh fabric.",
        "reviews": [
            "My back pain is gone since I started using this chair.",
            "Easy to assemble and very comfortable.",
            "A bit pricey, but a good investment for health."
        ]
    },
    {
        "id": "P005",
        "name": "Portable Bluetooth Speaker",
        "description": "Compact and waterproof Bluetooth speaker with 360-degree sound and long battery life. Perfect for outdoor activities.",
        "reviews": [
            "Excellent sound for its size, very durable.",
            "Took it to the beach, worked perfectly.",
            "Battery lasts all day, highly recommend."
        ]
    },
]

# 2. Model Loading
# Using a pre-trained SentenceTransformer model for generating embeddings
# 'all-MiniLM-L6-v2' is a good balance of performance and speed.
model = SentenceTransformer('all-MiniLM-L6-v2')

# 3. Embedding Generation (for products)
product_texts = []
for product in products_data:
    combined_text = f"{product['name']}. {product['description']}. {' '.join(product['reviews'])}"
    product_texts.append(combined_text)

product_embeddings = model.encode(product_texts, convert_to_numpy=True)

# 4. Recommendation Function
def recommend_products(user_query: str, top_n: int = 3):
    # Encode the user query
    query_embedding = model.encode([user_query], convert_to_numpy=True)

    # Calculate cosine similarity between query and all product embeddings
    similarities = cosine_similarity(query_embedding, product_embeddings)[0]

    # Get indices of top_n most similar products
    top_n_indices = np.argsort(similarities)[::-1][:top_n]

    # Retrieve and return recommended products
    recommended_products = []
    for i in top_n_indices:
        product_info = products_data[i].copy()
        product_info["similarity_score"] = similarities[i]
        recommended_products.append(product_info)
    return recommended_products

# 5. Example Usage
if __name__ == "__main__":
    print("\n--- E-commerce Product Recommender ---")

    # Example 1: General query
    query1 = "I need a new pair of headphones for listening to music at home and while traveling."
    print(f"\nRecommendations for: '{query1}'")
    recs1 = recommend_products(query1, top_n=2)
    for rec in recs1:
        print(f"  - {rec['name']} (Score: {rec['similarity_score']:.4f})")

    # Example 2: Fitness related item
    query2 = "Show me smart devices to track my health and workouts."
    print(f"\nRecommendations for: '{query2}'")
    recs2 = recommend_products(query2, top_n=1)
    for rec in recs2:
        print(f"  - {rec['name']} (Score: {rec['similarity_score']:.4f})")

    # Example 3: Office equipment
    query3 = "Looking for comfortable seating for long hours at my desk."
    print(f"\nRecommendations for: '{query3}'")
    recs3 = recommend_products(query3, top_n=2)
    for rec in recs3:
        print(f"  - {rec['name']} (Score: {rec['similarity_score']:.4f})")

    # Example 4: Cold-start product (hypothetical, as all products are embedded)
    # This demonstrates how even without user history, the rich product description
    # and reviews help in making relevant recommendations.
    query4 = "I want a speaker that I can take to the beach and listen to music."
    print(f"\nRecommendations for: '{query4}'")
    recs4 = recommend_products(query4, top_n=1)
    for rec in recs4:
        print(f"  - {rec['name']} (Score: {rec['similarity_score']:.4f})")