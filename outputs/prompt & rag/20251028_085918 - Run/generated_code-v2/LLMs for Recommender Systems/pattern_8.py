import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

class Product:
    def __init__(self, product_id, name, description, category, reviews=None):
        self.product_id = product_id
        self.name = name
        self.description = description
        self.category = category
        self.reviews = reviews if reviews is not None else []

    def get_text_content(self):
        return f"{self.name}. {self.description}. {' '.join(self.reviews)}"

class EmbeddingGenerator:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def generate_embedding(self, text):
        return self.model.encode(text, convert_to_numpy=True)

class VectorDatabase:
    def __init__(self, dimension):
        self.index = faiss.IndexFlatIP(dimension) # Inner Product for cosine similarity
        self.product_ids = []

    def add_product_embedding(self, product_id, embedding):
        self.index.add(np.expand_dims(embedding, axis=0))
        self.product_ids.append(product_id)

    def search_similar_products(self, query_embedding, k=5):
        D, I = self.index.search(np.expand_dims(query_embedding, axis=0), k)
        return [(self.product_ids[idx], D[0][i]) for i, idx in enumerate(I[0])]

class RecommendationEngine:
    def __init__(self, products_data):
        self.products = {p.product_id: p for p in products_data}
        self.embedding_generator = EmbeddingGenerator()
        
        # Initialize FAISS index with the dimension of the embeddings
        # We'll generate one embedding to get the dimension
        sample_embedding = self.embedding_generator.generate_embedding("sample text")
        self.vector_db = VectorDatabase(sample_embedding.shape[0])
        
        self._build_product_index()

    def _build_product_index(self):
        print("Building product embeddings and FAISS index...")
        for product_id, product in self.products.items():
            text_content = product.get_text_content()
            embedding = self.embedding_generator.generate_embedding(text_content)
            self.vector_db.add_product_embedding(product_id, embedding)
        print("Product index built.")

    def get_recommendations_by_query(self, query_text, k=5):
        query_embedding = self.embedding_generator.generate_embedding(query_text)
        similar_products_with_scores = self.vector_db.search_similar_products(query_embedding, k)
        
        recommendations = []
        for product_id, score in similar_products_with_scores:
            product = self.products.get(product_id)
            if product:
                recommendations.append({
                    "product_id": product.product_id,
                    "name": product.name,
                    "description": product.description,
                    "category": product.category,
                    "similarity_score": float(score) # Convert numpy float to Python float
                })
        return recommendations

    def get_recommendations_for_product(self, target_product_id, k=5):
        if target_product_id not in self.products:
            return []
        
        target_product = self.products[target_product_id]
        target_embedding = self.embedding_generator.generate_embedding(target_product.get_text_content())
        
        similar_products_with_scores = self.vector_db.search_similar_products(target_embedding, k+1) # +1 to exclude itself
        
        recommendations = []
        for product_id, score in similar_products_with_scores:
            if product_id == target_product_id: # Skip the product itself
                continue
            product = self.products.get(product_id)
            if product:
                recommendations.append({
                    "product_id": product.product_id,
                    "name": product.name,
                    "description": product.description,
                    "category": product.category,
                    "similarity_score": float(score)
                })
        return recommendations[:k]

# Example Usage
if __name__ == "__main__":
    # Simulate Product Catalog Data
    product_data = [
        Product("P001", "Wireless Bluetooth Headphones", "High-quality sound, comfortable earcups, 20-hour battery life.", "Electronics", ["Great sound for the price!", "Battery lasts forever."]),
        Product("P002", "Ergonomic Office Chair", "Adjustable lumbar support, breathable mesh, suitable for long working hours.", "Office Furniture", ["Very comfortable, my back feels much better.", "Easy to assemble."]),
        Product("P003", "Portable SSD 1TB", "Fast data transfer, compact design, USB-C compatible.", "Electronics", ["Super fast, perfect for my laptop.", "Small and fits in my pocket."]),
        Product("P004", "Noise Cancelling Earbuds", "Active noise cancellation, secure fit, perfect for commuting.", "Electronics", ["Amazing for travel, blocks out all noise.", "Fits well in my ears."]),
        Product("P005", "Smart Home Security Camera", "1080p HD video, motion detection, two-way audio, cloud storage.", "Smart Home", ["Easy setup, clear video quality.", "Love the motion alerts."]),
        Product("P006", "Organic Green Tea Bags", "Premium quality organic green tea, rich in antioxidants, 100 tea bags.", "Groceries", ["Delicious and refreshing.", "Good for health."]),
    ]

    # Initialize the Recommendation Engine
    recommender = RecommendationEngine(product_data)

    print("\n--- Recommendations based on user query --- ")
    query = "headphones for music"
    print(f"Query: '{query}'")
    recommendations = recommender.get_recommendations_by_query(query, k=2)
    for rec in recommendations:
        print(f"- {rec['name']} (Category: {rec['category']}, Score: {rec['similarity_score']:.4f})")

    print("\n--- Recommendations for a specific product (P001 - Wireless Bluetooth Headphones) --- ")
    product_id_to_recommend_for = "P001"
    recommendations = recommender.get_recommendations_for_product(product_id_to_recommend_for, k=3)
    for rec in recommendations:
        print(f"- {rec['name']} (Category: {rec['category']}, Score: {rec['similarity_score']:.4f})")

    print("\n--- Recommendations for a cold-start query (new product category) --- ")
    cold_start_query = "comfortable chair for office work"
    print(f"Query: '{cold_start_query}'")
    recommendations = recommender.get_recommendations_by_query(cold_start_query, k=1)
    for rec in recommendations:
        print(f"- {rec['name']} (Category: {rec['category']}, Score: {rec['similarity_score']:.4f})")

    print("\n--- Recommendations for another cold-start query (related to home security) --- ")
    another_cold_start_query = "camera for home surveillance"
    print(f"Query: '{another_cold_start_query}'")
    recommendations = recommender.get_recommendations_by_query(another_cold_start_query, k=1)
    for rec in recommendations:
        print(f"- {rec['name']} (Category: {rec['category']}, Score: {rec['similarity_score']:.4f})")
