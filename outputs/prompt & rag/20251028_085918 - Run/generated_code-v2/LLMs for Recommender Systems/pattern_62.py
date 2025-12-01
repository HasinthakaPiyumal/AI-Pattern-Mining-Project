from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class DataLayer:
    def __init__(self):
        self.products = [
            {"id": "P001", "name": "Smartwatch X", "category": "Electronics", "description": "A sleek smartwatch with health tracking and notification features.", "reviews": ["Great battery life!", "Accurate heart rate monitor."]},
            {"id": "P002", "name": "Organic Coffee Beans", "category": "Groceries", "description": "Premium organic Arabica coffee beans for a rich, aromatic brew.", "reviews": ["Best coffee ever!", "Smooth taste."]},
            {"id": "P003", "name": "Noise-Cancelling Headphones", "category": "Electronics", "description": "Immersive audio experience with active noise cancellation and comfortable earcups.", "reviews": ["Perfect for travel.", "Sound quality is amazing."]},
            {"id": "P004", "name": "Yoga Mat Pro", "category": "Sports & Outdoors", "description": "High-density, non-slip yoga mat for all types of yoga and pilates.", "reviews": ["Very durable.", "Good grip."]},
            {"id": "P005", "name": "Ergonomic Office Chair", "category": "Furniture", "description": "Adjustable office chair designed for maximum comfort and posture support.", "reviews": ["My back feels so much better.", "Easy to assemble."]},
            {"id": "P006", "name": "Waterproof Bluetooth Speaker", "category": "Electronics", "description": "Portable speaker with crisp sound and waterproof design, perfect for outdoor use.", "reviews": ["Loud and clear sound.", "Takes a beating and keeps playing."]},
            {"id": "P007", "name": "Fiction Novel: The Quantum Paradox", "category": "Books", "description": "A gripping science fiction novel exploring parallel universes and time travel.", "reviews": ["Couldn't put it down!", "Mind-bending plot."]}
        ]

    def get_products(self):
        return self.products

class EmbeddingService:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.product_embeddings = {}

    def generate_embedding(self, text):
        return self.model.encode(text, convert_to_tensor=False)

    def precompute_product_embeddings(self, products):
        for product in products:
            combined_text = f"{product['name']}. {product['category']}. {product['description']}. {' '.join(product['reviews'])}"
            self.product_embeddings[product['id']] = self.generate_embedding(combined_text)

    def get_product_embedding(self, product_id):
        return self.product_embeddings.get(product_id)

    def get_all_product_embeddings(self):
        return self.product_embeddings

class RecommendationEngine:
    def __init__(self, embedding_service, products):
        self.embedding_service = embedding_service
        self.products = products
        self.product_id_to_index = {product['id']: i for i, product in enumerate(products)}
        self.index_to_product = {i: product for i, product in enumerate(products)}

    def recommend(self, query, top_k=3):
        query_embedding = self.embedding_service.generate_embedding(query)
        
        product_ids = list(self.embedding_service.get_all_product_embeddings().keys())
        product_embeddings_list = np.array([self.embedding_service.get_all_product_embeddings()[pid] for pid in product_ids])
        
        similarities = cosine_similarity([query_embedding], product_embeddings_list)[0]
        
        # Get indices of top_k most similar products
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        recommended_products = []
        for i in top_indices:
            product_id = product_ids[i]
            product_info = next((p for p in self.products if p['id'] == product_id), None)
            if product_info:
                recommended_products.append(product_info)
        
        return recommended_products

class ReasoningGenerator:
    def generate_explanation(self, user_query, recommended_product):
        product_name = recommended_product['name']
        product_description = recommended_product['description']
        product_category = recommended_product['category']
        
        # Simplified rule-based reasoning for demonstration
        explanation = f"We recommend '{product_name}' (Category: {product_category}) because it closely matches your interest in '{user_query}'. "
        
        if "health" in user_query.lower() or "fitness" in user_query.lower():
            if "health tracking" in product_description.lower() or "yoga" in product_description.lower():
                explanation += "Its features for health and fitness align with your query."
        elif "music" in user_query.lower() or "audio" in user_query.lower():
            if "sound" in product_description.lower() or "audio experience" in product_description.lower():
                explanation += "Its focus on audio quality and immersive sound is relevant to your search."
        elif "work" in user_query.lower() or "office" in user_query.lower():
            if "office chair" in product_description.lower() or "ergonomic" in product_description.lower():
                explanation += "This product is designed for comfort and productivity, suitable for your work needs."
        elif "book" in user_query.lower() or "read" in user_query.lower():
            if "novel" in product_description.lower() or "fiction" in product_description.lower():
                explanation += "This captivating novel fits your interest in reading and fiction."
        else:
             explanation += "Its description highlights features like " + product_description.lower() + ", which we found relevant."
        
        return explanation


class SemanticShopAPI:
    def __init__(self):
        self.data_layer = DataLayer()
        self.embedding_service = EmbeddingService()
        self.reasoning_generator = ReasoningGenerator()
        
        self.products = self.data_layer.get_products()
        self.embedding_service.precompute_product_embeddings(self.products)
        self.recommendation_engine = RecommendationEngine(self.embedding_service, self.products)

    def get_recommendations(self, user_query, top_k=3):
        recommended_products = self.recommendation_engine.recommend(user_query, top_k)
        
        results = []
        for product in recommended_products:
            explanation = self.reasoning_generator.generate_explanation(user_query, product)
            results.append({"product": product, "explanation": explanation})
        
        return results

if __name__ == "__main__":
    semantic_shop = SemanticShopAPI()

    print("\n--- SemanticShop Recommendations ---")
    
    queries = [
        "I need a good book to read",
        "something for my home office setup",
        "looking for a new gadget",
        "best coffee for mornings",
        "outdoor activity gear",
        "headphones for travel"
    ]

    for query in queries:
        print(f"\nUser Query: '{query}'")
        recommendations = semantic_shop.get_recommendations(query, top_k=2)
        for rec in recommendations:
            product = rec["product"]
            explanation = rec["explanation"]
            print(f"  Recommended: {product['name']} (Category: {product['category']})")
            print(f"  Explanation: {explanation}")
            print(f"  Description: {product['description']}")

    print("\n--- Demonstrating Cold Start (new product conceptual query) ---")
    new_product_query = "eco-friendly reusable water bottle for hiking"
    print(f"\nUser Query for a conceptual new product: '{new_product_query}'")
    cold_start_recs = semantic_shop.get_recommendations(new_product_query, top_k=1)
    for rec in cold_start_recs:
        product = rec["product"]
        explanation = rec["explanation"]
        print(f"  Recommended (Cold Start): {product['name']} (Category: {product['category']})")
        print(f"  Explanation: {explanation}")
        print(f"  Description: {product['description']}")