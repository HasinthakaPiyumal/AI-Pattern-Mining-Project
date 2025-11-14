import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import faiss

# --- 1. Data Layer Simulation ---
# In a real application, these would be retrieved from databases.
product_catalog_data = [
    {"id": "P001", "name": "UltraBook Pro", "description": "Powerful laptop with 16GB RAM, 1TB SSD, and high-resolution display. Ideal for professionals and heavy tasks.", "category": "Electronics", "price": 1500, "key_feature": "performance", "highlight": "speed"},
    {"id": "P002", "name": "EcoLap 3000", "description": "Energy-efficient laptop with long battery life, lightweight design, and basic functionalities. Perfect for students and casual use.", "category": "Electronics", "price": 800, "key_feature": "battery life", "highlight": "endurance"},
    {"id": "P003", "name": "SpeedRunner X", "description": "Lightweight running shoes with advanced cushioning technology, breathable mesh upper, and strong grip. Best for marathons and serious runners.", "category": "Footwear", "price": 120, "key_feature": "cushioning", "highlight": "comfort"},
    {"id": "P004", "name": "SprintFlow 5", "description": "Breathable and comfortable running shoes for daily jogs and casual athletic activities. Very versatile.", "category": "Footwear", "price": 90, "key_feature": "breathability", "highlight": "lightweight"},
    {"id": "P005", "name": "Classic Denim Jeans", "description": "Durable and stylish blue denim jeans for everyday wear. Classic fit.", "category": "Apparel", "price": 60, "key_feature": "durability", "highlight": "style"},
    {"id": "P006", "name": "Wireless Ergonomic Mouse", "description": "Comfortable wireless mouse designed for extended use, reducing strain. Adjustable DPI settings.", "category": "Electronics", "price": 45, "key_feature": "ergonomics", "highlight": "comfort"},
    {"id": "P007", "name": "Noise-Cancelling Headphones", "description": "Premium over-ear headphones with active noise cancellation, crystal-clear audio, and long-lasting battery. Immerse yourself in music.", "category": "Electronics", "price": 250, "key_feature": "noise cancellation", "highlight": "audio quality"},
    {"id": "P008", "name": "Yoga Mat Pro", "description": "High-density yoga mat for superior grip, comfort, and stability during intense yoga sessions. Eco-friendly material.", "category": "Sports & Outdoors", "price": 50, "key_feature": "grip", "highlight": "stability"},
    {"id": "P009", "name": "Bluetooth Speaker Mini", "description": "Compact and portable Bluetooth speaker with surprisingly rich sound. Ideal for outdoor use.", "category": "Electronics", "price": 75, "key_feature": "portability", "highlight": "sound quality"},
    {"id": "P010", "name": "Organic Green Tea", "description": "Premium organic green tea leaves for a refreshing and healthy beverage. Rich in antioxidants.", "category": "Food & Beverage", "price": 25, "key_feature": "organic", "highlight": "health benefits"},
]

user_interactions_data = [
    {"user_id": "U001", "product_id": "P001", "interaction_type": "view", "timestamp": "2023-10-26T10:00:00"},
    {"user_id": "U001", "product_id": "P006", "interaction_type": "view", "timestamp": "2023-10-26T10:05:00"},
    {"user_id": "U001", "product_id": "P001", "interaction_type": "purchase", "timestamp": "2023-10-26T11:00:00"},
    {"user_id": "U001", "product_id": "P007", "interaction_type": "view", "timestamp": "2023-10-26T11:30:00"},
    {"user_id": "U002", "product_id": "P003", "interaction_type": "view", "timestamp": "2023-10-26T12:00:00"},
    {"user_id": "U002", "product_id": "P004", "interaction_type": "view", "timestamp": "2023-10-26T12:10:00"},
    {"user_id": "U002", "product_id": "P003", "interaction_type": "purchase", "timestamp": "2023-10-26T12:30:00"},
    {"user_id": "U003", "product_id": "P005", "interaction_type": "view", "timestamp": "2023-10-26T13:00:00"},
    {"user_id": "U003", "product_id": "P007", "interaction_type": "view", "timestamp": "2023-10-26T13:05:00"},
    {"user_id": "U003", "product_id": "P009", "interaction_type": "view", "timestamp": "2023-10-26T13:15:00"},
    {"user_id": "U004", "product_id": "P008", "interaction_type": "view", "timestamp": "2023-10-26T14:00:00"},
    {"user_id": "U004", "product_id": "P010", "interaction_type": "view", "timestamp": "2023-10-26T14:05:00"},
]

# --- Mock Embedding Model (in case SentenceTransformer fails to load) ---
class MockSentenceTransformer:
    def encode(self, text, convert_to_tensor=True):
        # Returns a fixed-size random vector for mock purposes
        if isinstance(text, list):
            return np.array([np.random.rand(384) for _ in text]) # 384 is common dimension for 'all-MiniLM-L6-v2'
        return np.random.rand(384)

# --- 2.1 Recommender System Service ---
class RecommenderService:
    def __init__(self, products, interactions, embedding_model):
        self.products = products
        self.interactions = interactions
        self.embedding_model = embedding_model
        self.product_id_to_idx = {p['id']: i for i, p in enumerate(self.products)}
        self.idx_to_product_id = {i: p['id'] for i, p in enumerate(self.products)}
        self.product_embeddings = self._get_product_embeddings()

    def _get_product_embeddings(self):
        product_descriptions = [p['description'] for p in self.products]
        return self.embedding_model.encode(product_descriptions)

    def get_content_based_recommendations(self, query_embedding, excluded_product_ids=None, top_n=5):
        if excluded_product_ids is None:
            excluded_product_ids = []

        similarities = cosine_similarity(query_embedding.reshape(1, -1), self.product_embeddings)[0]
        sorted_indices = np.argsort(similarities)[::-1]

        recommendations = []
        for idx in sorted_indices:
            product = self.products[idx]
            if product['id'] not in excluded_product_ids:
                recommendations.append(product)
                if len(recommendations) >= top_n:
                    break
        return recommendations

    def get_collaborative_recommendations(self, user_id, excluded_product_ids=None, top_n=5):
        if excluded_product_ids is None:
            excluded_product_ids = []

        # Simplified Collaborative Filtering: Recommend popular items from categories the user interacted with
        user_viewed_categories = set()
        for interaction in self.interactions:
            if interaction['user_id'] == user_id:
                product = next((p for p in self.products if p['id'] == interaction['product_id']), None)
                if product: 
                    user_viewed_categories.add(product['category'])

        candidate_products = []
        for p in self.products:
            if p['category'] in user_viewed_categories and p['id'] not in excluded_product_ids:
                candidate_products.append(p)
        
        # Sort by a dummy popularity score (e.g., higher price for demonstration)
        candidate_products.sort(key=lambda x: x['price'], reverse=True)

        return candidate_products[:top_n]

    def get_hybrid_recommendations(self, user_id, user_query=None, excluded_product_ids=None, top_n=5):
        if excluded_product_ids is None:
            excluded_product_ids = []

        content_recs = []
        if user_query:
            query_embedding = self.embedding_model.encode(user_query)
            content_recs = self.get_content_based_recommendations(query_embedding, excluded_product_ids=excluded_product_ids, top_n=top_n)

        collab_recs = self.get_collaborative_recommendations(user_id, excluded_product_ids=excluded_product_ids, top_n=top_n)

        # Simple merging strategy: prioritize content-based if query, then add collaborative
        combined_recommendations = []
        seen_product_ids = set(excluded_product_ids)

        for rec in content_recs:
            if rec['id'] not in seen_product_ids:
                combined_recommendations.append(rec)
                seen_product_ids.add(rec['id'])
        
        for rec in collab_recs:
            if rec['id'] not in seen_product_ids:
                combined_recommendations.append(rec)
                seen_product_ids.add(rec['id'])

        return combined_recommendations[:top_n]

# --- 2.2 LLM Orchestration Service ---
class LLMOrchestrationService:
    def __init__(self, products, recommender_service, embedding_model):
        self.products = products
        self.recommender_service = recommender_service
        self.embedding_model = embedding_model
        self.product_id_to_idx = {p['id']: i for i, p in enumerate(self.products)}
        self.idx_to_product_id = {i: p['id'] for i, p in enumerate(self.products)}

        # Initialize FAISS index for product search
        product_embeddings_for_faiss = np.array([self.embedding_model.encode(p['description']) for p in self.products]).astype('float32')
        self.faiss_index = faiss.IndexFlatIP(product_embeddings_for_faiss.shape[1])
        self.faiss_index.add(product_embeddings_for_faiss)

    def generate_explanation(self, user_id, recommended_product_id):
        product = next((p for p in self.products if p['id'] == recommended_product_id), None)
        if not product:
            return "Could not find product details for explanation."

        # Simulate LLM generating an explanation
        # In a real scenario, this would involve calling an actual LLM API with a detailed prompt
        # Prompt example: "Explain why '{product_name}' was recommended to user {user_id} given product features: {product_features}"
        explanation_template = (
            f"Based on your preferences and the features of '{product['name']}' ({product['category']}), "
            f"we recommend this product because of its excellent '{product.get('key_feature', 'quality')}' "
            f"and positive attributes. It aligns with similar items you've shown interest in or products that complement your past purchases. "
            f"Its description highlights: '{product['description']}'. Customers often praise its '{product.get('highlight', 'durability')}' and its overall value."
        )
        return explanation_template

    def conversational_search(self, user_query, user_id=None, chat_history=None, top_k=3):
        query_embedding = self.embedding_model.encode(user_query).astype('float32')

        # RAG-like approach: Retrieve relevant products from FAISS
        D, I = self.faiss_index.search(query_embedding.reshape(1, -1), top_k)
        retrieved_product_ids = [self.idx_to_product_id[idx] for idx in I[0]]
        retrieved_products = [next(p for p in self.products if p['id'] == pid) for pid in retrieved_product_ids]

        context_products_str = "\n".join([f"- {p['name']} ({p['category']}): {p['description']}" for p in retrieved_products])

        # Simulate LLM response for conversational search
        # In a real system, this would be a sophisticated prompt to an LLM, 
        # incorporating chat history and retrieved context.
        llm_prompt = (
            f"User query: '{user_query}'\n\n"
            f"Relevant Products Found:\n{context_products_str}\n\n"
            "Based on the query and relevant products, answer the user's question or provide suitable recommendations. "
            "Keep the response helpful, concise, and engaging."
        )
        
        # Hardcoded responses for demonstration, simulating LLM intelligence
        if "laptop" in user_query.lower() and "powerful" in user_query.lower():
            return f"For a powerful laptop, I highly recommend the 'UltraBook Pro'. It features 16GB RAM and a 1TB SSD, perfect for heavy tasks. Would you like to know more about it?"
        elif "running shoes" in user_query.lower() or "shoes for running" in user_query.lower():
            return f"If you're looking for running shoes, the 'SpeedRunner X' offers advanced cushioning ideal for marathons, while 'SprintFlow 5' is great for daily jogs. Which one sounds better for your needs?"
        elif "yoga" in user_query.lower():
            found_yoga_mat = next((p for p in retrieved_products if "yoga mat" in p['name'].lower()), None)
            if found_yoga_mat:
                return f"Yes, we have the '{found_yoga_mat['name']}'. It's a high-density mat offering superior grip and comfort. Perfect for your yoga sessions!"
            else:
                return f"I found products related to your query, such as the 'Yoga Mat Pro'. Can I provide more details or suggest other items?"
        elif "headphones" in user_query.lower() and "noise-cancelling" in user_query.lower():
            return f"Absolutely! Our 'Noise-Cancelling Headphones' offer premium sound quality and effective noise cancellation for an immersive audio experience. They are a top choice for music lovers and professionals."
        elif retrieved_products:
            product_names = ", ".join([p['name'] for p in retrieved_products])
            return f"Based on your search for '{user_query}', I found products like {product_names}. Can I tell you more about any of them, or help you find something more specific?"
        else:
            return "I'm sorry, I couldn't find exactly what you're looking for based on your query. Could you please rephrase it or provide more details?"


# --- Main Application Logic (API Simulation) ---
def main():
    print("Initializing E-commerce Recommender with LLM Enhancements...")
    
    # Initialize Embedding Model
    try:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("SentenceTransformer model loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load SentenceTransformer ('all-MiniLM-L6-v2'). Using a mock embedding model. Error: {e}")
        embedding_model = MockSentenceTransformer()

    # Initialize Services
    recommender_service = RecommenderService(product_catalog_data, user_interactions_data, embedding_model)
    llm_orchestration_service = LLMOrchestrationService(product_catalog_data, recommender_service, embedding_model)

    print("\n--- Intelligent E-commerce Recommender Demo ---")

    # Demo 1: Get Hybrid Recommendations and Explanation
    print("\n--- Demo 1: Hybrid Recommendations with Explanation (User U001, query 'powerful laptop') ---")
    user_id_1 = "U001"
    query_1 = "powerful laptop for work"
    recommendations_1 = recommender_service.get_hybrid_recommendations(user_id_1, user_query=query_1, top_n=2)
    print(f"Recommendations for User {user_id_1} with query '{query_1}':")
    if recommendations_1:
        for rec in recommendations_1:
            print(f"  - {rec['name']} (ID: {rec['id']})")
            explanation = llm_orchestration_service.generate_explanation(user_id_1, rec['id'])
            print(f"    Explanation: {explanation}")
    else:
        print("  No recommendations found.")

    print("\n--- Demo 1.1: Hybrid Recommendations (User U002, query 'comfortable shoes') ---")
    user_id_1_1 = "U002"
    query_1_1 = "comfortable shoes for daily activities"
    recommendations_1_1 = recommender_service.get_hybrid_recommendations(user_id_1_1, user_query=query_1_1, top_n=2)
    print(f"Recommendations for User {user_id_1_1} with query '{query_1_1}':")
    if recommendations_1_1:
        for rec in recommendations_1_1:
            print(f"  - {rec['name']} (ID: {rec['id']})")
            explanation = llm_orchestration_service.generate_explanation(user_id_1_1, rec['id'])
            print(f"    Explanation: {explanation}")
    else:
        print("  No recommendations found.")


    # Demo 2: Conversational Search
    print("\n--- Demo 2: Conversational Search ---")
    user_id_2 = "U002"

    conversational_query_1 = "I am looking for a new laptop, what do you suggest?"
    response_1 = llm_orchestration_service.conversational_search(conversational_query_1, user_id=user_id_2)
    print(f"User U002: {conversational_query_1}")
    print(f"Assistant: {response_1}\n")

    conversational_query_2 = "I need some comfortable running shoes."
    response_2 = llm_orchestration_service.conversational_search(conversational_query_2, user_id=user_id_2)
    print(f"User U002: {conversational_query_2}")
    print(f"Assistant: {response_2}\n")

    conversational_query_3 = "Do you have any products for yoga?"
    response_3 = llm_orchestration_service.conversational_search(conversational_query_3)
    print(f"User (anonymous): {conversational_query_3}")
    print(f"Assistant: {response_3}\n")

    conversational_query_4 = "I want a product that helps me focus while working."
    response_4 = llm_orchestration_service.conversational_search(conversational_query_4)
    print(f"User (anonymous): {conversational_query_4}")
    print(f"Assistant: {response_4}\n")

    conversational_query_5 = "What about stylish jeans?"
    response_5 = llm_orchestration_service.conversational_search(conversational_query_5)
    print(f"User (anonymous): {conversational_query_5}")
    print(f"Assistant: {response_5}\n")


if __name__ == "__main__":
    main()
