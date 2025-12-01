import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class ProductRecommendationEngine:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.products_df = None
        self.product_embeddings = None
        self.user_interactions_df = None

    def ingest_data(self, products_data_path, user_interactions_data_path):
        self.products_df = pd.read_csv(products_data_path)
        self.user_interactions_df = pd.read_csv(user_interactions_data_path)
        self.products_df['product_id'] = self.products_df['product_id'].astype(str)
        self.user_interactions_df['product_id'] = self.user_interactions_df['product_id'].astype(str)

    def preprocess_text(self, text):
        return str(text).lower() # Simple preprocessing

    def generate_product_embeddings(self):
        if self.products_df is None:
            raise ValueError("Product data not loaded. Call ingest_data first.")

        product_texts = self.products_df['description'].apply(self.preprocess_text).tolist()
        self.product_embeddings = self.model.encode(product_texts, show_progress_bar=True)
        print("Product embeddings generated.")

    def get_cold_start_recommendations(self, num_recommendations=5):
        if self.product_embeddings is None:
            raise ValueError("Product embeddings not generated. Call generate_product_embeddings first.")

        # For cold-start, recommend highly-rated or popular items (simplified)
        # Here we'll just return the first few products as a placeholder for 'popular'
        # In a real system, you'd use more sophisticated popularity metrics
        cold_start_products = self.products_df.head(num_recommendations)['product_id'].tolist()
        return cold_start_products

    def get_personalized_recommendations(self, user_id, num_recommendations=5):
        if self.products_df is None or self.user_interactions_df is None or self.product_embeddings is None:
            raise ValueError("Data or embeddings not loaded. Call ingest_data and generate_product_embeddings first.")

        user_interactions = self.user_interactions_df[self.user_interactions_df['user_id'] == user_id]
        if user_interactions.empty:
            print(f"No interactions found for user {user_id}. Providing cold-start recommendations.")
            return self.get_cold_start_recommendations(num_recommendations)

        # Get embeddings of products the user has interacted with (e.g., purchased, viewed positively)
        interacted_product_ids = user_interactions['product_id'].tolist()
        interacted_indices = self.products_df[self.products_df['product_id'].isin(interacted_product_ids)].index.tolist()

        if not interacted_indices:
            print(f"No embeddings found for interacted products for user {user_id}. Providing cold-start recommendations.")
            return self.get_cold_start_recommendations(num_recommendations)

        interacted_embeddings = self.product_embeddings[interacted_indices]

        # Create a 'user profile' embedding by averaging interacted product embeddings
        user_embedding = np.mean(interacted_embeddings, axis=0).reshape(1, -1)

        # Calculate similarity between user embedding and all product embeddings
        similarities = cosine_similarity(user_embedding, self.product_embeddings)[0]

        # Exclude products the user has already interacted with
        all_product_ids = self.products_df['product_id'].tolist()
        candidate_product_indices = [i for i, pid in enumerate(all_product_ids) if pid not in interacted_product_ids]
        candidate_similarities = similarities[candidate_product_indices]
        candidate_product_ids = [all_product_ids[i] for i in candidate_product_indices]

        # Get top N recommendations
        top_indices = np.argsort(candidate_similarities)[::-1][:num_recommendations]
        recommended_product_ids = [candidate_product_ids[i] for i in top_indices]

        return recommended_product_ids


if __name__ == "__main__":
    # Create dummy data files for demonstration
    products_data = {
        'product_id': ["1", "2", "3", "4", "5", "6"],
        'description': [
            "High-quality noise-cancelling headphones for immersive audio experience.",
            "Ergonomic wireless mouse with customizable buttons for productivity.",
            "Smartwatch with health tracking features and long battery life.",
            "Portable Bluetooth speaker with rich bass and waterproof design.",
            "4K Ultra HD LED Smart TV with vibrant colors and smart features.",
            "Gaming mechanical keyboard with RGB backlighting and tactile switches."
        ],
        'category': ["Electronics", "Electronics", "Wearable Tech", "Audio", "Electronics", "Gaming"]
    }
    products_df_dummy = pd.DataFrame(products_data)
    products_df_dummy.to_csv("products.csv", index=False)

    user_interactions_data = {
        'user_id': ["user_A", "user_A", "user_B", "user_B", "user_C"],
        'product_id': ["1", "3", "2", "4", "5"],
        'interaction_type': ["purchase", "view", "purchase", "view", "purchase"]
    }
    user_interactions_df_dummy = pd.DataFrame(user_interactions_data)
    user_interactions_df_dummy.to_csv("user_interactions.csv", index=False)

    print("Dummy data files created: products.csv, user_interactions.csv")

    engine = ProductRecommendationEngine()
    engine.ingest_data("products.csv", "user_interactions.csv")
    engine.generate_product_embeddings()

    print("\n--- Cold-start Recommendations ---")
    cold_start_recs = engine.get_cold_start_recommendations(num_recommendations=3)
    print(f"Recommended product IDs: {cold_start_recs}")

    print("\n--- Personalized Recommendations for User A ---")
    user_a_recs = engine.get_personalized_recommendations(user_id="user_A", num_recommendations=3)
    print(f"Recommended product IDs for user A: {user_a_recs}")

    print("\n--- Personalized Recommendations for User B ---")
    user_b_recs = engine.get_personalized_recommendations(user_id="user_B", num_recommendations=3)
    print(f"Recommended product IDs for user B: {user_b_recs}")

    print("\n--- Personalized Recommendations for User C (with fewer interactions) ---")
    user_c_recs = engine.get_personalized_recommendations(user_id="user_C", num_recommendations=3)
    print(f"Recommended product IDs for user C: {user_c_recs}")

    print("\n--- Personalized Recommendations for New User (no interactions) ---")
    new_user_recs = engine.get_personalized_recommendations(user_id="new_user_D", num_recommendations=3)
    print(f"Recommended product IDs for new user D: {new_user_recs}")
