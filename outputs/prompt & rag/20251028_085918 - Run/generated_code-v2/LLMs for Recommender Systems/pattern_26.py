import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class FashionRecommender:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.item_df = None
        self.user_interactions_df = None
        self.item_embeddings = None
        self.user_profiles = {}

    def load_data(self, item_data_path, user_interactions_data_path):
        self.item_df = pd.read_csv(item_data_path)
        self.user_interactions_df = pd.read_csv(user_interactions_data_path)
        self.item_df['item_id'] = self.item_df['item_id'].astype(str)
        self.user_interactions_df['user_id'] = self.user_interactions_df['user_id'].astype(str)
        self.user_interactions_df['item_id'] = self.user_interactions_df['item_id'].astype(str)

    def generate_item_embeddings(self):
        if self.item_df is None:
            raise ValueError("Item data not loaded. Please call load_data() first.")
        item_descriptions = self.item_df['description'].tolist()
        print(f"Generating embeddings for {len(item_descriptions)} items...")
        self.item_embeddings = self.model.encode(item_descriptions, show_progress_bar=True)
        print("Item embeddings generated.")

    def create_user_profiles(self):
        if self.user_interactions_df is None or self.item_embeddings is None:
            raise ValueError("User interactions or item embeddings not available. Load data and generate embeddings first.")

        item_id_to_index = {item_id: idx for idx, item_id in enumerate(self.item_df['item_id'])}

        for user_id in self.user_interactions_df['user_id'].unique():
            user_items = self.user_interactions_df[self.user_interactions_df['user_id'] == user_id]['item_id'].tolist()
            
            # Filter out items that might not be in our item_df (e.g., cold start for items)
            known_item_indices = [item_id_to_index[item_id] for item_id in user_items if item_id in item_id_to_index]
            
            if known_item_indices:
                user_interaction_embeddings = self.item_embeddings[known_item_indices]
                self.user_profiles[user_id] = np.mean(user_interaction_embeddings, axis=0)
            else:
                self.user_profiles[user_id] = None # User has no interactions with known items
        print("User profiles created.")

    def recommend_items(self, user_id, top_n=5):
        if user_id not in self.user_profiles or self.user_profiles[user_id] is None:
            print(f"User {user_id} not found or has no valid profile. Cannot provide recommendations.")
            return []

        user_profile_embedding = self.user_profiles[user_id].reshape(1, -1)

        # Exclude items the user has already interacted with from recommendations
        interacted_item_ids = set(self.user_interactions_df[self.user_interactions_df['user_id'] == user_id]['item_id'].tolist())
        available_item_ids = self.item_df[~self.item_df['item_id'].isin(interacted_item_ids)]['item_id'].tolist()
        available_item_indices = [idx for idx, item_id in enumerate(self.item_df['item_id']) if item_id in available_item_ids]

        if not available_item_indices:
            print(f"No new items to recommend for user {user_id}.")
            return []
            
        available_item_embeddings = self.item_embeddings[available_item_indices]
        similarities = cosine_similarity(user_profile_embedding, available_item_embeddings)[0]

        top_n_indices = np.argsort(similarities)[::-1][:top_n]
        recommended_item_global_indices = [available_item_indices[i] for i in top_n_indices]
        
        recommended_items = self.item_df.iloc[recommended_item_global_indices]
        return recommended_items[['item_id', 'name', 'description']].to_dict(orient='records')

if __name__ == '__main__':
    # Create dummy data for demonstration
    item_data = {
        'item_id': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
        'name': [
            'Blue Denim Jeans',
            'Striped Cotton T-shirt',
            'Leather Biker Jacket',
            'Floral Maxi Dress',
            'Classic White Sneakers',
            'Wool Blend Scarf',
            'Slim Fit Chinos',
            'Graphic Print Hoodie',
            'High Waist Skirt',
            'Casual Canvas Backpack'
        ],
        'description': [
            'Comfortable blue denim jeans for everyday wear. Straight fit.',
            'Soft cotton t-shirt with horizontal stripes. Perfect for summer.',
            'Stylish black leather jacket. Ideal for a cool, edgy look.',
            'Elegant floral print long dress. Suitable for various occasions.',
            'Timeless white sneakers, comfortable and versatile for any outfit.',
            'Warm grey wool blend scarf, perfect for winter accessories.',
            'Modern slim fit chinos in beige. Smart casual wear.',
            'Trendy hoodie with a unique graphic design. Relaxed fit.',
            'Fashionable high waist skirt in a solid color. Versatile.',
            'Durable canvas backpack for daily essentials. Casual style.'
        ]
    }
    user_interactions_data = {
        'user_id': ['U1', 'U1', 'U1', 'U2', 'U2', 'U3', 'U3', 'U3', 'U3'],
        'item_id': ['1', '2', '5', '3', '6', '1', '4', '7', '10'] # U1 liked Jeans, T-shirt, Sneakers; U2 liked Jacket, Scarf; U3 liked Jeans, Dress, Chinos, Backpack
    }

    item_df = pd.DataFrame(item_data)
    user_interactions_df = pd.DataFrame(user_interactions_data)

    # Save dummy data to CSV files
    item_df.to_csv('fashion_items.csv', index=False)
    user_interactions_df.to_csv('user_interactions.csv', index=False)

    recommender = FashionRecommender()
    recommender.load_data('fashion_items.csv', 'user_interactions.csv')
    recommender.generate_item_embeddings()
    recommender.create_user_profiles()

    # Get recommendations for a specific user
    user_id_to_recommend = 'U1'
    recommendations = recommender.recommend_items(user_id_to_recommend, top_n=3)
    print(f"\nRecommendations for user {user_id_to_recommend}:")
    if recommendations:
        for rec in recommendations:
            print(f"- {rec['name']} (ID: {rec['item_id']})")
    else:
        print("No recommendations found.")

    user_id_to_recommend = 'U2'
    recommendations = recommender.recommend_items(user_id_to_recommend, top_n=3)
    print(f"\nRecommendations for user {user_id_to_recommend}:")
    if recommendations:
        for rec in recommendations:
            print(f"- {rec['name']} (ID: {rec['item_id']})")
    else:
        print("No recommendations found.")

    user_id_to_recommend = 'U4' # A new user with no interactions
    recommendations = recommender.recommend_items(user_id_to_recommend, top_n=3)
    print(f"\nRecommendations for user {user_id_to_recommend}:")
    if recommendations:
        for rec in recommendations:
            print(f"- {rec['name']} (ID: {rec['item_id']})")
    else:
        print("No recommendations found for this user.")
