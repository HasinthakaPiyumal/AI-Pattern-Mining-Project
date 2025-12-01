import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ProductRecommender:
    def __init__(self, users_df, products_df, interactions_df):
        self.users_df = users_df
        self.products_df = products_df
        self.interactions_df = interactions_df
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.product_features_matrix = self._preprocess_products()

    def _preprocess_products(self):
        self.products_df['combined_features'] = self.products_df['category'] + " " + self.products_df['description']
        return self.tfidf_vectorizer.fit_transform(self.products_df['combined_features'])

    def _get_user_history(self, user_id):
        user_interactions = self.interactions_df[self.interactions_df['user_id'] == user_id]
        purchased_product_ids = user_interactions[user_interactions['event_type'] == 'purchase']['product_id'].tolist()
        viewed_product_ids = user_interactions[user_interactions['event_type'] == 'view']['product_id'].tolist()
        return list(set(purchased_product_ids + viewed_product_ids))

    def _get_content_based_recommendations(self, user_id, num_recommendations=5):
        user_history_product_ids = self._get_user_history(user_id)
        if not user_history_product_ids:
            return [], []

        history_products_features = self.product_features_matrix[self.products_df[self.products_df['product_id'].isin(user_history_product_ids)].index]
        
        # Calculate average feature vector for user's history
        user_profile_vector = history_products_features.mean(axis=0)

        # Calculate similarity with all products
        similarities = cosine_similarity(user_profile_vector, self.product_features_matrix).flatten()
        
        # Exclude products already in user's history
        candidate_indices = [i for i, product_id in enumerate(self.products_df['product_id']) if product_id not in user_history_product_ids]
        candidate_similarities = similarities[candidate_indices]
        candidate_product_ids = self.products_df.iloc[candidate_indices]['product_id'].tolist()

        top_indices = np.argsort(candidate_similarities)[::-1][:num_recommendations]
        recommended_product_ids = [candidate_product_ids[i] for i in top_indices]
        reasons = []
        for prod_id in recommended_product_ids:
            prod_name = self.products_df[self.products_df['product_id'] == prod_id]['name'].iloc[0]
            reasons.append(f"Based on your past interactions, you might like '{prod_name}'. It shares similarities with items you've previously shown interest in.")
        return recommended_product_ids, reasons

    def _get_collaborative_recommendations(self, user_id, num_recommendations=5):
        # Simplified collaborative filtering: find users with similar preferences and recommend what they bought
        target_user_prefs = self.users_df[self.users_df['user_id'] == user_id]['preferences'].iloc[0]
        
        similar_users = self.users_df[self.users_df['preferences'].apply(lambda x: any(p in target_user_prefs for p in x)) & (self.users_df['user_id'] != user_id)]
        
        if similar_users.empty:
            return [], []

        similar_user_purchases = self.interactions_df[self.interactions_df['user_id'].isin(similar_users['user_id']) & (self.interactions_df['event_type'] == 'purchase')]
        
        user_history = self._get_user_history(user_id)
        
        # Recommend products frequently bought by similar users, not yet interacted with by target user
        recommended_product_ids = similar_user_purchases['product_id'].value_counts().index.tolist()
        recommended_product_ids = [pid for pid in recommended_product_ids if pid not in user_history][:num_recommendations]

        reasons = []
        for prod_id in recommended_product_ids:
            prod_name = self.products_df[self.products_df['product_id'] == prod_id]['name'].iloc[0]
            reasons.append(f"Users similar to you have also purchased '{prod_name}'.")
        return recommended_product_ids, reasons

    def recommend(self, user_id, num_recommendations=10):
        content_recs, content_reasons = self._get_content_based_recommendations(user_id, num_recommendations // 2)
        collab_recs, collab_reasons = self._get_collaborative_recommendations(user_id, num_recommendations // 2)

        # Combine and deduplicate recommendations
        combined_recs = []
        combined_reasons_map = {}
        
        for i, rec_id in enumerate(content_recs):
            if rec_id not in combined_recs:
                combined_recs.append(rec_id)
                combined_reasons_map[rec_id] = [content_reasons[i]]
        
        for i, rec_id in enumerate(collab_recs):
            if rec_id not in combined_recs:
                combined_recs.append(rec_id)
                combined_reasons_map[rec_id] = [collab_reasons[i]]
            else:
                combined_reasons_map[rec_id].append(collab_reasons[i]) # Add additional reasons

        final_recommendations = combined_recs[:num_recommendations]
        final_reasons = {rec_id: combined_reasons_map.get(rec_id, []) for rec_id in final_recommendations}
        
        return final_recommendations, final_reasons

class LLMExplainer:
    def __init__(self):
        pass # In a real system, this would initialize an LLM client

    def generate_explanation(self, product_name, extracted_reasons, user_preferences):
        base_explanation = f"We recommend '{product_name}' to you. "
        
        reason_strings = []
        if extracted_reasons:
            for reason in extracted_reasons:
                reason_strings.append(reason)
        
        if reason_strings:
            return base_explanation + "Here's why: " + " ".join(reason_strings)
        else:
            return base_explanation + "This is a popular item that might interest you."


# --- Simulated Data Layer ---
users_data = {
    'user_id': [1, 2, 3, 4, 5],
    'age': [30, 24, 45, 29, 35],
    'gender': ['M', 'F', 'M', 'F', 'M'],
    'preferences': [['sports', 'tech'], ['fashion', 'beauty'], ['books', 'tech'], ['fashion', 'home'], ['sports', 'books']]
}
users_df = pd.DataFrame(users_data)

products_data = {
    'product_id': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    'name': ['Running Shoes', 'Smartwatch', 'Fiction Novel', 'Summer Dress', 'Coffee Maker', 'Yoga Mat', 'Bluetooth Headphones', 'Cookbook', 'Desk Chair', 'Fitness Tracker'],
    'category': ['sports', 'tech', 'books', 'fashion', 'home', 'sports', 'tech', 'books', 'home', 'sports'],
    'description': [
        'High-performance running shoes with advanced cushioning.',
        'Feature-rich smartwatch with health monitoring.',
        'Bestselling contemporary fiction novel.',
        'Lightweight and stylish summer dress.',
        'Programmable coffee maker with grinder.',
        'Eco-friendly yoga mat for all levels.',
        'Noise-cancelling Bluetooth headphones.',
        'Delicious recipes for healthy eating.',
        'Ergonomic desk chair for comfort.',
        'Monitor your activity and sleep with this tracker.'
    ]
}
products_df = pd.DataFrame(products_data)

interactions_data = {
    'user_id': [1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 1, 2, 3, 4, 5, 1, 2],
    'product_id': [101, 102, 106, 104, 107, 103, 108, 104, 105, 101, 103, 109, 110, 105, 106, 107, 102, 103],
    'rating': [5, 4, 5, 4, 3, 5, 4, 5, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4],
    'event_type': ['purchase', 'view', 'purchase', 'purchase', 'view', 'purchase', 'view', 'purchase', 'view', 'purchase', 'view', 'view', 'purchase', 'view', 'purchase', 'view', 'view', 'purchase']
}
interactions_df = pd.DataFrame(interactions_data)

# --- Main Application Logic ---
recommender = ProductRecommender(users_df, products_df, interactions_df)
explainer = LLMExplainer()

print("\n--- Recommendations for User 1 ---")
user_id_to_test = 1
recommendations, reasons_map = recommender.recommend(user_id_to_test)

if recommendations:
    print(f"Recommended products for User {user_id_to_test}:")
    for prod_id in recommendations:
        product_name = products_df[products_df['product_id'] == prod_id]['name'].iloc[0]
        user_prefs = users_df[users_df['user_id'] == user_id_to_test]['preferences'].iloc[0]
        explanation = explainer.generate_explanation(product_name, reasons_map.get(prod_id, []), user_prefs)
        print(f"  - {product_name}: {explanation}")
else:
    print("No recommendations found for this user.")

print("\n--- Recommendations for User 2 ---")
user_id_to_test = 2
recommendations, reasons_map = recommender.recommend(user_id_to_test)

if recommendations:
    print(f"Recommended products for User {user_id_to_test}:")
    for prod_id in recommendations:
        product_name = products_df[products_df['product_id'] == prod_id]['name'].iloc[0]
        user_prefs = users_df[users_df['user_id'] == user_id_to_test]['preferences'].iloc[0]
        explanation = explainer.generate_explanation(product_name, reasons_map.get(prod_id, []), user_prefs)
        print(f"  - {product_name}: {explanation}")
else:
    print("No recommendations found for this user.")

print("\n--- Recommendations for User 5 ---")
user_id_to_test = 5
recommendations, reasons_map = recommender.recommend(user_id_to_test)

if recommendations:
    print(f"Recommended products for User {user_id_to_test}:")
    for prod_id in recommendations:
        product_name = products_df[products_df['product_id'] == prod_id]['name'].iloc[0]
        user_prefs = users_df[users_df['user_id'] == user_id_to_test]['preferences'].iloc[0]
        explanation = explainer.generate_explanation(product_name, reasons_map.get(prod_id, []), user_prefs)
        print(f"  - {product_name}: {explanation}")
else:
    print("No recommendations found for this user.")