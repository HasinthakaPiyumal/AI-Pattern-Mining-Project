import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

class FashionRecommender:
    def __init__(self):
        self.items_df = self._load_dummy_items()
        self.user_preferences = self._load_dummy_user_preferences()
        self.tfidf_vectorizer = TfidfVectorizer()
        self.item_feature_matrix = None

    def _load_dummy_items(self):
        data = {
            "item_id": ["item_001", "item_002", "item_003", "item_004", "item_005", "item_006", "item_007", "item_008", "item_009", "item_010"],
            "name": [
                "Blue Denim Jacket", "Floral Sundress", "Leather Boots", "Striped T-Shirt", "Cargo Pants",
                "Knitted Sweater", "Running Shoes", "Casual Blazer", "Evening Gown", "Classic White Shirt"
            ],
            "category": [
                "Jacket", "Dress", "Footwear", "Top", "Bottom",
                "Top", "Footwear", "Jacket", "Dress", "Top"
            ],
            "style": [
                "Casual, Streetwear", "Bohemian, Summer", "Edgy, Classic", "Casual, Minimalist", "Utility, Casual",
                "Cozy, Winter", "Sporty, Athleisure", "Smart Casual, Business", "Elegant, Formal", "Timeless, Versatile"
            ],
            "material": [
                "Denim", "Cotton, Viscose", "Leather", "Cotton", "Cotton, Polyester",
                "Wool, Acrylic", "Mesh, Rubber", "Linen, Cotton", "Silk, Polyester", "Cotton"
            ],
            "description": [
                "A classic blue denim jacket, perfect for layering on a cool evening.",
                "Light and airy floral sundress, ideal for beach days or summer outings.",
                "Durable and stylish leather boots, suitable for both casual and semi-formal wear.",
                "Simple striped t-shirt, a versatile staple for any wardrobe.",
                "Comfortable cargo pants with multiple pockets, great for outdoor activities.",
                "Warm knitted sweater, perfect for cold weather comfort and style.",
                "Lightweight running shoes designed for optimal performance and comfort.",
                "A versatile casual blazer, easily dressed up or down for various occasions.",
                "Elegant evening gown, perfect for formal events and special occasions.",
                "A timeless classic white shirt, essential for a smart and polished look."
            ]
        }
        return pd.DataFrame(data)

    def _load_dummy_user_preferences(self):
        # Simulate user preferences for styles, materials, and categories
        # In a real system, this would come from user history, explicit inputs, etc.
        return {
            "user_1": {"style": ["Casual", "Streetwear"], "material": ["Denim", "Cotton"], "category": ["Jacket", "Top"]},
            "user_2": {"style": ["Bohemian", "Summer"], "material": ["Cotton", "Viscose"], "category": ["Dress"]},
            "user_3": {"style": ["Elegant", "Formal"], "material": ["Silk", "Polyester"], "category": ["Dress", "Jacket"]},
            "user_4": {"style": ["Sporty", "Athleisure"], "material": ["Mesh", "Rubber"], "category": ["Footwear"]}
        }

    def _get_item_features(self):
        # Combine relevant text features for TF-IDF
        self.items_df["features"] = self.items_df["category"] + " " + \
                                     self.items_df["style"] + " " + \
                                     self.items_df["material"] + " " + \
                                     self.items_df["description"]
        self.item_feature_matrix = self.tfidf_vectorizer.fit_transform(self.items_df["features"])

    def _get_user_profile_vector(self, user_id):
        user_prefs = self.user_preferences.get(user_id, {})
        combined_prefs = []
        for key in ["style", "material", "category"]:
            combined_prefs.extend(user_prefs.get(key, []))
        
        if not combined_prefs:
            # Default to general if no preferences
            return self.tfidf_vectorizer.transform(["casual comfortable versatile"]).toarray()

        # Ensure the vectorizer has been fitted
        if self.item_feature_matrix is None:
            self._get_item_features()

        return self.tfidf_vectorizer.transform([" ".join(combined_prefs)]).toarray()

    def get_recommendations(self, user_id, num_items=5):
        if user_id not in self.user_preferences:
            print(f"Warning: User {user_id} not found. Providing general recommendations.")
            # Fallback to general popular items or diverse set
            return self.items_df.sample(num_items).to_dict(orient="records")

        if self.item_feature_matrix is None:
            self._get_item_features()

        user_vector = self._get_user_profile_vector(user_id)
        
        # Calculate cosine similarity between user profile and all items
        similarities = cosine_similarity(user_vector, self.item_feature_matrix).flatten()
        
        # Get top N item indices
        top_item_indices = similarities.argsort()[-num_items:][::-1]
        
        recommended_items = self.items_df.iloc[top_item_indices].copy()
        recommended_items["similarity_score"] = similarities[top_item_indices]
        
        # Add a simple reasoning for explanation
        user_prefs = self.user_preferences.get(user_id, {})
        reasoning_parts = []
        if user_prefs.get("style"):
            reasoning_parts.append(f"matches your preferred styles like {', '.join(user_prefs['style'])}")
        if user_prefs.get("material"):
            reasoning_parts.append(f"features materials such as {', '.join(user_prefs['material'])}")
        if user_prefs.get("category"):
            reasoning_parts.append(f"falls into categories you enjoy, such as {', '.join(user_prefs['category'])}")
        
        general_reasoning = "This recommendation aligns with your preferences. " + ". ".join(reasoning_parts) + "."
        recommended_items["recommendation_reasoning"] = general_reasoning

        return recommended_items.to_dict(orient="records")

if __name__ == "__main__":
    recommender = FashionRecommender()
    user_id = "user_1"
    recommendations = recommender.get_recommendations(user_id, num_items=3)
    print(f"Recommendations for {user_id}:")
    for rec in recommendations:
        print(f"  - {rec['name']} ({rec['item_id']}) - Score: {rec['similarity_score']:.2f}")
        print(f"    Reasoning: {rec['recommendation_reasoning']}")

    print("\n--- Testing with unknown user ---")
    user_id_unknown = "user_x"
    recommendations_unknown = recommender.get_recommendations(user_id_unknown, num_items=2)
    print(f"Recommendations for {user_id_unknown}:")
    for rec in recommendations_unknown:
        print(f"  - {rec['name']} ({rec['item_id']})")