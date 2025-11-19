import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# --- 1. Data Layer (Simulated In-Memory Databases) ---

products_db = {
    "p1": {"name": "Laptop A", "category": "Electronics", "price": 1200, "features": [0.8, 0.7, 0.9, 0.5], "description": "Powerful laptop for everyday use.", "keywords": ["laptop", "electronics", "work"]},
    "p2": {"name": "Smartphone X", "category": "Electronics", "price": 800, "features": [0.7, 0.9, 0.6, 0.8], "description": "Latest smartphone with amazing camera.", "keywords": ["smartphone", "electronics", "camera"]},
    "p3": {"name": "Headphones B", "category": "Audio", "price": 150, "features": [0.3, 0.2, 0.8, 0.7], "description": "Noise-cancelling over-ear headphones.", "keywords": ["headphones", "audio", "noise-cancelling"]},
    "p4": {"name": "Smartwatch Y", "category": "Wearables", "price": 300, "features": [0.6, 0.5, 0.7, 0.9], "description": "Feature-rich smartwatch for fitness tracking.", "keywords": ["smartwatch", "wearables", "fitness"]},
    "p5": {"name": "Gaming PC Z", "category": "Electronics", "price": 2500, "features": [0.9, 0.8, 0.95, 0.6], "description": "High-performance gaming desktop.", "keywords": ["gaming pc", "electronics", "gaming"]},
    "p6": {"name": "Wireless Earbuds", "category": "Audio", "price": 100, "features": [0.4, 0.3, 0.7, 0.6], "description": "Compact wireless earbuds for on-the-go.", "keywords": ["earbuds", "audio", "wireless"]},
    "p7": {"name": "Mechanical Keyboard", "category": "Peripherals", "price": 120, "features": [0.5, 0.4, 0.6, 0.5], "description": "Durable mechanical keyboard for typing and gaming.", "keywords": ["keyboard", "peripherals", "gaming"]},
}

user_interactions_db = {
    "user1": {
        "purchases": ["p1", "p3"],
        "views": ["p1", "p2", "p3", "p4"],
        "ratings": {"p1": 5, "p3": 4}
    },
    "user2": {
        "purchases": ["p2", "p4"],
        "views": ["p1", "p2", "p4", "p5"],
        "ratings": {"p2": 4, "p4": 5}
    },
    "user3": {
        "purchases": ["p1", "p5"],
        "views": ["p1", "p3", "p5", "p7"],
        "ratings": {"p1": 5, "p5": 4}
    },
}

# --- 2. Core Recommender System (Backend Service) ---

class RecommenderEngine:
    def __init__(self, products, interactions):
        self.products = products
        self.interactions = interactions
        self.product_ids = list(products.keys())
        self.product_features_df = self._prepare_product_features()

    def _prepare_product_features(self):
        features_data = []
        for pid, data in self.products.items():
            features_data.append(data["features"])
        return pd.DataFrame(features_data, index=self.product_ids)

    def _get_purchased_product_features(self, user_id):
        purchased_pids = self.interactions.get(user_id, {}).get("purchases", [])
        if not purchased_pids:
            return None
        purchased_features = [self.products[pid]["features"] for pid in purchased_pids if pid in self.products]
        return np.mean(purchased_features, axis=0) if purchased_features else None

    def content_based_recommendation(self, user_id, num_recommendations=5):
        user_avg_features = self._get_purchased_product_features(user_id)
        if user_avg_features is None:
            return []

        all_product_features = self.product_features_df.values
        similarities = cosine_similarity([user_avg_features], all_product_features)[0]

        product_scores = pd.Series(similarities, index=self.product_ids)
        # Exclude already purchased items
        purchased_pids = self.interactions.get(user_id, {}).get("purchases", [])
        product_scores = product_scores[~product_scores.index.isin(purchased_pids)]

        top_products = product_scores.nlargest(num_recommendations).index.tolist()
        return top_products

    def collaborative_filtering_recommendation(self, user_id, num_recommendations=5):
        # Simplified collaborative filtering: find users with similar purchase history
        # and recommend what they bought but the current user hasn't.
        user_purchases = set(self.interactions.get(user_id, {}).get("purchases", []))
        if not user_purchases:
            return []

        user_similarities = {}
        for other_user_id, other_data in self.interactions.items():
            if other_user_id == user_id:
                continue
            other_purchases = set(other_data.get("purchases", []))
            common_purchases = len(user_purchases.intersection(other_purchases))
            union_purchases = len(user_purchases.union(other_purchases))
            if union_purchases > 0:
                user_similarities[other_user_id] = common_purchases / union_purchases
        
        sorted_similar_users = sorted(user_similarities.items(), key=lambda item: item[1], reverse=True)
        
        recommendations = defaultdict(float)
        for sim_user_id, similarity_score in sorted_similar_users:
            for pid in self.interactions[sim_user_id].get("purchases", []):
                if pid not in user_purchases:
                    recommendations[pid] += similarity_score
        
        sorted_recommendations = sorted(recommendations.items(), key=lambda item: item[1], reverse=True)
        return [pid for pid, _ in sorted_recommendations][:num_recommendations]


    def popularity_based_recommendation(self, num_recommendations=5):
        purchase_counts = defaultdict(int)
        for user_data in self.interactions.values():
            for pid in user_data.get("purchases", []):
                purchase_counts[pid] += 1
        
        sorted_popular = sorted(purchase_counts.items(), key=lambda item: item[1], reverse=True)
        return [pid for pid, _ in sorted_popular][:num_recommendations]

    def rank_candidates(self, user_id, content_cands, collab_cands, pop_cands, num_recommendations=5):
        candidate_scores = defaultdict(float)
        
        # Weighted scoring (example weights)
        for pid in content_cands:
            candidate_scores[pid] += 0.5
        for pid in collab_cands:
            candidate_scores[pid] += 0.3
        for pid in pop_cands:
            candidate_scores[pid] += 0.2
            
        # Further refine based on user views (implicit feedback)
        user_views = set(self.interactions.get(user_id, {}).get("views", []))
        for pid in candidate_scores.keys():
            if pid in user_views:
                candidate_scores[pid] += 0.1 # Boost items user has viewed
                
        # Remove items already purchased by the user
        purchased_pids = set(self.interactions.get(user_id, {}).get("purchases", []))
        ranked_candidates = {pid: score for pid, score in candidate_scores.items() if pid not in purchased_pids}

        sorted_ranked = sorted(ranked_candidates.items(), key=lambda item: item[1], reverse=True)
        return [pid for pid, _ in sorted_ranked][:num_recommendations]


# --- 3. LLM Integration Service (Backend Service) ---

class LLMService:
    def __init__(self, mock_mode=True):
        self.mock_mode = mock_mode
        # In a real application, initialize OpenAI client or Hugging Face pipeline here

    def _mock_llm_api_call(self, prompt):
        # Simulate LLM response based on keywords in prompt
        if "explain why" in prompt.lower() or "recommendation for" in prompt.lower():
            if "similar to Laptop A" in prompt:
                return "This product is recommended because its features closely align with your previous purchase, Laptop A, offering comparable performance and value in the electronics category."
            elif "users who bought Smartphone X also liked" in prompt:
                return "Many users who enjoyed Smartphone X also purchased this item, suggesting a strong preference match based on popular trends among similar customers."
            elif "popular choice" in prompt:
                return "This item is a popular choice among many customers, indicating high satisfaction and broad appeal within its category."
            else:
                return "Based on our advanced AI analysis, this recommendation is tailored to your unique preferences and browsing history."
        elif "dynamic description" in prompt.lower() or "create a compelling description" in prompt.lower():
            if "Laptop A" in prompt:
                return "Unleash your productivity with the cutting-edge Laptop A, engineered for seamless multitasking and immersive entertainment. Its sleek design and powerful processor make it the ultimate companion for work and play."
            elif "Smartphone X" in prompt:
                return "Capture life's moments in stunning detail with Smartphone X, featuring an revolutionary camera system and an intuitive user experience. Experience the future of mobile technology in your palm."
            else:
                return f"Discover the {prompt.split('product: ')[-1].split('description')[0].strip()}, a revolutionary product designed to elevate your everyday experience. Enjoy unparalleled features and a sleek design that stands out."
        return "Mock LLM response: Information requested."

    def generate_explanation(self, product_name, recommendation_reason):
        prompt = f"Explain why '{product_name}' is a good recommendation for a user, considering the reason: {recommendation_reason}. Provide a natural and personalized explanation."
        if self.mock_mode:
            return self._mock_llm_api_call(prompt)
        else:
            # Real LLM API call would go here
            pass # Placeholder for actual LLM integration

    def generate_dynamic_description(self, product_details):
        product_name = product_details.get("name", "unknown product")
        product_category = product_details.get("category", "")
        product_keywords = ", ".join(product_details.get("keywords", []))
        prompt = f"Create a compelling, detailed, and personalized product description for: Product Name: {product_name}, Category: {product_category}, Keywords: {product_keywords}. Focus on engaging the user and highlighting key benefits. This is for an e-commerce website product page."
        if self.mock_mode:
            return self._mock_llm_api_call(prompt)
        else:
            # Real LLM API call would go here
            pass # Placeholder for actual LLM integration


# --- Main Application Flow ---

def get_ai_powered_recommendations(user_id, num_recs=3):
    recommender = RecommenderEngine(products_db, user_interactions_db)
    llm_service = LLMService(mock_mode=True)

    print(f"\n--- Generating Recommendations for User: {user_id} ---")
    
    # 1. Candidate Generation
    content_candidates = recommender.content_based_recommendation(user_id, num_recommendations=num_recs*2)
    collab_candidates = recommender.collaborative_filtering_recommendation(user_id, num_recommendations=num_recs*2)
    popular_candidates = recommender.popularity_based_recommendation(num_recommendations=num_recs*2)

    all_candidates = list(set(content_candidates + collab_candidates + popular_candidates))
    
    print(f"Candidate Pool ({len(all_candidates)} items): {all_candidates}")

    # 2. Ranking
    ranked_product_ids = recommender.rank_candidates(user_id, content_candidates, collab_candidates, popular_candidates, num_recommendations=num_recs)

    if not ranked_product_ids:
        print("No specific recommendations found. Here are some popular items instead:")
        ranked_product_ids = recommender.popularity_based_recommendation(num_recommendations=num_recs)

    recommendations = []
    for pid in ranked_product_ids:
        product = products_db.get(pid)
        if product:
            # Determine a simple reason for explanation generation
            reason = ""
            if pid in content_candidates and pid in user_interactions_db.get(user_id, {}).get("purchases", []):
                 reason = f"similar to your previous purchases like {user_interactions_db.get(user_id, {}).get("purchases", [])[0] if user_interactions_db.get(user_id, {}).get("purchases", []) else 'other items'}"
            elif pid in content_candidates:
                reason = f"similar to your past interests in the '{product['category']}' category"
            elif pid in collab_candidates:
                reason = "users who share similar tastes also purchased this item"
            elif pid in popular_candidates:
                reason = "a popular choice among many customers"
            else:
                reason = "tailored to your potential interests"

            explanation = llm_service.generate_explanation(product["name"], reason)
            dynamic_description = llm_service.generate_dynamic_description(product)
            
            recommendations.append({
                "product_id": pid,
                "name": product["name"],
                "price": product["price"],
                "category": product["category"],
                "description": dynamic_description,
                "llm_explanation": explanation
            })
    return recommendations


# --- Example Usage ---
if __name__ == "__main__":
    # Get recommendations for User 1
    user1_recs = get_ai_powered_recommendations("user1", num_recs=2)
    print("\n--- Recommendations for User1 ---")
    for rec in user1_recs:
        print(f"Product: {rec['name']} ({rec['product_id']})")
        print(f"  Category: {rec['category']}, Price: ${rec['price']}")
        print(f"  Description: {rec['description']}")
        print(f"  Why this? {rec['llm_explanation']}\n")

    # Get recommendations for User 2
    user2_recs = get_ai_powered_recommendations("user2", num_recs=2)
    print("\n--- Recommendations for User2 ---")
    for rec in user2_recs:
        print(f"Product: {rec['name']} ({rec['product_id']})")
        print(f"  Category: {rec['category']}, Price: ${rec['price']}")
        print(f"  Description: {rec['description']}")
        print(f"  Why this? {rec['llm_explanation']}\n")

    # Get recommendations for a new user (or user with limited history)
    # Simulating a new user with no interactions for simplicity
    # For this, popularity-based will mostly kick in.
    user_interactions_db["user4"] = {"purchases": [], "views": [], "ratings": {}}
    user4_recs = get_ai_powered_recommendations("user4", num_recs=2)
    print("\n--- Recommendations for User4 (New User) ---")
    for rec in user4_recs:
        print(f"Product: {rec['name']} ({rec['product_id']})")
        print(f"  Category: {rec['category']}, Price: ${rec['price']}")
        print(f"  Description: {rec['description']}")
        print(f"  Why this? {rec['llm_explanation']}\n")
