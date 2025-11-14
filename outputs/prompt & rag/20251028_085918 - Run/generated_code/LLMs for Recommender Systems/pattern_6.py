"""
This module implements an E-commerce Product Recommender System enhanced by simulated Large Language Model (LLM) capabilities.
The system provides product recommendations, generates human-centric explanations for these recommendations,
and allows users to refine their preferences through a natural language interface.
"""

import random

class ProductRecommender:
    """
    An E-commerce Product Recommender System with simulated LLM enhancements.
    """

    def __init__(self, products_data, users_data):
        """
        Initializes the recommender with product catalog and user profiles.

        Args:
            products_data (dict): A dictionary where keys are product IDs and values are product details.
            users_data (dict): A dictionary where keys are user IDs and values are user profiles.
        """
        self.products = products_data
        self.users = users_data

    def _get_product_features(self, product_id):
        """
        Extracts relevant features for content-based filtering.
        """
        product = self.products.get(product_id)
        if not product:
            return []
        return product["category"].lower().split() + product["description"].lower().split()

    def _content_based_recommendations(self, user_id, num_recommendations=5):
        """
        Generates initial content-based recommendations for a user.
        This is a simplified implementation based on common categories/keywords in user history.
        """
        user_history = self.users.get(user_id, {}).get("purchase_history", [])
        if not user_history:
            return random.sample(list(self.products.keys()), min(num_recommendations, len(self.products)))

        # Collect features from user's purchased products
        user_features = set()
        for item_id in user_history:
            user_features.update(self._get_product_features(item_id))

        scores = {}
        for prod_id, prod_details in self.products.items():
            if prod_id in user_history:
                continue # Don't recommend already purchased items
            prod_features = set(self._get_product_features(prod_id))
            # Simple overlap score
            score = len(user_features.intersection(prod_features))
            if score > 0:
                scores[prod_id] = score

        sorted_recommendations = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [prod_id for prod_id, _ in sorted_recommendations[:num_recommendations]]

    def _simulate_llm_enhance_recommendations(self, initial_recommendations, user_id, context):
        """
        Simulates an LLM enhancing initial recommendations.
        This can involve re-ranking, filtering, or adding new items based on 'semantic understanding'.
        For this simulation, it might filter out items based on a 'dislike' in context or promote 'trending' items.
        """
        user_prefs = self.users.get(user_id, {}).get("preferences", {})
        enhanced_recs = []

        # Simulate LLM filtering based on user dislikes
        disliked_categories = user_prefs.get("disliked_categories", [])
        for prod_id in initial_recommendations:
            product = self.products.get(prod_id)
            if product and product["category"] not in disliked_categories:
                enhanced_recs.append(prod_id)

        # Simulate adding a 'trending' item if not enough recommendations
        if len(enhanced_recs) < len(initial_recommendations) and "trending" in context:
            # A very simplistic 'trending' item simulation
            trending_item_id = next((pid for pid, p in self.products.items() if "trending" in p.get("features", []) and pid not in enhanced_recs), None)
            if trending_item_id:
                enhanced_recs.append(trending_item_id)
        
        # Ensure we return at least some items if filtering was too aggressive
        if not enhanced_recs and initial_recommendations:
            return initial_recommendations[:3] # Fallback
            
        return enhanced_recs[:random.randint(min(len(initial_recommendations), 3), min(len(initial_recommendations), 5))] # Simulate dynamic length

    def _simulate_llm_generate_explanation(self, user_id, product_id):
        """
        Simulates an LLM generating a human-centric explanation for a recommendation.
        """
        product = self.products.get(product_id)
        user = self.users.get(user_id)

        if not product or not user:
            return "We couldn't generate a detailed explanation for this recommendation at the moment."

        explanation_templates = [
            f"Based on your interest in {random.choice(user['purchase_history_categories'] or ['similar items'])}, we think you'll love the {product['name']}. It's a fantastic {product['category']} with {random.choice(product['features'] or ['great features'])}.",
            f"The {product['name']} is a perfect match because you've previously enjoyed items in the {product['category']} category. Its {random.choice(product['features'] or ['unique design'])} stands out.",
            f"Considering your past purchases, particularly in {random.choice(user['purchase_history_categories'] or ['related categories'])}, the {product['name']} is highly recommended. You'll appreciate its {random.choice(product['features'] or ['quality and value'])}."
        ]

        return random.choice(explanation_templates)

    def _simulate_llm_process_nli(self, user_id, user_query):
        """
        Simulates an LLM interpreting natural language input to update user preferences.
        """
        user_profile = self.users.get(user_id)
        if not user_profile:
            return "Sorry, I can't find your profile. Please try again."

        updated_prefs = user_profile.get("preferences", {})
        response_message = "Understood! I'll keep that in mind for future recommendations."

        query_lower = user_query.lower()

        if "don't show me" in query_lower or "dislike" in query_lower:
            if "electronics" in query_lower:
                updated_prefs.setdefault("disliked_categories", []).append("Electronics")
                response_message = "Okay, I've noted that you'd prefer fewer electronics recommendations."
            elif "books" in query_lower:
                updated_prefs.setdefault("disliked_categories", []).append("Books")
                response_message = "Got it. I'll reduce recommendations for books."
            elif "clothing" in query_lower:
                updated_prefs.setdefault("disliked_categories", []).append("Clothing")
                response_message = "Understood. I'll try to recommend less clothing."
            else:
                response_message = "I've noted your general dislike. Can you be more specific about what you don't want to see?"
        elif "show me more" in query_lower or "interested in" in query_lower:
            if "gadgets" in query_lower or "tech" in query_lower:
                updated_prefs.setdefault("liked_categories", []).append("Electronics")
                response_message = "Great! I'll focus more on electronics and gadgets for you."
            elif "outdoor" in query_lower or "adventure" in query_lower:
                updated_prefs.setdefault("liked_categories", []).append("Outdoor & Sports")
                response_message = "Fantastic! I'll look for more outdoor and adventure gear for you."
            elif "home decor" in query_lower:
                updated_prefs.setdefault("liked_categories", []).append("Home & Kitchen")
                response_message = "Understood. I'll prioritize home decor recommendations."
            else:
                response_message = "I've noted your general interest. Can you tell me more about what you're looking for?"
        elif "trending" in query_lower or "popular" in query_lower:
            updated_prefs["focus_trending"] = True
            response_message = "I'll make sure to highlight popular and trending items for you."
        else:
            response_message = "I'm not sure how to process that specific request, but I'll do my best to learn!"

        self.users[user_id]["preferences"] = updated_prefs
        return response_message

    def get_recommendations(self, user_id, num_recommendations=5, context=""):
        """
        Generates a list of recommended product IDs for a given user, enhanced by LLM simulation.
        """
        if user_id not in self.users:
            return [], "User not found."

        initial_recs = self._content_based_recommendations(user_id, num_recommendations * 2) # Get more candidates
        final_recs = self._simulate_llm_enhance_recommendations(initial_recs, user_id, context)
        
        recommendation_details = []
        for prod_id in final_recs[:num_recommendations]:
            product = self.products.get(prod_id)
            if product:
                recommendation_details.append(product)
        
        if not recommendation_details:
            return [], "No recommendations could be generated based on your preferences and available products."

        return recommendation_details, "Recommendations generated successfully."

    def get_explanation(self, user_id, product_id):
        """
        Retrieves a human-centric explanation for why a product was recommended.
        """
        if user_id not in self.users or product_id not in self.products:
            return "Invalid user or product ID."
        return self._simulate_llm_generate_explanation(user_id, product_id)

    def process_user_query(self, user_id, query):
        """
        Processes a natural language query from the user to refine preferences.
        """
        if user_id not in self.users:
            return "User not found."
        return self._simulate_llm_process_nli(user_id, query)


# --- Example Usage ---
if __name__ == "__main__":
    # Simulated Data Ingestion and Management
    product_catalog = {
        "P001": {"name": "Smartwatch X", "description": "Latest generation smartwatch with health tracking.", "category": "Electronics", "price": 299.99, "features": ["GPS", "Heart Rate Monitor", "Waterproof", "Trending"]},
        "P002": {"name": "Noise-Cancelling Headphones", "description": "Premium headphones for immersive audio experience.", "category": "Electronics", "price": 199.99, "features": ["Bluetooth 5.0", "Active Noise Cancellation"]},
        "P003": {"name": "Python Programming Book", "description": "Beginner-friendly guide to Python programming.", "category": "Books", "price": 35.00, "features": ["Beginner-friendly", "Code examples"]},
        "P004": {"name": "Hiking Backpack 50L", "description": "Durable backpack for multi-day hiking trips.", "category": "Outdoor & Sports", "price": 120.00, "features": ["Water-resistant", "Comfortable straps", "Large capacity"]},
        "P005": {"name": "Coffee Maker Deluxe", "description": "Programmable coffee maker with built-in grinder.", "category": "Home & Kitchen", "price": 79.99, "features": ["Programmable", "Grinder", "Timer"]},
        "P006": {"name": "Classic T-Shirt", "description": "Comfortable cotton t-shirt for everyday wear.", "category": "Clothing", "price": 15.00, "features": ["100% Cotton", "Breathable"]},
        "P007": {"name": "Gaming Keyboard RGB", "description": "Mechanical gaming keyboard with customizable RGB lighting.", "category": "Electronics", "price": 89.99, "features": ["Mechanical Keys", "RGB Lighting", "Programmable Macros"]},
        "P008": {"name": "Yoga Mat Eco-Friendly", "description": "Non-slip yoga mat made from sustainable materials.", "category": "Outdoor & Sports", "price": 45.00, "features": ["Eco-friendly", "Non-slip", "Durable"]},
    }

    user_profiles = {
        "U001": {
            "name": "Alice",
            "purchase_history": ["P001", "P002", "P007"], # Buys electronics
            "purchase_history_categories": ["Electronics"],
            "preferences": {}
        },
        "U002": {
            "name": "Bob",
            "purchase_history": ["P003"], # Buys books
            "purchase_history_categories": ["Books"],
            "preferences": {}
        },
        "U003": {
            "name": "Charlie",
            "purchase_history": ["P004", "P008"], # Buys outdoor/sports
            "purchase_history_categories": ["Outdoor & Sports"],
            "preferences": {}
        }
    }

    recommender = ProductRecommender(product_catalog, user_profiles)

    print("--- Initial Recommendations for Alice (U001) ---")
    alice_recs, msg = recommender.get_recommendations("U001")
    print(f"Message: {msg}")
    for rec in alice_recs:
        print(f"- {rec['name']} ({rec['category']}) - ${rec['price']}")
    if alice_recs:
        print(f"Explanation for {alice_recs[0]['name']}: {recommender.get_explanation('U001', alice_recs[0]['name'])}")
    print("\n")

    print("--- Alice expresses dislike for Electronics ---")
    nli_response = recommender.process_user_query("U001", "I don't want to see any more electronics.")
    print(f"User U001 says: 'I don't want to see any more electronics.'")
    print(f"System: {nli_response}")
    print(f"Alice's updated preferences: {user_profiles['U001']['preferences']}\n")

    print("--- Recommendations for Alice after preference update ---")
    alice_recs_updated, msg = recommender.get_recommendations("U001", context="user_disliked_electronics")
    print(f"Message: {msg}")
    for rec in alice_recs_updated:
        print(f"- {rec['name']} ({rec['category']}) - ${rec['price']}")
    if alice_recs_updated:
        print(f"Explanation for {alice_recs_updated[0]['name']}: {recommender.get_explanation('U001', alice_recs_updated[0]['name'])}")
    print("\n")

    print("--- Bob (U002) is interested in outdoor gear ---")
    nli_response = recommender.process_user_query("U002", "Show me more outdoor and adventure gear.")
    print(f"User U002 says: 'Show me more outdoor and adventure gear.'")
    print(f"System: {nli_response}")
    print(f"Bob's updated preferences: {user_profiles['U002']['preferences']}\n")

    print("--- Recommendations for Bob after preference update ---")
    bob_recs_updated, msg = recommender.get_recommendations("U002", context="user_likes_outdoor")
    print(f"Message: {msg}")
    for rec in bob_recs_updated:
        print(f"- {rec['name']} ({rec['category']}) - ${rec['price']}")
    print("\n")

    print("--- Charlie (U003) asks for trending items ---")
    nli_response = recommender.process_user_query("U003", "What are some trending products?")
    print(f"User U003 says: 'What are some trending products?'")
    print(f"System: {nli_response}")
    print(f"Charlie's updated preferences: {user_profiles['U003']['preferences']}\n")

    print("--- Recommendations for Charlie (U003) with trending focus ---")
    charlie_recs, msg = recommender.get_recommendations("U003", context="trending")
    print(f"Message: {msg}")
    for rec in charlie_recs:
        print(f"- {rec['name']} ({rec['category']}) - ${rec['price']}")
    print("\n")