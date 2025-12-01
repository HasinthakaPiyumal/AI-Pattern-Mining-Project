import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI
from typing import List, Dict, Any
import uvicorn
import random

# --- 1. Data Layer (Simulated In-Memory) ---

products_db = {
    "p1": {"name": "Wireless Bluetooth Headphones", "description": "High-quality sound, noise-cancelling, comfortable fit, long battery life.", "category": "Electronics", "price": 79.99},
    "p2": {"name": "Smartwatch with Fitness Tracker", "description": "Tracks heart rate, steps, sleep. Notifications, waterproof, long-lasting battery.", "category": "Electronics", "price": 129.99},
    "p3": {"name": "Organic Green Tea Bags", "description": "Premium organic green tea, rich in antioxidants, soothing aroma. 100 bags.", "category": "Food & Beverage", "price": 15.00},
    "p4": {"name": "Ergonomic Office Chair", "description": "Adjustable lumbar support, breathable mesh, comfortable for long hours. Easy assembly.", "category": "Home & Office", "price": 199.50},
    "p5": {"name": "Portable SSD 1TB", "description": "Ultra-fast read/write speeds, compact design, durable. USB-C compatible.", "category": "Electronics", "price": 100.00},
    "p6": {"name": "Yoga Mat Eco-Friendly", "description": "Non-slip, thick cushioning, made from sustainable materials. Ideal for all yoga styles.", "category": "Sports & Outdoors", "price": 35.00},
    "p7": {"name": "Espresso Coffee Machine", "description": "Professional quality espresso at home. Milk frother, easy to clean, stylish design.", "category": "Home & Kitchen", "price": 250.00},
    "p8": {"name": "Noise-Cancelling Earbuds", "description": "Compact, powerful sound, excellent noise cancellation for travel and daily commute.", "category": "Electronics", "price": 99.00},
    "p9": {"name": "Bluetooth Speaker Portable", "description": "Crisp audio, deep bass, waterproof, 20-hour playtime. Perfect for parties and outdoors.", "category": "Electronics", "price": 60.00},
    "p10": {"name": "Mystery Thriller Novel", "description": "Gripping plot, unexpected twists, keeps you on the edge of your seat. Best-selling author.", "category": "Books", "price": 12.50},
}

users_db = {
    "user1": {"preferences": ["Electronics", "Sports & Outdoors"], "purchase_history": ["p1", "p2", "p6"]},
    "user2": {"preferences": ["Home & Office", "Food & Beverage"], "purchase_history": ["p4", "p3"]},
    "user3": {"preferences": ["Electronics", "Books"], "purchase_history": ["p5", "p10"]},
    "user4": {"preferences": [], "purchase_history": []} # New user
}

# --- 2. Recommendation Engine Module ---

class RecommendationEngine:
    def __init__(self, products: Dict[str, Dict[str, Any]]):
        self.products = products
        self.product_ids = list(products.keys())
        self.product_descriptions = [p["description"] for p in products.values()]

        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.product_tfidf_matrix = self.vectorizer.fit_transform(self.product_descriptions)

        self.item_similarity_matrix = cosine_similarity(self.product_tfidf_matrix)
        self.product_id_to_idx = {pid: idx for idx, pid in enumerate(self.product_ids)}

    def get_recommendations(self, user_id: str, num_recommendations: int = 5) -> List[Dict[str, Any]]:
        user_data = users_db.get(user_id)
        if not user_data:
            # For a new or unknown user, recommend popular/random items
            random_products = random.sample(self.product_ids, num_recommendations)
            return [{
                "product_id": pid,
                "reasoning_context": f"Popular item in {self.products[pid]['category']}"
            } for pid in random_products]

        purchase_history = user_data.get("purchase_history", [])
        if not purchase_history:
            # If no purchase history, use preferences or random
            random_products = random.sample(self.product_ids, num_recommendations)
            return [{
                "product_id": pid,
                "reasoning_context": f"Recommended based on general popular items in {self.products[pid]['category']}"
            } for pid in random_products]

        # Aggregate similarity scores for purchased items
        scores = {pid: 0.0 for pid in self.product_ids}
        for purchased_pid in purchase_history:
            if purchased_pid in self.product_id_to_idx:
                purchased_idx = self.product_id_to_idx[purchased_pid]
                for i, score in enumerate(self.item_similarity_matrix[purchased_idx]):
                    current_pid = self.product_ids[i]
                    if current_pid != purchased_pid and current_pid not in purchase_history:
                        scores[current_pid] += score
        
        # Sort and get top recommendations
        sorted_recommendations = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        
        recommendations = []
        for pid, score in sorted_recommendations:
            if score > 0 and pid not in purchase_history:
                recommendations.append({
                    "product_id": pid,
                    "reasoning_context": f"Similar to your past purchases: {', '.join(purchase_history)}"
                })
                if len(recommendations) >= num_recommendations:
                    break
        
        # Fallback if not enough recommendations are found
        while len(recommendations) < num_recommendations:
            available_pids = [pid for pid in self.product_ids if pid not in purchase_history and pid not in [r["product_id"] for r in recommendations]]
            if not available_pids:
                break
            new_pid = random.choice(available_pids)
            recommendations.append({
                "product_id": new_pid,
                "reasoning_context": f"Popular item in {self.products[new_pid]['category']} (fallback)"
            })

        return recommendations

# --- 3. LLM Explainer Module (Simulated) ---

class LLMExplainer:
    def __init__(self, products: Dict[str, Dict[str, Any]]):
        self.products = products

    def _craft_prompt(self, user_context: Dict[str, Any], product_details: Dict[str, Any], recommendation_context: str) -> str:
        user_preferences = ", ".join(user_context.get("preferences", [])) if user_context.get("preferences") else "no explicit preferences"
        purchase_history_names = [self.products[pid]["name"] for pid in user_context.get("purchase_history", []) if pid in self.products]
        purchase_history_str = ", ".join(purchase_history_names) if purchase_history_names else "no previous purchases"
        
        prompt = f"""
        Generate a personalized and natural explanation for why the user should consider '{product_details['name']}'.

        Product Details:
        - Name: {product_details['name']}
        - Description: {product_details['description']}
        - Category: {product_details['category']}
        - Price: ${product_details['price']:.2f}

        User Context:
        - User ID: {user_context['id']}
        - Preferences: {user_preferences}
        - Recent Purchases: {purchase_history_str}

        Recommendation Basis: {recommendation_context}

        Your explanation should be engaging, concise, and highlight relevant aspects of the product based on the user's profile and the recommendation's basis. Focus on how the product aligns with their past behavior or preferences.
        """
        return prompt

    def generate_explanation(self, user_id: str, product_id: str, recommendation_context: str) -> str:
        user_context = users_db.get(user_id, {"id": user_id, "preferences": [], "purchase_history": []})
        product_details = self.products.get(product_id)

        if not product_details:
            return f"Could not find details for product {product_id}."

        # In a real application, you would send the prompt to an LLM API.
        # For this prototype, we simulate an LLM response.
        prompt = self._craft_prompt(user_context, product_details, recommendation_context)
        
        explanation_templates = [
            f"Given your interest in {product_details['category']} and your past purchases, we think you'll love the {product_details['name']} because {product_details['description'].lower()}.",
            f"Based on {recommendation_context.lower()}, the {product_details['name']} is a great pick. It offers {product_details['description'].lower().split(',')[0]} and fits perfectly with your profile.",
            f"We've selected the {product_details['name']} for you. With its {product_details['description'].lower().split(',')[0]}, it's similar to items you've enjoyed before. It's a top choice in {product_details['category']}.",
            f"Considering your recent activity and preferences, the {product_details['name']} stands out. Its {product_details['description'].lower().split(',')[0]} makes it an excellent match for your needs."
        ]
        return random.choice(explanation_templates)

# --- 4. API Layer (Backend) ---

app = FastAPI()

# Initialize modules
rec_engine = RecommendationEngine(products_db)
llm_explainer = LLMExplainer(products_db)

@app.get("/recommendations/{user_id}", response_model=List[Dict[str, Any]])
async def get_user_recommendations(user_id: str):
    raw_recommendations = rec_engine.get_recommendations(user_id)
    
    final_recommendations = []
    for rec in raw_recommendations:
        product_id = rec["product_id"]
        recommendation_context = rec["reasoning_context"]
        
        explanation = llm_explainer.generate_explanation(
            user_id=user_id,
            product_id=product_id,
            recommendation_context=recommendation_context
        )
        
        product_details = products_db.get(product_id, {})
        final_recommendations.append({
            "product_id": product_id,
            "name": product_details.get("name", "Unknown Product"),
            "category": product_details.get("category", "Unknown"),
            "price": product_details.get("price", 0.0),
            "explanation": explanation
        })
    return final_recommendations

@app.get("/explain/{user_id}/{product_id}", response_model=Dict[str, Any])
async def get_explanation_for_product(user_id: str, product_id: str):
    product_details = products_db.get(product_id)
    if not product_details:
        return {"error": f"Product {product_id} not found"}

    # Simulate a basic recommendation context for a single product explanation
    user_history_pids = users_db.get(user_id, {}).get("purchase_history", [])
    if product_id in user_history_pids:
        reasoning_context = f"You previously purchased similar items."
    elif users_db.get(user_id, {}).get("preferences") and product_details["category"] in users_db.get(user_id).get("preferences"):
         reasoning_context = f"Matches your preferred category: {product_details['category']}."
    else:
        reasoning_context = f"Based on its general popularity and features."

    explanation = llm_explainer.generate_explanation(
        user_id=user_id,
        product_id=product_id,
        recommendation_context=reasoning_context
    )
    
    return {
        "product_id": product_id,
        "name": product_details.get("name", "Unknown Product"),
        "category": product_details.get("category", "Unknown"),
        "price": product_details.get("price", 0.0),
        "explanation": explanation
    }


if __name__ == "__main__":
    # To run this, save it as a Python file (e.g., main.py) and execute:
    # uvicorn main:app --reload --port 8000
    # Then access in browser: http://127.0.0.1:8000/recommendations/user1
    # Or: http://127.0.0.1:8000/explain/user1/p1
    print("To run the API, save this code as a Python file (e.g., main.py) and execute:")
    print("uvicorn main:app --reload --port 8000")
    print("Access recommendations: http://127.0.0.1:8000/recommendations/user1")
    print("Access explanation for a product: http://127.0.0.1:8000/explain/user1/p1")