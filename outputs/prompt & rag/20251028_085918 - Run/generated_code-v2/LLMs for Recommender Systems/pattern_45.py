import pandas as pd
import random

# --- 1. Sample Data ---
products_data = [
    {"id": "P001", "name": "Wireless Noise-Cancelling Headphones", "category": "Electronics", "tags": ["audio", "travel", "work", "premium"], "description": "High-fidelity sound with industry-leading noise cancellation."},
    {"id": "P002", "name": "Ergonomic Office Chair", "category": "Furniture", "tags": ["home office", "comfort", "productivity", "health"], "description": "Designed for long hours, promoting good posture and comfort."},
    {"id": "P003", "name": "Smartwatch with Heart Rate Monitor", "category": "Electronics", "tags": ["fitness", "health", "tech", "wearable"], "description": "Track your health and stay connected on the go."},
    {"id": "P004", "name": "Organic Coffee Beans (Dark Roast)", "category": "Food & Beverage", "tags": ["coffee", "organic", "breakfast", "gourmet"], "description": "Rich, bold flavor from ethically sourced organic beans."},
    {"id": "P005", "name": "Yoga Mat with Carrying Strap", "category": "Fitness", "tags": ["yoga", "exercise", "wellness", "portable"], "description": "Non-slip and durable for all your yoga and workout needs."},
    {"id": "P006", "name": "Portable Bluetooth Speaker", "category": "Electronics", "tags": ["audio", "outdoor", "party", "compact"], "description": "Powerful sound in a compact, waterproof design."},
    {"id": "P007", "name": "Gourmet Chocolate Assortment", "category": "Food & Beverage", "tags": ["dessert", "gift", "premium", "sweet"], "description": "Handcrafted chocolates perfect for any occasion."},
    {"id": "P008", "name": "Laptop Backpack with USB Charging", "category": "Accessories", "tags": ["travel", "work", "tech", "convenience"], "description": "Secure and stylish backpack with integrated charging port."}
]

users_data = {
    "U001": {"past_purchases": ["P001", "P003"], "browsing_history": ["P006"], "preferences": {"category": "Electronics", "tags": ["tech", "audio", "fitness"]}},
    "U002": {"past_purchases": ["P002", "P005"], "browsing_history": ["P004"], "preferences": {"category": "Home", "tags": ["comfort", "wellness", "organic"]}},
}

products_df = pd.DataFrame(products_data)

# --- 2. Simplified Recommendation Engine ---
def get_recommendations_for_user(user_id, products_df, users_data, num_recommendations=3):
    user_info = users_data.get(user_id)
    if not user_info:
        return []

    recommended_products = []

    user_past_purchases = user_info["past_purchases"]
    user_browsing_history = user_info["browsing_history"]
    user_prefs_tags = user_info["preferences"].get("tags", [])

    all_relevant_product_ids = list(set(user_past_purchases + user_browsing_history))
    
    # Prioritize items similar to past purchases/browsing
    for prod_id in all_relevant_product_ids:
        product = products_df[products_df["id"] == prod_id].iloc[0]
        # Find other products in the same category or with similar tags
        similar_products = products_df[
            ((products_df["category"] == product["category"]) |
             products_df["tags"].apply(lambda x: any(tag in x for tag in product["tags"]))) &
            (~products_df["id"].isin(user_past_purchases)) &
            (~products_df["id"].isin(user_browsing_history)) &
            (~products_df["id"].isin([rec["product"]["id"] for rec in recommended_products]))
        ]
        if not similar_products.empty:
            # Pick one random similar product
            rec_prod = similar_products.sample(1).iloc[0]
            recommended_products.append({"product": rec_prod.to_dict(), "reason": f"Similar to your past interaction with '{product['name']}' ({product['id']})."})
            if len(recommended_products) >= num_recommendations:
                break
    
    # If not enough, recommend based on general preferences or popular items
    if len(recommended_products) < num_recommendations:
        # Example: recommend based on user preferences tags
        for _ in range(num_recommendations - len(recommended_products)):
            potential_recs = products_df[
                products_df["tags"].apply(lambda x: any(tag in x for tag in user_prefs_tags)) &
                (~products_df["id"].isin(user_past_purchases)) &
                (~products_df["id"].isin(user_browsing_history)) &
                (~products_df["id"].isin([rec["product"]["id"] for rec in recommended_products]))
            ]
            if not potential_recs.empty:
                rec_prod = potential_recs.sample(1).iloc[0]
                recommended_products.append({"product": rec_prod.to_dict(), "reason": f"Matches your general interest in '{', '.join(user_prefs_tags)}'."})
            else:
                # Fallback: recommend random non-purchased/browsed items
                unseen_products = products_df[
                    (~products_df["id"].isin(user_past_purchases)) &
                    (~products_df["id"].isin(user_browsing_history)) &
                    (~products_df["id"].isin([rec["product"]["id"] for rec in recommended_products]))
                ]
                if not unseen_products.empty:
                    rec_prod = unseen_products.sample(1).iloc[0]
                    recommended_products.append({"product": rec_prod.to_dict(), "reason": "Based on general popularity."})
                else:
                    break # No more products to recommend

    return recommended_products[:num_recommendations]

# --- 3. Simulated LLM Explainer ---
def generate_llm_explanation(user_info, recommended_product_details, recommendation_reason, query=None):
    product_name = recommended_product_details["name"]
    product_category = recommended_product_details["category"]
    product_description = recommended_product_details["description"]
    product_tags = ", ".join(recommended_product_details["tags"])

    explanation_base = f"We thought you might like the '{product_name}' ({product_category}). Here's why:\n"

    if "Similar to your past interaction" in recommendation_reason:
        related_item_info = recommendation_reason.split("with '")[1].split("'")[0]
        explanation_base += f"- It's similar to '{related_item_info}', which you've shown interest in before. For example, both share characteristics like {product_tags}.\n"
    elif "Matches your general interest" in recommendation_reason:
        user_interests = recommendation_reason.split("in '")[1].split("'")[0]
        explanation_base += f"- This item aligns with your stated interests in {user_interests}, offering features like {product_tags}.\n"
    elif "Based on general popularity" in recommendation_reason:
        explanation_base += f"- This is a highly-rated and popular item in the {product_category} category, known for its {product_description.lower().split(',')[0]}.\n"
    else:
        explanation_base += f"- Our system identified this as a good match for you based on its {product_tags} features and your overall profile.\n"
    
    if query:
        if "more about features" in query.lower():
            return explanation_base + f"Specifically, the '{product_name}' boasts features such as: {product_description}. It's quite robust and highly functional!"
        elif "why specifically for me" in query.lower() or "personal" in query.lower():
            user_past_items = [products_df[products_df['id'] == pid]['name'].iloc[0] for pid in user_info.get("past_purchases", []) if pid in products_df['id'].values] 
            user_browsed_items = [products_df[products_df['id'] == pid]['name'].iloc[0] for pid in user_info.get("browsing_history", []) if pid in products_df['id'].values]
            return explanation_base + f"Considering your past interest in items like {', '.join(user_past_items or ['other products'])} and your browsing of {', '.join(user_browsed_items or ['various products'])}, this recommendation takes into account your specific tastes and recent activities."
        else:
            return explanation_base + f"You asked about '{query}'. We believe this product is a great fit because it offers {product_tags}. Is there anything else you'd like to know?"
    
    return explanation_base + "Would you like to know more about its features, or why it's a personal fit for you?"

# --- 4. Main Application Flow (Demonstration) ---
def run_explainer_demo():
    print("Welcome to the Explainable E-commerce Recommender Demo!")
    user_id = "U001" # We'll use a hardcoded user for the demo
    user_info = users_data.get(user_id)
    if not user_info:
        print(f"User {user_id} not found.")
        return

    print(f"\n--- User: {user_id} ---")
    past_purchased_names = [products_df[products_df['id'] == pid]['name'].iloc[0] for pid in user_info['past_purchases'] if pid in products_df['id'].values]
    browsed_names = [products_df[products_df['id'] == pid]['name'].iloc[0] for pid in user_info['browsing_history'] if pid in products_df['id'].values]

    print(f"Past Purchases: {', '.join(past_purchased_names)}")
    print(f"Browsing History: {', '.join(browsed_names)}")

    print("\n--- Generating Recommendations ---")
    recommendations_with_reasons = get_recommendations_for_user(user_id, products_df, users_data)

    if not recommendations_with_reasons:
        print("No recommendations found for this user.")
        return

    print("\n--- Your Personalized Recommendations ---")
    for i, rec_item in enumerate(recommendations_with_reasons):
        product = rec_item["product"]
        reason_code = rec_item["reason"]
        print(f"\nRecommendation {i+1}: '{product['name']}' ({product['category']})")
        print(f"  Initial Explanation (LLM Generated):")
        explanation = generate_llm_explanation(user_info, product, reason_code)
        print(explanation)

        # Simulate interactive follow-up for the first recommendation
        if i == 0:
            print("\n  (Simulating User Interaction for this recommendation)")
            follow_up_query_1 = "Tell me more about its features."
            print(f"  User asks: '{follow_up_query_1}'")
            refined_explanation_1 = generate_llm_explanation(user_info, product, reason_code, query=follow_up_query_1)
            print(f"  Refined Explanation: {refined_explanation_1}")

            follow_up_query_2 = "Why is it a personal fit for me?"
            print(f"  User asks: '{follow_up_query_2}'")
            refined_explanation_2 = generate_llm_explanation(user_info, product, reason_code, query=follow_up_query_2)
            print(f"  Refined Explanation: {refined_explanation_2}")

if __name__ == "__main__":
    run_explainer_demo()