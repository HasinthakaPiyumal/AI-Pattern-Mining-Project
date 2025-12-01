user_data_store = {
    "user123": {
        "browsing_history": ["prod_A", "prod_C", "cat_electronics"],
        "purchase_history": ["prod_B", "prod_D"],
        "favorite_category": "electronics"
    },
    "user456": {
        "browsing_history": ["prod_E", "cat_books"],
        "purchase_history": [],
        "favorite_category": "books"
    }
}

product_catalog_store = {
    "prod_A": {
        "name": "Wireless Headphones",
        "description": "High-fidelity wireless headphones with noise cancellation.",
        "category": "electronics",
        "features": ["Bluetooth 5.0", "30-hour battery", "Active Noise Cancellation"],
        "price": 199.99,
        "average_rating": 4.5,
        "number_of_reviews": 1200,
        "related_products": ["prod_B", "prod_F"]
    },
    "prod_B": {
        "name": "Smartwatch Pro",
        "description": "Advanced smartwatch with health tracking and GPS.",
        "category": "electronics",
        "features": ["Heart Rate Monitor", "GPS", "Waterproof"],
        "price": 249.99,
        "average_rating": 4.7,
        "number_of_reviews": 900,
        "related_products": ["prod_A", "prod_G"]
    },
    "prod_C": {
        "name": "Ergonomic Office Chair",
        "description": "Comfortable office chair designed for long working hours.",
        "category": "office_furniture",
        "features": ["Adjustable lumbar support", "Breathable mesh", "Reclining function"],
        "price": 299.00,
        "average_rating": 4.2,
        "number_of_reviews": 500,
        "related_products": ["prod_H"]
    },
    "prod_D": {
        "name": "Portable SSD 1TB",
        "description": "Ultra-fast external SSD for data storage and backup.",
        "category": "electronics",
        "features": ["USB-C 3.2 Gen2", "1050 MB/s Read Speed", "Compact Design"],
        "price": 129.99,
        "average_rating": 4.8,
        "number_of_reviews": 750,
        "related_products": ["prod_I"]
    },
    "prod_E": {
        "name": "Classic Novel Collection",
        "description": "A timeless collection of literary classics.",
        "category": "books",
        "features": ["Hardcover Edition", "Set of 10 Books", "Premium Paper"],
        "price": 79.99,
        "average_rating": 4.6,
        "number_of_reviews": 300,
        "related_products": ["prod_J"]
    }
}

def get_simulated_recommendations(user_id):
    if user_id == "user123":
        return [
            {"product_id": "prod_B", "reason": "similar to viewed items in electronics"},
            {"product_id": "prod_A", "reason": "popular with users who bought similar products"}
        ]
    elif user_id == "user456":
        return [
            {"product_id": "prod_E", "reason": "based on your browsing history in books"}
        ]
    return []

def call_llm_api(prompt):
    if "Wireless Headphones" in prompt and "noise cancellation" in prompt:
        return "Based on your interest in electronics and recent views, we recommend the Wireless Headphones. They offer high-fidelity sound and excellent active noise cancellation, perfect for an immersive audio experience, similar to other premium audio gear you\'ve explored."
    elif "Smartwatch Pro" in prompt and "health tracking" in prompt:
        return "Given your past purchases and engagement with electronics, the Smartwatch Pro is a great fit. It features advanced health tracking and GPS, making it ideal for an active lifestyle, complementing your existing tech gadgets."
    elif "Classic Novel Collection" in prompt and "books" in prompt:
        return "Since you\'ve been browsing books, we think you\'ll appreciate the Classic Novel Collection. This set offers a rich literary experience, bringing together timeless stories in a premium hardcover format, aligning with your interest in reading."
    return "We recommend this product because it aligns with your preferences."

def generate_explanation_prompt(user_id, recommended_product_id, recommendation_reason, user_data, product_data):
    user_browsing = ", ".join(user_data.get("browsing_history", []))
    user_purchases = ", ".join(user_data.get("purchase_history", []))
    fav_category = user_data.get("favorite_category", "")

    prod_name = product_data.get("name", "unknown product")
    prod_desc = product_data.get("description", "")
    prod_category = product_data.get("category", "")
    prod_features = ", ".join(product_data.get("features", []))
    prod_rating = product_data.get("average_rating", "N/A")

    prompt = f"""
Generate a personalized and persuasive explanation for recommending a product on an e-commerce platform.

User Context (User ID: {user_id}):
- Recently browsed: {user_browsing if user_browsing else 'nothing specific'}
- Previously purchased: {user_purchases if user_purchases else 'nothing specific'}
- Favorite category: {fav_category if fav_category else 'not specified'}

Recommended Product (ID: {recommended_product_id}):
- Name: {prod_name}
- Category: {prod_category}
- Description: {prod_desc}
- Key Features: {prod_features}
- Average Rating: {prod_rating}

High-level Recommendation Reason: {recommendation_reason}

Instructions:
- Explain why this product is a good fit for the user, referencing their context and the product's features.
- Make the explanation natural, engaging, and trustworthy.
- Keep it concise, around 2-3 sentences.
- Focus on personalization and relevance.
"""
    return prompt

def get_product_recommendations_with_explanations(user_id):
    recommendations = get_simulated_recommendations(user_id)
    user_data = user_data_store.get(user_id, {})
    
    results = []
    for rec in recommendations:
        product_id = rec["product_id"]
        recommendation_reason = rec["reason"]
        product_data = product_catalog_store.get(product_id, {})
        
        if not product_data:
            results.append({"product_id": product_id, "explanation": f"Could not retrieve details for product {product_id}."})
            continue

        prompt = generate_explanation_prompt(user_id, product_id, recommendation_reason, user_data, product_data)
        explanation = call_llm_api(prompt)
        
        results.append({
            "product_id": product_id,
            "product_name": product_data.get("name"),
            "explanation": explanation
        })
    return results

if __name__ == "__main__":
    print("\n--- Recommendations for user123 ---")
    user1_recs = get_product_recommendations_with_explanations("user123")
    for rec in user1_recs:
        print(f"Product: {rec['product_name']} (ID: {rec['product_id']})\nExplanation: {rec['explanation']}\n")

    print("\n--- Recommendations for user456 ---")
    user2_recs = get_product_recommendations_with_explanations("user456")
    for rec in user2_recs:
        print(f"Product: {rec['product_name']} (ID: {rec['product_id']})\nExplanation: {rec['explanation']}\n")

    print("\n--- Recommendations for unknown_user ---")
    unknown_user_recs = get_product_recommendations_with_explanations("unknown_user")
    for rec in unknown_user_recs:
        print(f"Product: {rec['product_name']} (ID: {rec['product_id']})\nExplanation: {rec['explanation']}\n")
