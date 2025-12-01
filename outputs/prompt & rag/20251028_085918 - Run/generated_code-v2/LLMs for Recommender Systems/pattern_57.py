import pandas as pd

products_data = [
    {"product_id": "P001", "name": "Trail Running Shoes", "category": "Footwear", "features": "Excellent grip, cushioning, waterproof, sustainable materials"},
    {"product_id": "P002", "name": "Hiking Backpack 30L", "category": "Outdoor Gear", "features": "Lightweight, durable, hydration compatible, multiple compartments"},
    {"product_id": "P003", "name": "Smartwatch Series 7", "category": "Electronics", "features": "Heart rate monitor, GPS, long battery life, cellular support"},
    {"product_id": "P004", "name": "Wireless Noise-Cancelling Headphones", "category": "Electronics", "features": "Superior sound, active noise cancellation, comfortable, 30-hour battery"},
    {"product_id": "P005", "name": "Yoga Mat Pro", "category": "Fitness", "features": "Eco-friendly material, non-slip, extra thick, portable"},
    {"product_id": "P006", "name": "Water Bottle 1L Stainless Steel", "category": "Outdoor Gear", "features": "Insulated, leak-proof, wide mouth, BPA-free"},
    {"product_id": "P007", "name": "Road Running Shoes Light", "category": "Footwear", "features": "Ultra-lightweight, responsive cushioning, breathable mesh, neutral support"},
    {"product_id": "P008", "name": "Fitness Tracker Band", "category": "Fitness", "features": "Steps, calories, sleep tracking, waterproof, color display"}
]

user_purchases_data = {
    "U001": ["P001", "P002", "P006"], # Buys outdoor gear, trail shoes
    "U002": ["P003", "P004"],      # Buys electronics
    "U003": ["P005"]             # Buys fitness
}

products_df = pd.DataFrame(products_data)

def get_product_details(product_id):
    return products_df[products_df["product_id"] == product_id].iloc[0].to_dict() if not products_df[products_df["product_id"] == product_id].empty else None

def simulate_recommendations(user_id, num_recommendations=3):
    user_purchased_products = user_purchases_data.get(user_id, [])
    user_purchased_categories = products_df[products_df["product_id"].isin(user_purchased_products)]["category"].unique()

    recommendations = []

    # Try to recommend based on past categories
    for category in user_purchased_categories:
        candidate_products = products_df[
            (products_df["category"] == category) & 
            (~products_df["product_id"].isin(user_purchased_products))
        ]
        if not candidate_products.empty:
            for _, row in candidate_products.sample(min(len(candidate_products), num_recommendations - len(recommendations))).iterrows():
                recommendations.append((row["product_id"], "similar_category"))
                if len(recommendations) >= num_recommendations:
                    return recommendations
    
    # Fill with popular items if not enough recommendations
    if len(recommendations) < num_recommendations:
        all_product_ids = products_df["product_id"].tolist()
        unpurchased_popular = [pid for pid in all_product_ids if pid not in user_purchased_products]
        for pid in unpurchased_popular[:num_recommendations - len(recommendations)]:
            recommendations.append((pid, "popular_item"))

    return recommendations

def generate_explanation(user_id, product_details, recommendation_reason, user_query=None):
    product_name = product_details["name"]
    product_category = product_details["category"]
    product_features = product_details["features"]

    base_explanation = f