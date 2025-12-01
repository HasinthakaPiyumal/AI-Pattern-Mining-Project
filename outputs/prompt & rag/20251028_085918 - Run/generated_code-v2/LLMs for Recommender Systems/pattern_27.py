import pandas as pd
import numpy as np

class Product:
    def __init__(self, product_id: str, name: str, category: str, features: dict, description: str):
        self.product_id = product_id
        self.name = name
        self.category = category
        self.features = features  # e.g., {'brand': 'Sony', 'sensor': 'full-frame', 'price': 1500}
        self.description = description

    def __repr__(self):
        return f"Product(id={self.product_id}, name=\"{self.name}\", category=\"{self.category}\")"

class RecommendationEngine:
    def __init__(self, products_data: list):
        self.products = {p.product_id: p for p in products_data}
        self.product_df = pd.DataFrame([vars(p) for p in products_data])

    def get_recommendations(self, user_profile: dict, num_recommendations: int = 3) -> list:
        """
        A simplified recommendation logic for demonstration purposes.
        In a real-world scenario, this would be a complex deep learning model.
        This example provides recommendations based on category and simulates 'reasons'.
        """
        print(f"\n--- Recommendation Engine: Generating recommendations for user {user_profile.get('user_id', 'N/A')} ---")
        user_preferred_categories = user_profile.get('preferred_categories', [])
        user_recent_views = user_profile.get('recent_views', []) # list of product_ids

        candidate_products = []
        recommendation_reasons = {}

        # Priority 1: Products from recently viewed items' categories
        if user_recent_views:
            viewed_product_categories = set()
            for p_id in user_recent_views:
                if p_id in self.products:
                    viewed_product_categories.add(self.products[p_id].category)
            
            for category in viewed_product_categories:
                category_products = self.product_df[self.product_df['category'] == category].to_dict(orient='records')
                for prod_data in category_products:
                    product_obj = self.products[prod_data['product_id']]
                    if product_obj.product_id not in [p.product_id for p in candidate_products]:
                        candidate_products.append(product_obj)
                        recommendation_reasons[product_obj.product_id] = f"Based on your recent interest in {category} items."
        
        # Priority 2: Products from explicitly preferred categories
        for category in user_preferred_categories:
            category_products = self.product_df[self.product_df['category'] == category].to_dict(orient='records')
            for prod_data in category_products:
                product_obj = self.products[prod_data['product_id']]
                if product_obj.product_id not in [p.product_id for p in candidate_products]:
                    candidate_products.append(product_obj)
                    recommendation_reasons[product_obj.product_id] = f"You've shown a preference for {category} products."

        # Fallback: Random products if not enough candidates
        if len(candidate_products) < num_recommendations:
            remaining_needed = num_recommendations - len(candidate_products)
            all_product_ids = list(self.products.keys())
            np.random.shuffle(all_product_ids)
            
            for p_id in all_product_ids:
                product_obj = self.products[p_id]
                if product_obj.product_id not in [p.product_id for p in candidate_products]:
                    candidate_products.append(product_obj)
                    recommendation_reasons[product_obj.product_id] = "Popular choice among similar users."
                    if len(candidate_products) >= num_recommendations:
                        break

        # Return top N unique recommendations with their reasons
        recommended_products_with_reasons = []
        for product in candidate_products[:num_recommendations]:
            reason = recommendation_reasons.get(product.product_id, "Recommended based on general popularity.")
            recommended_products_with_reasons.append({
                'product': product,
                'reason_keywords': reason # This will be used by the LLM for explanation
            })
        
        return recommended_products_with_reasons

# Example Usage (for testing the module directly)
if __name__ == "__main__":
    sample_products = [
        Product("P001", "DSLR Camera", "Electronics", {'brand': 'Canon', 'sensor': 'APS-C', 'price': 800, 'megapixels': 24, 'feature_highlight': 'excellent image quality'}, "A professional DSLR camera for enthusiasts."),
        Product("P002", "Mirrorless Camera", "Electronics", {'brand': 'Sony', 'sensor': 'Full-Frame', 'price': 1800, 'megapixels': 32, 'feature_highlight': 'compact and powerful'}, "High-performance mirrorless camera, great for travel."),
        Product("P003", "Tripod Stand", "Accessories", {'brand': 'Manfrotto', 'material': 'aluminum', 'price': 120, 'feature_highlight': 'stable and lightweight'}, "Durable and portable tripod for all cameras."),
        Product("P004", "Laptop Pro", "Electronics", {'brand': 'Dell', 'processor': 'i7', 'ram': 16, 'price': 1200, 'feature_highlight': 'fast performance for professionals'}, "Powerful laptop for work and creativity."),
        Product("P005", "Gaming Mouse", "Peripherals", {'brand': 'Logitech', 'type': 'wireless', 'dpi': 16000, 'price': 70, 'feature_highlight': 'ergonomic design'}, "Precision gaming mouse with customizable buttons."),
        Product("P006", "Bluetooth Speaker", "Audio", {'brand': 'JBL', 'battery_life': '12 hours', 'waterproof': True, 'price': 99, 'feature_highlight': 'rich bass and portable'}, "Waterproof speaker for outdoor adventures."),
        Product("P007", "Hiking Boots", "Outdoor Gear", {'brand': 'Merrell', 'material': 'leather', 'waterproof': True, 'price': 150, 'feature_highlight': 'durable and comfortable'}, "Rugged boots for challenging trails."),
        Product("P008", "Backpack 60L", "Outdoor Gear", {'brand': 'Osprey', 'capacity': 60, 'material': 'nylon', 'price': 200, 'feature_highlight': 'spacious and ergonomic'}, "Large capacity backpack for multi-day treks."),
    ]

    engine = RecommendationEngine(sample_products)

    user_1_profile = {
        'user_id': 'user_A',
        'preferred_categories': ['Electronics'],
        'recent_views': ['P001', 'P003'],
        'purchase_history': ['P001'],
        'demographics': {'age': 30, 'location': 'NY'}
    }

    user_2_profile = {
        'user_id': 'user_B',
        'preferred_categories': ['Outdoor Gear'],
        'recent_views': ['P007'],
        'purchase_history': [],
        'demographics': {'age': 25, 'location': 'CA'}
    }

    recs_user1 = engine.get_recommendations(user_1_profile)
    for rec in recs_user1:
        print(f"Recommended: {rec['product'].name}, Reason: {rec['reason_keywords']}")

    recs_user2 = engine.get_recommendations(user_2_profile, num_recommendations=2)
    for rec in recs_user2:
        print(f"Recommended: {rec['product'].name}, Reason: {rec['reason_keywords']}")