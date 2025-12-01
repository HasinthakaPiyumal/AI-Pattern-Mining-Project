import json
import random

class Product:
    def __init__(self, product_id: str, name: str, description: str, image_url: str, categories: list[str]):
        self.product_id = product_id
        self.name = name
        self.description = description
        self.image_url = image_url
        self.categories = categories

class RecommendationSystem:
    def __init__(self, product_catalog: list[Product]):
        self.product_catalog = {p.product_id: p for p in product_catalog}
        self.all_product_ids = list(self.product_catalog.keys())

    def get_recommendations(self, user_history: list[str], num_recommendations: int = 5) -> str:
        # For simplicity, this example recommends random products not in user history
        # In a real system, this would involve more sophisticated logic (e.g., collaborative filtering, content-based filtering)
        
        seen_product_ids = set(user_history)
        available_for_recommendation = [pid for pid in self.all_product_ids if pid not in seen_product_ids]
        
        if len(available_for_recommendation) < num_recommendations:
            # If not enough new products, just recommend what's available
            recommended_ids = random.sample(available_for_recommendation, len(available_for_recommendation))
        else:
            recommended_ids = random.sample(available_for_recommendation, num_recommendations)

        recommendations_data = []
        for prod_id in recommended_ids:
            product = self.product_catalog.get(prod_id)
            if product:
                recommendations_data.append({
                    "product_id": product.product_id,
                    "product_name": product.name,
                    "image_url": product.image_url,
                    "description": product.description
                })
        
        return json.dumps(recommendations_data, indent=4)

# --- Example Usage ---
if __name__ == "__main__":
    # Simulate a product catalog
    catalog = [
        Product("P001", "Laptop Pro X", "High-performance laptop for professionals.", "http://example.com/img/laptop_pro_x.jpg", ["electronics", "computers"]),
        Product("P002", "Wireless Mouse", "Ergonomic wireless mouse.", "http://example.com/img/wireless_mouse.jpg", ["electronics", "accessories"]),
        Product("P003", "Mechanical Keyboard", "Gaming mechanical keyboard with RGB.", "http://example.com/img/keyboard.jpg", ["electronics", "gaming"]),
        Product("P004", "USB-C Hub", "Multi-port USB-C adapter.", "http://example.com/img/usb_c_hub.jpg", ["electronics", "accessories"]),
        Product("P005", "Monitor 27-inch 4K", "Ultra HD display for vivid visuals.", "http://example.com/img/monitor_4k.jpg", ["electronics", "monitors"]),
        Product("P006", "Gaming Headset", "Immersive sound for gaming.", "http://example.com/img/headset.jpg", ["electronics", "gaming"]),
        Product("P007", "Webcam Full HD", "Crisp video for online meetings.", "http://example.com/img/webcam.jpg", ["electronics", "accessories"]),
    ]

    # Initialize the recommendation system
    recommender = RecommendationSystem(catalog)

    # Simulate user history
    user1_history = ["P001", "P002"]
    user2_history = ["P003"]
    user3_history = ["P001", "P005", "P006", "P007"]

    print("\n--- Recommendations for User 1 (viewed P001, P002) ---")
    recommendations1 = recommender.get_recommendations(user1_history, num_recommendations=3)
    print(recommendations1)

    print("\n--- Recommendations for User 2 (viewed P003) ---")
    recommendations2 = recommender.get_recommendations(user2_history, num_recommendations=4)
    print(recommendations2)

    print("\n--- Recommendations for User 3 (viewed P001, P005, P006, P007) ---")
    recommendations3 = recommender.get_recommendations(user3_history, num_recommendations=2)
    print(recommendations3)