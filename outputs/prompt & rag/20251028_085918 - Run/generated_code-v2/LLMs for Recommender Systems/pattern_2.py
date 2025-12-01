import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pandas as pd # For data handling, if needed

# --- 1. LLM Integration (Content Interpretation & Reasoning) ---
class LLMContentInterpreter:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initializes the LLM Content Interpreter.
        In a real scenario, this would involve loading a fine-tuned LLM
        or a distilled version of it.
        We use a SentenceTransformer as a proxy for generating text embeddings.
        """
        try:
            self.model = SentenceTransformer(model_name)
            print(f"Loaded SentenceTransformer model: {model_name}")
        except ImportError:
            print("SentenceTransformer not installed. Please install with: pip install sentence-transformers")
            print("Using a mock embedding generator instead.")
            self.model = None

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Generates a semantic embedding for the given text.
        This simulates the LLM's ability to capture deep semantic information.
        """
        if self.model:
            return self.model.encode(text, convert_to_tensor=False)
        else:
            # Mock embedding for demonstration if SentenceTransformer is not available
            return np.random.rand(384) # Common output dimension for MiniLM

    def generate_reasoning_prompt(self, user_history: list, product_description: str) -> str:
        """
        Generates a natural language instruction/reasoning prompt based on user history.
        In a real LLM setup, this prompt would be fed to the LLM to generate
        additional features or direct recommendations.
        """
        history_summary = "; ".join([f"previously liked '{item['product_name']}' from category '{item['category']}'" for item in user_history])
        return (
            f"Based on user's past preferences ({history_summary}), "
            f"explain why the product '{product_description[:50]}...' might be a good fit, "
            f"or suggest a similar type of item they would enjoy."
        )

# --- 2. Data Simulation ---
def load_simulated_data():
    products_data = [
        {"id": "P001", "name": "Organic Green Tea Pack", "description": "High-quality organic green tea leaves from sustainable farms. Rich in antioxidants. Perfect for a healthy lifestyle.", "category": "Beverages", "reviews": ["Great taste, very refreshing!", "Love the organic quality.", "My daily dose of health."]},
        {"id": "P002", "name": "Noise-Cancelling Headphones", "description": "Premium over-ear headphones with active noise cancellation. Enjoy immersive audio with deep bass and clear trebles.", "category": "Electronics", "reviews": ["Amazing sound quality!", "Comfortable for long listening sessions.", "Blocks out all office noise."]},
        {"id": "P003", "name": "Ergonomic Office Chair", "description": "Adjustable ergonomic chair designed for maximum comfort and support during long working hours. Improves posture.", "category": "Office Furniture", "reviews": ["Transformed my home office setup.", "Very sturdy and comfortable.", "Easy to assemble."]},
        {"id": "P004", "name": "Herbal Sleep Aid Supplement", "description": "Natural herbal blend to promote restful sleep. Non-habit forming formula with chamomile and valerian root.", "category": "Health & Wellness", "reviews": ["Helped me fall asleep faster.", "Wake up feeling refreshed.", "No grogginess."]},
        {"id": "P005", "name": "Smart Fitness Tracker", "description": "Monitor your heart rate, steps, and sleep patterns. Features GPS tracking for outdoor activities and smartphone notifications.", "category": "Electronics", "reviews": ["Accurate tracking, love the app!", "Motivates me to stay active.", "Good battery life."]},
        {"id": "P006", "name": "Artisan Coffee Beans (Dark Roast)", "description": "Premium dark roast coffee beans sourced from ethical farms. Rich, bold flavor with notes of chocolate and caramel.", "category": "Beverages", "reviews": ["Best coffee I've ever had!", "Smooth and full-bodied.", "Definitely buying again."]},
        {"id": "P007", "name": "Yoga Mat (Eco-Friendly)", "description": "Non-slip, extra-thick yoga mat made from sustainable, eco-friendly materials. Provides excellent cushioning.", "category": "Sports & Outdoors", "reviews": ["Perfect grip for yoga.", "Comfortable and thick.", "Happy it's eco-friendly."]}
    ]

    user_history_data = {
        "U001": [
            {"product_id": "P001", "product_name": "Organic Green Tea Pack", "category": "Beverages"},
            {"product_id": "P006", "product_name": "Artisan Coffee Beans (Dark Roast)", "category": "Beverages"}
        ],
        "U002": [
            {"product_id": "P002", "product_name": "Noise-Cancelling Headphones", "category": "Electronics"},
            {"product_id": "P005", "product_name": "Smart Fitness Tracker", "category": "Electronics"}
        ]
    }
    return products_data, user_history_data

# --- 3. Recommendation Engine ---
class LLMProductRecommender:
    def __init__(self, llm_interpreter: LLMContentInterpreter):
        self.llm_interpreter = llm_interpreter
        self.products = {}
        self.product_embeddings = {}

    def ingest_products(self, products_data: list):
        """
        Processes product data and generates embeddings using the LLM interpreter.
        """
        print("Ingesting products and generating embeddings...")
        for product in products_data:
            self.products[product["id"]] = product
            combined_text = (
                f"{product['name']}. {product['description']}. "
                + " ".join(product["reviews"])
            )
            self.product_embeddings[product["id"]] = self.llm_interpreter.get_embedding(combined_text)
        print(f"Ingested {len(self.products)} products.")

    def get_cold_start_recommendations(self, new_product_description: str, top_n: int = 3):
        """
        Provides recommendations for a 'cold-start' product (new product without history).
        Uses LLM to interpret the new product's description and finds similar existing products.
        """
        print(f"\nGetting cold-start recommendations for: '{new_product_description[:50]}...'\n")
        new_product_embedding = self.llm_interpreter.get_embedding(new_product_description)

        similarities = []
        for prod_id, embedding in self.product_embeddings.items():
            sim = cosine_similarity(new_product_embedding.reshape(1, -1), embedding.reshape(1, -1))[0][0]
            similarities.append((prod_id, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        top_recommendations = []
        for prod_id, sim in similarities[:top_n]:
            top_recommendations.append(self.products[prod_id])

        return top_recommendations

    def get_personalized_recommendations(self, user_id: str, user_history: list, top_n: int = 3):
        """
        Provides personalized recommendations based on user history and LLM-enhanced interpretation.
        This demonstrates how LLMs can generate reasoning or capture nuanced user preferences.
        """
        print(f"\nGetting personalized recommendations for user '{user_id}' based on history...\n")

        if not user_history:
            print("User history is empty. Reverting to popular items or broader category suggestions.")
            return []

        # Aggregate embeddings of items the user has interacted with
        user_history_embeddings = []
        for item in user_history:
            if item["product_id"] in self.product_embeddings:
                user_history_embeddings.append(self.product_embeddings[item["product_id"]])

        if not user_history_embeddings:
            print("Could not find embeddings for user's history items. Cannot make personalized recommendations.")
            return []

        # Average user history embeddings to get a user profile embedding
        user_profile_embedding = np.mean(user_history_embeddings, axis=0)

        # Generate a reasoning prompt (conceptual use for LLM)
        # This prompt itself could be used with a larger LLM to directly generate recommendations
        # or guide a subsequent retrieval step. For this simplified example, we'll just print it.
        example_product_for_reasoning = self.products[user_history[0]["product_id"]]["description"] # Use a product from history
        reasoning_prompt = self.llm_interpreter.generate_reasoning_prompt(user_history, example_product_for_reasoning)
        print(f"LLM Reasoning Hint (for advanced LLM usage): {reasoning_prompt}\n")

        # Find similar products to the user's profile, excluding already seen items
        seen_product_ids = {item["product_id"] for item in user_history}
        similarities = []
        for prod_id, embedding in self.product_embeddings.items():
            if prod_id not in seen_product_ids:
                sim = cosine_similarity(user_profile_embedding.reshape(1, -1), embedding.reshape(1, -1))[0][0]
                similarities.append((prod_id, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        top_recommendations = []
        for prod_id, sim in similarities[:top_n]:
            top_recommendations.append(self.products[prod_id])

        return top_recommendations

# --- Main Execution ---
if __name__ == "__main__":
    # Initialize LLM Content Interpreter
    # This represents the fine-tuned or distilled LLM for content interpretation
    llm_interpreter = LLMContentInterpreter()

    # Load simulated data
    products_data, user_history_data = load_simulated_data()

    # Initialize Recommender and ingest products
    recommender = LLMProductRecommender(llm_interpreter)
    recommender.ingest_products(products_data)

    # --- Demonstrate Cold-Start Recommendation ---
    new_product_description = "Revolutionary AI-powered smart home assistant that learns your habits."
    cold_start_recs = recommender.get_cold_start_recommendations(new_product_description, top_n=2)
    print("Cold-Start Recommendations:")
    for rec in cold_start_recs:
        print(f"- {rec['name']} ({rec['category']})")

    # --- Demonstrate Personalized Recommendation ---
    user_id_1 = "U001"
    personalized_recs_1 = recommender.get_personalized_recommendations(user_id_1, user_history_data[user_id_1], top_n=2)
    print(f"Personalized Recommendations for {user_id_1}:")
    for rec in personalized_recs_1:
        print(f"- {rec['name']} ({rec['category']})")

    user_id_2 = "U002"
    personalized_recs_2 = recommender.get_personalized_recommendations(user_id_2, user_history_data[user_id_2], top_n=2)
    print(f"Personalized Recommendations for {user_id_2}:")
    for rec in personalized_recs_2:
        print(f"- {rec['name']} ({rec['category']})")

    # --- Handling cases with no history (new user or very sparse history) ---
    print("\n--- Example: New User / No History ---")
    new_user_id = "U003"
    personalized_recs_3 = recommender.get_personalized_recommendations(new_user_id, [], top_n=2)
    if not personalized_recs_3:
        print("No specific personalized recommendations for new user. Consider popular items or general category suggestions.")
