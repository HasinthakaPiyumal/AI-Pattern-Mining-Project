
import random
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class LLMService:
    """Simulated LLM Service for embeddings, explanations, and query understanding."""
    def __init__(self):
        self.embedding_dim = 768 # Standard embedding dimension like for Sentence-BERT

    def get_embedding(self, text: str) -> list[float]:
        """Simulates getting a dense vector embedding for a given text."""
        # In a real scenario, this would call a transformers model (e.g., Sentence-BERT)
        # For simulation, return a random vector.
        return [random.uniform(-1, 1) for _ in range(self.embedding_dim)]

    def generate_explanation(self, product_details: dict, user_context: dict, reason_type: str) -> str:
        """Simulates generating a human-centric explanation for a recommendation."""
        product_name = product_details.get("name", "this product")
        product_category = product_details.get("category", "a general category")
        user_preference = user_context.get("preference", "your recent interests")

        if reason_type == "similarity":
            return f"Based on {user_preference}, we think you'll like {product_name} because it's similar to other {product_category} items you've shown interest in."
        elif reason_type == "new_arrival":
            return f"{product_name} is a new arrival in {product_category} that matches {user_preference}."
        elif reason_type == "popular":
            return f"Many users like you are interested in {product_name} from {product_category}. It's a popular choice!"
        else:
            return f"We recommend {product_name} based on its relevance to your interests and our catalog."

    def understand_query(self, natural_language_query: str) -> dict:
        """Simulates parsing a natural language query to extract intent and filters."""
        query_lower = natural_language_query.lower()
        intent = "recommendation"
        filters = {}

        if "category" in query_lower:
            for category in ["electronics", "books", "clothing", "home & kitchen"]:
                if category in query_lower:
                    filters["category"] = category
                    break
        if "price range" in query_lower or "expensive" in query_lower or "cheap" in query_lower:
            # Simple price range simulation
            if "cheap" in query_lower: filters["price_max"] = 50
            elif "expensive" in query_lower: filters["price_min"] = 100

        return {"intent": intent, "filters": filters}

class VectorDB:
    """Simulated Vector Database for storing and searching item embeddings."""
    def __init__(self):
        self.embeddings = {}
        self.item_ids = []
        self.vectors = []

    def add_item(self, item_id: str, embedding: list[float]):
        """Adds an item's ID and its embedding to the database."""
        self.embeddings[item_id] = embedding
        self.item_ids = list(self.embeddings.keys())
        self.vectors = list(self.embeddings.values())

    def search_similar(self, query_embedding: list[float], top_n: int = 5) -> list[str]:
        """Searches for the top_n most similar item IDs to the query embedding."""
        if not self.vectors:
            return []

        query_vec = np.array(query_embedding).reshape(1, -1)
        all_vectors = np.array(self.vectors)

        similarities = cosine_similarity(query_vec, all_vectors)[0]

        # Get indices of top_n most similar items
        top_indices = similarities.argsort()[-top_n:][::-1]

        # Map indices back to item IDs
        return [self.item_ids[i] for i in top_indices]


class RecommenderEngine:
    """Intelligent E-commerce Recommender System with LLM enhancements."""
    def __init__(self):
        self.llm_service = LLMService()
        self.vector_db = VectorDB()

        # --- Simulated Data Layer ---
        self.product_catalog = {
            "prod101": {"id": "prod101", "name": "Smartwatch X", "description": "Advanced smartwatch with health tracking.", "category": "electronics", "price": 199.99, "reviews": "Great battery life!"},
            "prod102": {"id": "prod102", "name": "Novel by A.B.C.", "description": "Bestselling fiction novel, a thrilling read.", "category": "books", "price": 15.00, "reviews": "Couldn't put it down!"},
            "prod103": {"id": "prod103", "name": "Bluetooth Headphones", "description": "High-quality wireless headphones with noise cancellation.", "category": "electronics", "price": 79.99, "reviews": "Excellent sound."},
            "prod104": {"id": "prod104", "name": "Cookbook: Italian Delights", "description": "Authentic Italian recipes for home cooks.", "category": "books", "price": 25.50, "reviews": "Easy to follow recipes."},
            "prod105": {"id": "prod105", "name": "Ergonomic Office Chair", "description": "Comfortable chair for long working hours.", "category": "home & kitchen", "price": 250.00, "reviews": "Worth every penny!"},
            "prod106": {"id": "prod106", "name": "Stylish T-Shirt", "description": "100% cotton casual t-shirt, various sizes.", "category": "clothing", "price": 29.99, "reviews": "Comfortable and fits well."},
        }

        self.user_interaction_log = {
            "user1": {"views": ["prod101", "prod103"], "purchases": ["prod101"], "searches": ["smartwatch", "headphones"]},
            "user2": {"views": ["prod102", "prod104"], "purchases": ["prod102"], "searches": ["fiction books", "cookbooks"]},
            "user3": {"views": ["prod105"], "purchases": [], "searches": ["office chair"]},
        }

        self._preprocess_data()

    def _preprocess_data(self):
        """Generates and stores embeddings for all products in the catalog."""
        print("\n--- Preprocessing Product Data ---")
        for product_id, details in self.product_catalog.items():
            text_to_embed = f"{details['name']}. {details['description']}. {details.get('reviews', '')}"
            embedding = self.llm_service.get_embedding(text_to_embed)
            self.vector_db.add_item(product_id, embedding)
            print(f"Generated embedding for {details['name']}")
        print("--- Data Preprocessing Complete ---")

    def _create_user_profile_embedding(self, user_id: str) -> list[float]:
        """Creates a user's interest embedding based on their interaction history."""
        interactions = self.user_interaction_log.get(user_id, {})
        combined_text = []

        for product_id in interactions.get("views", []) + interactions.get("purchases", []):
            if product_id in self.product_catalog:
                details = self.product_catalog[product_id]
                combined_text.append(f"{details['name']}. {details['description']}")
        
        for search_query in interactions.get("searches", []):
            combined_text.append(search_query)

        if not combined_text:
            # Default embedding if no history
            return self.llm_service.get_embedding("general interests")

        full_text = ". ".join(combined_text)
        return self.llm_service.get_embedding(full_text)

    def _generate_recommendation_explanation(self, product_id: str, user_id: str) -> str:
        """Generates a human-centric explanation for a specific recommendation."""
        product_details = self.product_catalog.get(product_id, {})
        user_context = {
            "preference": f"your recent interactions as user {user_id}"
        }
        # In a more advanced system, 'reason_type' would be determined by the recommendation algorithm itself
        return self.llm_service.generate_explanation(product_details, user_context, "similarity")

    def get_recommendations(self, user_id: str, natural_language_query: str = None, top_n: int = 3) -> list[dict]:
        """Generates personalized recommendations for a user, with optional NL refinement."""
        print(f"\n--- Getting Recommendations for User {user_id} ---")
        query_embedding = self._create_user_profile_embedding(user_id)
        filters = {}

        if natural_language_query:
            print(f"User provided natural language query: '{natural_language_query}'")
            parsed_query = self.llm_service.understand_query(natural_language_query)
            filters = parsed_query.get("filters", {})
            print(f"Parsed query filters: {filters}")
            # In a real system, the query_embedding might be adjusted based on intent
            # For this simulation, we'll primarily use filters.

        # Get similar items from the vector database
        recommended_product_ids = self.vector_db.search_similar(query_embedding, top_n=top_n * 2) # Get more to filter
        
        final_recommendations = []
        interacted_products = self.user_interaction_log.get(user_id, {}).get("views", []) + \
                              self.user_interaction_log.get(user_id, {}).get("purchases", [])
        interacted_products = set(interacted_products)

        for prod_id in recommended_product_ids:
            if prod_id in self.product_catalog:
                product_details = self.product_catalog[prod_id]
                
                # Apply filters
                if filters.get("category") and product_details.get("category") != filters["category"]:
                    continue
                if filters.get("price_min") and product_details.get("price", 0) < filters["price_min"]:
                    continue
                if filters.get("price_max") and product_details.get("price", float('inf')) > filters["price_max"]:
                    continue
                
                # Filter out already interacted products
                if prod_id not in interacted_products:
                    explanation = self._generate_recommendation_explanation(prod_id, user_id)
                    final_recommendations.append({"product": product_details, "explanation": explanation})

            if len(final_recommendations) >= top_n:
                break

        return final_recommendations

# --- Main Application Logic (Command-Line UI) ---
def run_recommender_app():
    engine = RecommenderEngine()

    print("\nWelcome to the LLM-Enhanced E-commerce Recommender System!")
    print("Available users: user1, user2, user3")
    
    current_user = None
    while current_user not in engine.user_interaction_log:
        user_input = input("Please enter your user ID: ").strip()
        if user_input in engine.user_interaction_log:
            current_user = user_input
        else:
            print("Invalid user ID. Please try again.")

    while True:
        print(f"\n--- Current User: {current_user} ---")
        print("1. Get Recommendations")
        print("2. Refine Recommendations with Natural Language")
        print("3. Change User")
        print("4. Exit")

        choice = input("Enter your choice: ").strip()

        if choice == "1":
            recommendations = engine.get_recommendations(current_user)
            if recommendations:
                print("\nYour personalized recommendations:")
                for rec in recommendations:
                    product = rec["product"]
                    print(f"  - {product['name']} ({product['category']}) - ${product['price']:.2f}")
                    print(f"    Explanation: {rec['explanation']}")
            else:
                print("No new recommendations found for you.")
        
        elif choice == "2":
            nl_query = input("Enter your refinement query (e.g., 'show me electronics', 'I want cheap books'): ").strip()
            recommendations = engine.get_recommendations(current_user, natural_language_query=nl_query)
            if recommendations:
                print("\nYour refined recommendations:")
                for rec in recommendations:
                    product = rec["product"]
                    print(f"  - {product['name']} ({product['category']}) - ${product['price']:.2f}")
                    print(f"    Explanation: {rec['explanation']}")
            else:
                print("No recommendations found matching your refinement.")

        elif choice == "3":
            new_user = None
            while new_user not in engine.user_interaction_log:
                user_input = input("Please enter the new user ID: ").strip()
                if user_input in engine.user_interaction_log:
                    new_user = user_input
                else:
                    print("Invalid user ID. Please try again.")
            current_user = new_user
        
        elif choice == "4":
            print("Exiting Recommender System. Goodbye!")
            break

        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    run_recommender_app()
