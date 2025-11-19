import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

# --- 1. Data Models ---

class Product:
    def __init__(self, id, name, description, category, features, image_url, price):
        self.id = id
        self.name = name
        self.description = description
        self.category = category
        self.features = features
        self.image_url = image_url
        self.price = price
        self.embedding = None # Will be populated by LLMEmbedder

    def __repr__(self):
        return f"Product(ID={self.id}, Name='{self.name}', Category='{self.category}', Price='${self.price:.2f}')"

class User:
    def __init__(self, id, name, preferences=None, interaction_history=None):
        self.id = id
        self.name = name
        self.preferences = preferences if preferences is not None else []
        self.interaction_history = interaction_history if interaction_history is not None else [] # List of product IDs

    def __repr__(self):
        return f"User(ID={self.id}, Name='{self.name}', Preferences={self.preferences})"

# --- 2. LLM Utility ---

class LLMEmbedder:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def get_embedding(self, text):
        return self.model.encode(text, convert_to_tensor=False)

class MockLLMGenerator:
    def generate_explanation(self, product_name, user_preferences, reason):
        pref_str = ", ".join(user_preferences)
        if "similar items" in reason:
            return f"Based on your interest in {pref_str}, we think you'll love the {product_name}. It's similar to items you've previously enjoyed."
        elif "features" in reason:
            return f"We recommend the {product_name} because its features align perfectly with what you look for, especially in {pref_str} categories."
        elif "category" in reason:
            return f"Since you've shown interest in {pref_str}, the {product_name} from the same category is a great match for you."
        return f"We recommend the {product_name} for you!"

    def parse_query(self, query):
        parsed_data = {"product_type": None, "color": None, "price_range": None, "gender": None, "brand": None}

        # Simple regex-based parsing (mocking LLM entity extraction)
        product_type_match = re.search(r'(shoes|shirt|laptop|book|watch|phone)', query, re.IGNORECASE)
        if product_type_match: parsed_data["product_type"] = product_type_match.group(1).lower()

        color_match = re.search(r'(red|blue|green|black|white)', query, re.IGNORECASE)
        if color_match: parsed_data["color"] = color_match.group(1).lower()

        price_match = re.search(r'under \$(\d+)|over \$(\d+)|\$(\d+)-\$(\d+)', query, re.IGNORECASE)
        if price_match:
            if price_match.group(1): parsed_data["price_range"] = ("max", float(price_match.group(1)))
            elif price_match.group(2): parsed_data["price_range"] = ("min", float(price_match.group(2)))
            elif price_match.group(3) and price_match.group(4): parsed_data["price_range"] = (float(price_match.group(3)), float(price_match.group(4)))

        gender_match = re.search(r'(men|women|unisex)', query, re.IGNORECASE)
        if gender_match: parsed_data["gender"] = gender_match.group(1).lower()

        brand_match = re.search(r'(nike|adidas|sony|apple|samsung)', query, re.IGNORECASE)
        if brand_match: parsed_data["brand"] = brand_match.group(1).lower()

        return parsed_data

    def generate_response(self, results, original_query, user_preferences=None):
        if not results:
            return f"I couldn't find any products matching your request for '{original_query}'. Please try a different query."
        
        response_lines = [f"Here are some recommendations based on your query '{original_query}':"]
        for product, explanation in results:
            response_lines.append(f"- {product.name} ({product.category}) - ${product.price:.2f}. {explanation}")
        return "\n".join(response_lines)

# --- 3. Product Catalog (In-Memory) ---

class ProductCatalog:
    def __init__(self, embedder):
        self.products = []
        self.product_id_map = {}
        self.embedder = embedder

    def add_product(self, product_data):
        product = Product(**product_data)
        text_to_embed = f"{product.name} {product.description} {product.category} {" ".join(product.features)}"
        product.embedding = self.embedder.get_embedding(text_to_embed)
        self.products.append(product)
        self.product_id_map[product.id] = product
        return product

    def get_product_by_id(self, product_id):
        return self.product_id_map.get(product_id)

    def get_all_products(self):
        return self.products

# --- 4. Recommender System ---

class Recommender:
    def __init__(self, catalog, llm_generator):
        self.catalog = catalog
        self.llm_generator = llm_generator

    def get_content_based_recommendations(self, query_embedding, user_preferences, exclude_product_ids=None, top_n=5):
        if exclude_product_ids is None:
            exclude_product_ids = []

        product_embeddings = []
        product_list = []
        for product in self.catalog.get_all_products():
            if product.id not in exclude_product_ids and product.embedding is not None:
                product_embeddings.append(product.embedding)
                product_list.append(product)

        if not product_embeddings:
            return []
        
        # Reshape for cosine_similarity: (n_samples, n_features)
        query_embedding_reshaped = query_embedding.reshape(1, -1)
        product_embeddings_array = np.array(product_embeddings)

        similarities = cosine_similarity(query_embedding_reshaped, product_embeddings_array)[0]

        # Sort by similarity and get top_n
        sorted_indices = np.argsort(similarities)[::-1]
        top_recommendations_with_scores = []
        for i in sorted_indices:
            if len(top_recommendations_with_scores) >= top_n:
                break
            product = product_list[i]
            explanation = self.llm_generator.generate_explanation(
                product.name, user_preferences, reason="features" # Simplified reason for demo
            )
            top_recommendations_with_scores.append((product, explanation, similarities[i]))

        return top_recommendations_with_scores

    def get_recommendations_for_user(self, user, top_n=5):
        if not user.interaction_history:
            # Fallback to general popular items or content-based on preferences if no history
            if user.preferences:
                # Create a pseudo-query embedding from user preferences
                pref_text = " ".join(user.preferences)
                query_embedding = self.catalog.embedder.get_embedding(pref_text)
                return self.get_content_based_recommendations(query_embedding, user.preferences, top_n=top_n)
            return [] # No history, no preferences, no recommendations

        # For simplicity, base recommendations on the last interacted product's embedding
        last_interacted_product_id = user.interaction_history[-1]
        last_product = self.catalog.get_product_by_id(last_interacted_product_id)
        if last_product and last_product.embedding is not None:
            return self.get_content_based_recommendations(
                last_product.embedding, user.preferences, exclude_product_ids=user.interaction_history, top_n=top_n
            )
        return []


# --- 5. Conversational Search Interface ---

class ConversationalAgent:
    def __init__(self, recommender, llm_generator, embedder, catalog):
        self.recommender = recommender
        self.llm_generator = llm_generator
        self.embedder = embedder
        self.catalog = catalog

    def handle_query(self, user_input, user=None):
        parsed_query_data = self.llm_generator.parse_query(user_input)
        user_preferences = user.preferences if user else []

        # Build a search query string for embedding based on parsed data
        search_terms = []
        if parsed_query_data["product_type"]: search_terms.append(parsed_query_data["product_type"])
        if parsed_query_data["color"]: search_terms.append(parsed_query_data["color"])
        if parsed_query_data["gender"]: search_terms.append(f"{parsed_query_data["gender"]} items")
        if parsed_query_data["brand"]: search_terms.append(parsed_query_data["brand"])
        # Add user preferences to search terms to personalize
        search_terms.extend(user_preferences)

        query_embedding_text = " ".join(search_terms) if search_terms else user_input
        query_embedding = self.embedder.get_embedding(query_embedding_text)

        # Filter products based on parsed data before recommending
        filtered_products_for_search = []
        for product in self.catalog.get_all_products():
            match = True
            if parsed_query_data["product_type"] and parsed_query_data["product_type"] not in product.category.lower():
                match = False
            if parsed_query_data["color"] and parsed_query_data["color"] not in product.description.lower() and parsed_query_data["color"] not in product.name.lower():
                match = False
            if parsed_query_data["gender"] and parsed_query_data["gender"] not in product.features:
                match = False
            if parsed_query_data["brand"] and parsed_query_data["brand"] not in product.name.lower():
                match = False
            if parsed_query_data["price_range"]:
                price_type, price_val = parsed_query_data["price_range"]
                if isinstance(price_type, str):
                    if price_type == "max" and product.price > price_val: match = False
                    if price_type == "min" and product.price < price_val: match = False
                else:
                    min_price, max_price = price_type, price_val
                    if not (min_price <= product.price <= max_price): match = False
            
            if match:
                filtered_products_for_search.append(product.id)
        
        # If no products match the hard filters, revert to searching all products
        if not filtered_products_for_search:
            print("No products match strict filters, performing broad search.")
            recommendations_with_scores = self.recommender.get_content_based_recommendations(
                query_embedding, user_preferences, top_n=5
            )
        else:
            # Get recommendations only from the filtered set by providing a dummy catalog or filtering based on product IDs
            # For simplicity, let's filter the results *after* getting recommendations from the main catalog
            # A more robust approach would involve passing the filtered product IDs to the recommender
            all_recs_with_scores = self.recommender.get_content_based_recommendations(
                query_embedding, user_preferences, top_n=20 # Get more to filter down
            )
            recommendations_with_scores = []
            for rec, exp, score in all_recs_with_scores:
                if rec.id in filtered_products_for_search:
                    recommendations_with_scores.append((rec, exp, score))
                if len(recommendations_with_scores) >= 5: # Limit to top 5 after filtering
                    break


        formatted_results = [
            (product, self.llm_generator.generate_explanation(product.name, user_preferences, reason="category"))
            for product, _, _ in recommendations_with_scores
        ]
        return self.llm_generator.generate_response(formatted_results, user_input, user_preferences)


# --- Main Execution Logic ---
if __name__ == "__main__":
    # Initialize components
    embedder = LLMEmbedder()
    llm_generator = MockLLMGenerator()
    catalog = ProductCatalog(embedder)
    recommender = Recommender(catalog, llm_generator)
    conversational_agent = ConversationalAgent(recommender, llm_generator, embedder, catalog)

    # Sample Product Data
    product_data = [
        {"id": "P001", "name": "Nike Air Max 270", "description": "Comfortable running shoes with large air unit.", "category": "Footwear", "features": ["running", "mens", "sport", "red"], "image_url": "url_p001", "price": 150.00},
        {"id": "P002", "name": "Adidas Ultraboost 21", "description": "High-performance running shoes for everyday training.", "category": "Footwear", "features": ["running", "mens", "sport", "black"], "image_url": "url_p002", "price": 180.00},
        {"id": "P003", "name": "Sony WH-1000XM4 Headphones", "description": "Industry-leading noise cancelling headphones.", "category": "Electronics", "features": ["audio", "wireless", "black"], "image_url": "url_p003", "price": 349.99},
        {"id": "P004", "name": "Apple MacBook Air M2", "description": "Thin and light laptop with powerful M2 chip.", "category": "Electronics", "features": ["laptop", "productivity", "silver"], "image_url": "url_p004", "price": 1199.00},
        {"id": "P005", "name": "The Great Gatsby Book", "description": "Classic American novel by F. Scott Fitzgerald.", "category": "Books", "features": ["fiction", "classic", "literature"], "image_url": "url_p005", "price": 12.99},
        {"id": "P006", "name": "Red Casual T-Shirt", "description": "Soft cotton t-shirt for everyday wear.", "category": "Apparel", "features": ["mens", "casual", "red", "cotton"], "image_url": "url_p006", "price": 25.00},
        {"id": "P007", "name": "Blue Denim Jeans", "description": "Durable blue denim jeans for men.", "category": "Apparel", "features": ["mens", "denim", "blue", "casual"], "image_url": "url_p007", "price": 60.00},
        {"id": "P008", "name": "Women's Yoga Mat", "description": "Non-slip yoga mat for fitness and exercise.", "category": "Fitness", "features": ["womens", "yoga", "exercise", "pink"], "image_url": "url_p008", "price": 35.00},
        {"id": "P009", "name": "Smart Watch Series 7", "description": "Advanced smartwatch with health tracking.", "category": "Wearables", "features": ["smartwatch", "fitness", "tech", "unisex"], "image_url": "url_p009", "price": 399.00}
    ]

    print("Loading products and generating embeddings...")
    for p_data in product_data:
        catalog.add_product(p_data)
    print(f"Loaded {len(catalog.get_all_products())} products.")

    # Sample User Data
    user1 = User("U001", "Alice", preferences=["electronics", "running", "black"], interaction_history=["P002", "P003"])
    user2 = User("U002", "Bob", preferences=["books", "fiction"], interaction_history=["P005"])

    print("\n--- User-based Recommendations (Alice) ---")
    alice_recs = recommender.get_recommendations_for_user(user1)
    if alice_recs:
        for product, explanation, score in alice_recs:
            print(f"Recommended: {product.name} (Score: {score:.2f}). Explanation: {explanation}")
    else:
        print("No recommendations for Alice yet.")

    print("\n--- Conversational Search (Bob) ---")
    print("Bob: Show me red running shoes under $100 for men.")
    response_bob_1 = conversational_agent.handle_query("Show me red running shoes under $100 for men.", user=user2)
    print(f"System: {response_bob_1}")

    print("\nBob: Find me a black laptop.")
    response_bob_2 = conversational_agent.handle_query("Find me a black laptop.", user=user2)
    print(f"System: {response_bob_2}")

    print("\n--- Conversational Search (Alice) ---")
    print("Alice: What headphones would you recommend for wireless audio?")
    response_alice_1 = conversational_agent.handle_query("What headphones would you recommend for wireless audio?", user=user1)
    print(f"System: {response_alice_1}")

    print("\nAlice: Show me more red running shoes.")
    response_alice_2 = conversational_agent.handle_query("Show me more red running shoes.", user=user1)
    print(f"System: {response_alice_2}")

    print("\n--- Conversational Search (General Query) ---")
    print("User: I'm looking for a classic novel.")
    response_general = conversational_agent.handle_query("I'm looking for a classic novel.")
    print(f"System: {response_general}")

    print("\nUser: Give me some product recommendations in the fitness category.")
    response_general_2 = conversational_agent.handle_query("Give me some product recommendations in the fitness category.")
    print(f"System: {response_general_2}")