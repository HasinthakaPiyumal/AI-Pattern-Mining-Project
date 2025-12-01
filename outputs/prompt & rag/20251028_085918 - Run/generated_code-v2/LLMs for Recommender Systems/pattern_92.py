"""
ProductSense AI Recommender System

This system simulates an intelligent product recommendation engine for an e-commerce platform.
It leverages the conceptual idea of using Large Language Models (LLMs) as content interpreters
to understand product descriptions and user queries deeply. Due to the constraint of using
only built-in Python libraries, the LLM embedding generation and similarity search components
are simulated using simplified functions.
"""

import math

class ProductSenseAIRecommender:
    def __init__(self, embedding_dim=768):
        """
        Initializes the ProductSense AI Recommender.
        
        Args:
            embedding_dim (int): The dimension of the simulated embeddings.
        """
        self.products = []
        self.product_embeddings = {}
        self.embedding_dim = embedding_dim
        self.product_id_counter = 0

    def _generate_simulated_embedding(self, text):
        """
        Simulates an LLM generating a continuous vector embedding for text.
        In a real scenario, this would involve a fine-tuned LLM like BERT or GPT.
        
        For this simulation, we generate a simple hash-based vector scaled to a unit vector.
        """
        # Create a deterministic, but unique-ish vector based on text content
        hash_val = hash(text) % (10**9)
        embedding = [(hash_val + i) % 1000 / 1000.0 for i in range(self.embedding_dim)]
        
        # Normalize to a unit vector (for cosine similarity)
        magnitude = math.sqrt(sum(x*x for x in embedding))
        if magnitude == 0:
            return [0.0] * self.embedding_dim
        return [x / magnitude for x in embedding]

    def add_product(self, name, description, category, features=None):
        """
        Adds a new product to the system and generates its embedding.
        
        Args:
            name (str): The name of the product.
            description (str): A detailed description of the product.
            category (str): The product category.
            features (list, optional): A list of key features. Defaults to None.
        
        Returns:
            dict: The added product with its assigned ID.
        """
        product_id = f"prod_{self.product_id_counter}"
        self.product_id_counter += 1
        
        product_text = f"{name} {description} {category} {' '.join(features) if features else ''}"
        embedding = self._generate_simulated_embedding(product_text)
        
        product = {
            "id": product_id,
            "name": name,
            "description": description,
            "category": category,
            "features": features if features else []
        }
        self.products.append(product)
        self.product_embeddings[product_id] = embedding
        print(f"Added product '{name}' with ID '{product_id}'.")
        return product

    def _cosine_similarity(self, vec1, vec2):
        """
        Calculates the cosine similarity between two vectors.
        """
        dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
        # Since embeddings are normalized unit vectors, magnitudes are 1, so denominator is 1
        return dot_product

    def recommend_products(self, query=None, product_id=None, top_k=5):
        """
        Recommends products based on a query or a similar product ID.
        
        Args:
            query (str, optional): A natural language query describing desired products.
            product_id (str, optional): The ID of a product to find similar items for.
            top_k (int): The number of top recommendations to return.
            
        Returns:
            list: A list of dictionaries for recommended products, sorted by similarity.
        
        Raises:
            ValueError: If neither query nor product_id is provided, or if product_id is invalid.
        """
        if not query and not product_id:
            raise ValueError("Either 'query' or 'product_id' must be provided.")

        query_embedding = None
        if query:
            print(f"Generating embedding for query: '{query}'")
            query_embedding = self._generate_simulated_embedding(query)
        elif product_id:
            if product_id not in self.product_embeddings:
                raise ValueError(f"Product ID '{product_id}' not found.")
            print(f"Using embedding for product ID: '{product_id}'")
            query_embedding = self.product_embeddings[product_id]
        
        similarities = []
        for prod in self.products:
            if product_id and prod["id"] == product_id:
                continue # Don't recommend the product itself if looking for similar items
            
            prod_embedding = self.product_embeddings[prod["id"]]
            similarity = self._cosine_similarity(query_embedding, prod_embedding)
            similarities.append((similarity, prod))
        
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        print(f"Found {len(similarities)} similar products.")
        return [prod for sim, prod in similarities[:top_k]]

    def generate_explanation(self, product_details):
        """
        Simulates an LLM generating a human-readable explanation for a recommendation.
        In a real system, an LLM would analyze why a product was recommended based on
        query, user history, and product features.
        
        Args:
            product_details (dict): The dictionary containing details of the recommended product.
            
        Returns:
            str: A natural language explanation for the recommendation.
        """
        name = product_details.get("name", "this product")
        description = product_details.get("description", "")
        category = product_details.get("category", "item")
        features = product_details.get("features", [])
        
        explanation = f"This recommendation for '{name}' ({category}) is based on its deep semantic similarity to your interests. "
        explanation += f"Its description highlights: '{description[:70]}...' "
        if features:
            explanation += f"Key attributes like {', '.join(features[:2])} make it a strong match. "
        explanation += "Our AI understands the nuanced details to find the perfect fit for you."
        return explanation


# --- Example Usage ---
if __name__ == "__main__":
    print("Initializing ProductSense AI Recommender...")
    recommender = ProductSenseAIRecommender()

    # 1. Add some sample products
    print("\nAdding sample products...")
    product1 = recommender.add_product(
        name="Smartwatch Pro",
        description="Advanced smartwatch with health tracking, GPS, and long battery life.",
        category="Electronics",
        features=["GPS", "Heart Rate Monitor", "Waterproof", "14-day battery"]
    )
    product2 = recommender.add_product(
        name="Wireless Earbuds X",
        description="High-fidelity wireless earbuds with noise cancellation and ergonomic design.",
        category="Electronics",
        features=["Noise Cancellation", "Bluetooth 5.2", "24-hour playback"]
    )
    product3 = recommender.add_product(
        name="Ergonomic Office Chair",
        description="Comfortable office chair with lumbar support and adjustable armrests for long hours.",
        category="Home & Office",
        features=["Lumbar Support", "Adjustable Height", "Breathable Mesh"]
    )
    product4 = recommender.add_product(
        name="Portable Bluetooth Speaker",
        description="Compact speaker with powerful bass, 10-hour battery, and robust design.",
        category="Electronics",
        features=["Portable", "Water Resistant", "Stereo Sound"]
    )
    product5 = recommender.add_product(
        name="Noise-Cancelling Headphones",
        description="Premium over-ear headphones with industry-leading noise cancellation and superb audio.",
        category="Electronics",
        features=["Active Noise Cancellation", "Hi-Res Audio", "Comfort Fit"]
    )
    product6 = recommender.add_product(
        name="Organic Green Tea Kit",
        description="A selection of premium organic green teas for a healthy lifestyle.",
        category="Groceries",
        features=["Organic", "Antioxidant-rich", "Variety Pack"]
    )

    # 2. Get recommendations based on a text query (cold-start scenario)
    print("\n--- Query-based Recommendation (Cold Start) ---")
    query = "I'm looking for a wearable device that tracks fitness and has GPS."
    recommended_by_query = recommender.recommend_products(query=query, top_k=2)
    print(f"Recommendations for '{query}':")
    for i, product in enumerate(recommended_by_query):
        print(f"  {i+1}. {product['name']} (ID: {product['id']})")
        print(f"     Explanation: {recommender.generate_explanation(product)}")

    # 3. Get recommendations similar to an existing product
    print("\n--- Product-to-Product Recommendation ---")
    target_product_id = product2["id"] # Wireless Earbuds X
    print(f"Finding products similar to '{product2['name']}' (ID: {target_product_id})...")
    recommended_similar = recommender.recommend_products(product_id=target_product_id, top_k=3)
    for i, product in enumerate(recommended_similar):
        print(f"  {i+1}. {product['name']} (ID: {product['id']})")
        print(f"     Explanation: {recommender.generate_explanation(product)}")

    # 4. Demonstrate cross-domain (though simulated, the embeddings conceptually allow it)
    print("\n--- Cross-Domain Recommendation (Conceptual) ---")
    query_cross_domain = "Healthy beverages or food for an active person."
    recommended_cross = recommender.recommend_products(query=query_cross_domain, top_k=2)
    print(f"Recommendations for '{query_cross_domain}':")
    for i, product in enumerate(recommended_cross):
        print(f"  {i+1}. {product['name']} (ID: {product['id']})")
        print(f"     Explanation: {recommender.generate_explanation(product)}")

    print("\nProductSense AI Recommender System demonstration complete.")
