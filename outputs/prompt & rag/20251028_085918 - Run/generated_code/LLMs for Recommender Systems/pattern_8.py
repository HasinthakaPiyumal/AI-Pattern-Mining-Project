import random

class LLMRecommendationSystem:
    def __init__(self, product_data):
        self.product_data = product_data  # A dictionary of product IDs to product details
        # In a real system, you'd initialize LLM clients, load embeddings, etc.
        print("LLMRecommendationSystem initialized with product data.")

    def _call_llm(self, prompt, task_type="general"):
        """
        Placeholder for calling an actual Large Language Model (LLM) API.
        In a real application, this would interact with services like OpenAI's GPT, Google's Gemini,
        or a self-hosted model using libraries like 'transformers' or 'langchain'.
        """
        print(f"\n--- Simulated LLM Call for {task_type} ---")
        print(f"Prompt (truncated): {prompt[:200]}...") # Print truncated prompt for brevity

        # Simulate different LLM responses based on task_type
        if "recommendation" in task_type.lower():
            # Simulate selecting random products for demonstration
            recommended_ids = random.sample(list(self.product_data.keys()), min(3, len(self.product_data)))
            products_info = [self.product_data[pid]['name'] for pid in recommended_ids]
            return f"[LLM_RESPONSE_RECOMMENDATION] Based on your input, I suggest the following products: {', '.join(products_info)}. Consider their features for your needs."
        elif "explanation" in task_type.lower():
            return "[LLM_RESPONSE_EXPLANATION] This product was recommended because it aligns with your stated preferences for durability and eco-friendly options, a common theme in your past interactions."
        elif "customer support" in task_type.lower():
            if "shipping" in prompt.lower():
                return "[LLM_RESPONSE_CUSTOMER_SUPPORT] Standard shipping within the US typically takes 3-5 business days. Expedited options are available at checkout."
            elif "return policy" in prompt.lower():
                return "[LLM_RESPONSE_CUSTOMER_SUPPORT] Our return policy allows for full refunds on unused items within 30 days of purchase. Please see our website for more details."
            elif any(word in prompt.lower() for word in [product['name'].lower() for product in self.product_data.values()]):
                # If a product name is mentioned, try to give a generic product info
                return "[LLM_RESPONSE_CUSTOMER_SUPPORT] That product is highly rated for its features and user satisfaction. Can I provide more specific details or compare it with another item?"
            else:
                return "[LLM_RESPONSE_CUSTOMER_SUPPORT] I'm here to help with product-related questions or recommendations. Can you tell me more about what you're looking for?"
        return "[LLM_RESPONSE_GENERIC] Simulated LLM response for an unspecified task."

    def get_recommendations(self, user_id, user_preferences, historical_interactions):
        """
        Generates personalized product recommendations by leveraging LLM's deep semantic understanding.
        """
        prompt = (
            f"Given User ID: {user_id}, User Preferences: {user_preferences}, and "
            f"Historical Interactions: {historical_interactions}, recommend 3 highly personalized "
            f"e-commerce products from the current product catalog: {list(self.product_data.keys())}. "
            "Provide a brief rationale for each recommendation based on the user's profile."
        )
        llm_response = self._call_llm(prompt, task_type="recommendation")
        # In a real system, you would parse `llm_response` to extract structured recommendations
        # For this demo, we'll return a placeholder list of product IDs and the raw LLM response.
        recommended_product_ids = random.sample(list(self.product_data.keys()), min(3, len(self.product_data)))
        return recommended_product_ids, llm_response

    def generate_explanation(self, recommended_product_id, user_preferences):
        """
        Generates a human-centric explanation for a given recommendation using an LLM.
        """
        if recommended_product_id not in self.product_data:
            return f"Explanation: Product '{recommended_product_id}' not found."

        product_details = self.product_data[recommended_product_id]
        prompt = (
            f"Explain why product '{product_details['name']}' (Details: {product_details}) "
            f"was recommended to a user with preferences: {user_preferences}. "
            "Focus on key features and how they align with user's likely needs/interests."
        )
        explanation = self._call_llm(prompt, task_type="explanation")
        return explanation

    def handle_customer_query(self, query, user_context=None):
        """
        Uses an LLM to answer customer support queries related to product discovery or general information.
        """
        prompt = (
            f"A customer has asked: '{query}'. Provide a concise and helpful response. "
            f"If relevant, consider the current product catalog: {list(self.product_data.keys())}. "
            f"User context: {user_context if user_context else 'None'}."
        )
        answer = self._call_llm(prompt, task_type="customer support")
        return answer

# --- Example Usage ---
if __name__ == "__main__":
    # Sample Product Data (representing an e-commerce catalog)
    sample_products = {
        "P001": {"name": "Premium Wireless Earbuds", "category": "Electronics", "price": 129.99, "features": "Noise-cancelling, Long battery life, Bluetooth 5.2", "description": "Immersive audio experience with active noise cancellation."},
        "P002": {"name": "Ergonomic Office Chair", "category": "Home & Office", "price": 249.00, "features": "Adjustable lumbar support, Breathable mesh, Swivel function", "description": "Designed for maximum comfort during long working hours."},
        "P003": {"name": "Organic Green Tea Blend", "category": "Food & Beverage", "price": 15.50, "features": "Loose leaf, Antioxidant-rich, Sustainably sourced", "description": "A soothing and healthy tea blend from organic farms."},
        "P004": {"name": "Smart Home Lighting Kit", "category": "Electronics", "price": 79.99, "features": "Voice control, Dimmable, RGB colors, Wi-Fi enabled", "description": "Transform your home's ambiance with intelligent lighting."},
        "P005": {"name": "Durable Hiking Backpack", "category": "Sports & Outdoors", "price": 89.00, "features": "Water-resistant, Multiple compartments, Padded straps", "description": "Perfect companion for your outdoor adventures."}
    }

    # Initialize the LLM-enhanced recommendation system
    recommender_system = LLMRecommendationSystem(sample_products)

    # Simulate a user's profile and interactions
    user_id = "user_jane_doe"
    user_preferences = {"interests": ["tech", "comfort", "healthy living"], "budget": "medium-high", "recent_views": ["P001", "P002"]}
    historical_interactions = ["bought P001", "viewed P002 extensively", "added P004 to cart"]

    # --- 1. Get Recommendations ---
    print("\n===== GENERATING RECOMMENDATIONS =====")
    recommended_product_ids, llm_rec_rationale = recommender_system.get_recommendations(user_id, user_preferences, historical_interactions)
    print(f"\nRecommendations for '{user_id}':")
    for pid in recommended_product_ids:
        print(f"  - {pid}: {sample_products[pid]['name']}")
    print(f"LLM's internal rationale summary: {llm_rec_rationale.split('[LLM_RESPONSE_RECOMMENDATION] ')[-1].strip()}")

    # --- 2. Generate Explanation for a Recommendation ---
    print("\n===== GENERATING EXPLANATION =====")
    if recommended_product_ids:
        product_to_explain = recommended_product_ids[0] # Pick the first recommended product
        explanation = recommender_system.generate_explanation(product_to_explain, user_preferences)
        print(f"\nExplanation for recommending '{sample_products[product_to_explain]['name']}':")
        print(explanation.split('[LLM_RESPONSE_EXPLANATION] ')[-1].strip())
    else:
        print("No recommendations to explain.")

    # --- 3. Handle Customer Support Queries ---
    print("\n===== HANDLING CUSTOMER SUPPORT QUERIES =====")
    customer_queries = [
        "What is the shipping policy for the Smart Home Lighting Kit?",
        "Can you tell me more about the Ergonomic Office Chair?",
        "What's your return policy for electronics?",
        "I'm looking for a gift for someone who loves the outdoors."
    ]

    for query in customer_queries:
        print(f"\nCustomer: '{query}'")
        answer = recommender_system.handle_customer_query(query, user_context=user_preferences)
        print(f"System: {answer.split('[LLM_RESPONSE_CUSTOMER_SUPPORT] ')[-1].strip()}")