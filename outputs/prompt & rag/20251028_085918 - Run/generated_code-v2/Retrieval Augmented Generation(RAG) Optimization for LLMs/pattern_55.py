from transformers import pipeline

class PersonalizedContentGenerator:
    def __init__(self):
        self.generator = pipeline("text-generation", model="distilgpt2")
        self.user_profiles = {
            "user_1": {
                "preferences": {"category": "electronics", "style": "innovative", "tone": "enthusiastic"},
                "browsing_history": ["smartphones", "laptops", "wearables"],
                "past_purchases": ["smartphone_X", "smartwatch_Y"]
            },
            "user_2": {
                "preferences": {"category": "home_decor", "style": "minimalist", "tone": "calm"},
                "browsing_history": ["lamps", "vases", "rugs"],
                "past_purchases": ["vase_A"]
            }
        }
        self.product_catalog = {
            "smartphone_X": {
                "name": "QuantumFlow Pro",
                "category": "electronics",
                "features": ["5G enabled", "OLED display", "AI camera", "long-lasting battery"],
                "price": 999.99,
                "description": "A cutting-edge smartphone with advanced features."
            },
            "smartwatch_Y": {
                "name": "ChronoHealth Watch",
                "category": "wearables",
                "features": ["heart rate monitor", "sleep tracker", "GPS", "waterproof"],
                "price": 199.99,
                "description": "Monitor your health and stay connected with this stylish smartwatch."
            },
            "vase_A": {
                "name": "Zen Serenity Vase",
                "category": "home_decor",
                "features": ["ceramic material", "minimalist design", "handmade"],
                "price": 49.99,
                "description": "A beautiful ceramic vase to enhance your home decor."
            }
        }

    def _fetch_user_profile(self, user_id):
        return self.user_profiles.get(user_id, {})

    def _fetch_product_details(self, product_id):
        return self.product_catalog.get(product_id, {})

    def _construct_prompt(self, user_profile, product_details, content_type):
        user_prefs = user_profile.get("preferences", {})
        browsing_history = ", ".join(user_profile.get("browsing_history", []))
        past_purchases = ", ".join(user_profile.get("past_purchases", []))

        product_name = product_details.get("name", "")
        product_category = product_details.get("category", "")
        product_features = ", ".join(product_details.get("features", []))
        product_description = product_details.get("description", "")
        product_price = product_details.get("price", "N/A")

        prompt = f"""
        Generate personalized {content_type} for a user with the following profile:
        - Preferences: {user_prefs.get("style", "standard")} style, {user_prefs.get("tone", "neutral")} tone, interested in {user_prefs.get("category", "various")}.
        - Browsing history includes: {browsing_history if browsing_history else 'nothing specific'}.
        - Past purchases include: {past_purchases if past_purchases else 'no prior purchases'}.

        Product details:
        - Name: {product_name}
        - Category: {product_category}
        - Features: {product_features}
        - Price: ${product_price}
        - General description: {product_description}

        Please generate a {content_type} that is highly engaging and tailored to the user's interests:
        """
        return prompt

    def generate_personalized_content(self, user_id, product_id, content_type="product description", max_length=150, num_return_sequences=1):
        user_profile = self._fetch_user_profile(user_id)
        product_details = self._fetch_product_details(product_id)

        if not user_profile or not product_details:
            return "Error: User or product not found."

        prompt = self._construct_prompt(user_profile, product_details, content_type)
        
        generated_text = self.generator(prompt, max_length=max_length, num_return_sequences=num_return_sequences, truncation=True, do_sample=True, top_k=50, top_p=0.95)[0]['generated_text']
        
        # Post-process to remove the prompt itself from the generated text
        # A more robust solution might involve specific stop tokens or generation parameters
        return generated_text.replace(prompt, "").strip()

if __name__ == "__main__":
    content_gen = PersonalizedContentGenerator()

    print("\n--- Generating Personalized Product Description for User 1 and Smartphone X ---")
    desc1 = content_gen.generate_personalized_content("user_1", "smartphone_X", "product description")
    print(desc1)

    print("\n--- Generating Personalized Ad Copy for User 1 and Smartwatch Y ---")
    ad_copy1 = content_gen.generate_personalized_content("user_1", "smartwatch_Y", "marketing ad copy")
    print(ad_copy1)

    print("\n--- Generating Personalized Product Description for User 2 and Vase A ---")
    desc2 = content_gen.generate_personalized_content("user_2", "vase_A", "product description")
    print(desc2)

    print("\n--- Generating Personalized Ad Copy for User 2 and Smartphone X (despite user preference) ---")
    ad_copy2 = content_gen.generate_personalized_content("user_2", "smartphone_X", "marketing ad copy")
    print(ad_copy2)
