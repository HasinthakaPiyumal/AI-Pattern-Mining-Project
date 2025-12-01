#!/usr/bin/env python3

import os

class ProductCatalog:
    """
    Manages a catalog of fashion products. Provides methods to retrieve all products
    and filter them based on query and category.
    """
    def __init__(self):
        self.products = [
            {"id": "P001", "name": "Blue Denim Jeans", "category": "Bottoms", "style": "Casual", "color": "Blue", "price": 49.99},
            {"id": "P002", "name": "White Cotton T-Shirt", "category": "Tops", "style": "Casual", "color": "White", "price": 19.99},
            {"id": "P003", "name": "Black Leather Jacket", "category": "Outerwear", "style": "Edgy", "color": "Black", "price": 199.99},
            {"id": "P004", "name": "Floral Summer Dress", "category": "Dresses", "style": "Bohemian", "color": "Multi", "price": 79.99},
            {"id": "P005", "name": "Grey Hoodie", "category": "Tops", "style": "Sporty", "color": "Grey", "price": 39.99},
            {"id": "P006", "name": "Slim Fit Chinos", "category": "Bottoms", "style": "Smart Casual", "color": "Khaki", "price": 59.99},
            {"id": "P007", "name": "Striped Button-Up Shirt", "category": "Tops", "style": "Formal", "color": "Blue/White", "price": 65.00},
            {"id": "P008", "name": "High-Waisted Skirt", "category": "Bottoms", "style": "Elegant", "color": "Black", "price": 45.00},
            {"id": "P009", "name": "Running Sneakers", "category": "Footwear", "style": "Sporty", "color": "White", "price": 89.99},
            {"id": "P010", "name": "Classic Loafers", "category": "Footwear", "style": "Business Casual", "color": "Brown", "price": 120.00},
        ]

    def get_all_products(self) -> list:
        """Returns the entire list of products."""
        return self.products

    def filter_products(self, query: str, category: str = None) -> list:
        """
        Filters products based on a query string and an optional category.
        A real application might use vector search or a more sophisticated filtering.
        """
        filtered = []
        query_lower = query.lower()
        for product in self.products:
            match_query = query_lower in product["name"].lower() or \
                          query_lower in product["category"].lower() or \
                          query_lower in product["style"].lower() or \
                          query_lower in product["color"].lower()
            match_category = True
            if category:
                match_category = product["category"].lower() == category.lower()

            if match_query and match_category:
                filtered.append(product)
        return filtered

class LLMRecommender:
    """
    Simulates an LLM-based recommender using in-context learning principles
    (zero-shot and Chain-of-Thought).
    In a real scenario, this would integrate with an actual LLM API like OpenAI's.
    """
    def __init__(self, api_key: str = "YOUR_OPENAI_API_KEY"):
        # In a real application, you'd initialize your LLM client here, e.g.:
        # from openai import OpenAI
        # self.client = OpenAI(api_key=api_key)
        self.api_key = api_key
        print("LLMRecommender initialized. Using a simulated LLM for responses.")

    def _format_candidates(self, candidates: list) -> str:
        """Formats a list of product dictionaries into a human-readable string for the LLM prompt."""
        if not candidates:
            return "No specific items available as candidates."
        formatted_items = []
        for item in candidates:
            formatted_items.append(
                f"- ID: {item.get('id', 'N/A')}, Name: {item.get('name', 'N/A')}, Category: {item.get('category', 'N/A')}, Style: {item.get('style', 'N/A')}, Price: ${item.get('price', 'N/A')}"
            )
        return "\n".join(formatted_items)

    def _simulate_llm_response(self, prompt: str) -> str:
        """
        Simulates an LLM's response based on keywords in the prompt.
        Replace this with actual LLM API calls in a production environment.
        """
        print(f"\n--- Simulated LLM Input Prompt ---\n{prompt}\n----------------------------------")
        if "casual summer outfit" in prompt.lower():
            return "Recommendation for a casual summer outfit:\n1. White Cotton T-Shirt (ID: P002)\n2. Blue Denim Jeans (ID: P001)\nThese are comfortable and versatile for summer." 
        elif "smart casual work attire" in prompt.lower() and "let's think step by step" in prompt.lower():
            return (
                "Let's break down the smart casual work attire recommendation:\n"
                "Step 1: Understand user's need: smart casual for work.\n"
                "Step 2: Review candidates for categories like 'Bottoms' (smart casual style), 'Tops' (formal/smart casual), 'Footwear' (business casual).\n"
                "Step 3: From candidates, 'Slim Fit Chinos' (P006) and 'Striped Button-Up Shirt' (P007) match the attire. 'Classic Loafers' (P010) complete it.\n"
                "Final Recommendation:\n"
                "1. Slim Fit Chinos (ID: P006)\n"
                "2. Striped Button-Up Shirt (ID: P007)\n"
                "3. Classic Loafers (ID: P010)"
            )
        elif "elegant evening event" in prompt.lower():
            return "For an elegant evening event, consider the High-Waisted Skirt (ID: P008). It's versatile and can be dressed up with a formal top (not in current catalog)." 
        return "General Recommendation: Blue Denim Jeans (ID: P001) and Floral Summer Dress (ID: P004)."

    def generate_zero_shot_recommendation(self, user_preference: str, candidates: list) -> str:
        """
        Generates a zero-shot recommendation by directly asking the LLM for a product
        based on user preference and a list of candidates.
        """
        formatted_candidates = self._format_candidates(candidates)
        prompt = (
            f"You are a fashion recommender. Recommend a product from the list below based on the user's preference.\n\n"
            f"User Preference: {user_preference}\n\n"
            f"Available Products:\n{formatted_candidates}\n\n"
            f"Your Recommendation (mentioning Product ID and Name):"
        )
        # In a real scenario, this would be an actual API call:
        # response = self.client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}])
        # return response.choices[0].message.content
        return self._simulate_llm_response(prompt)

    def generate_cot_recommendation(self, user_preference: str, candidates: list) -> str:
        """
        Generates a Chain-of-Thought (CoT) recommendation, guiding the LLM to
        reason through multiple steps before providing a final recommendation.
        """
        formatted_candidates = self._format_candidates(candidates)
        prompt = (
            f"You are a fashion recommender. I need you to provide a recommendation by thinking step-by-step. "
            f"First, analyze the user's preference and identify key requirements (style, occasion, items).\n"
            f"Second, filter the available products to find the best matches based on these requirements.\n"
            f"Third, justify your choice based on the style, category, and price of the recommended items.\n"
            f"Finally, present the top recommended products.\n\n"
            f"User Preference: {user_preference}\n\n"
            f"Available Products:\n{formatted_candidates}\n\n"
            f"Let's think step by step:" # This phrase triggers CoT reasoning in many LLMs
        )
        # In a real scenario, this would be an actual API call:
        # response = self.client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}])
        # return response.choices[0].message.content
        return self._simulate_llm_response(prompt)

# Main execution logic
def main():
    # In a real application, you would load your OpenAI API key from environment variables
    # For this simulated example, the API key is a placeholder and not strictly used by _simulate_llm_response.
    openai_api_key = os.getenv("OPENAI_API_KEY", "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx") 

    catalog = ProductCatalog()
    recommender = LLMRecommender(api_key=openai_api_key)

    print("\n================================================")
    print(" Welcome to the Personalized Fashion Recommender!")
    print("================================================\n")

    # --- Scenario 1: Zero-shot recommendation for a casual outfit ---
    user_pref_1 = "I need a comfortable and casual summer outfit."
    print(f"--- User Preference 1: {user_pref_1} ---")
    
    # Candidate Generation: Filter initial pool of items
    casual_candidates = catalog.filter_products(query="casual", category="Tops") + \
                        catalog.filter_products(query="casual", category="Bottoms")
    print(f"[Candidate Generation] Found {len(casual_candidates)} relevant candidates: {[p['name'] for p in casual_candidates]}\n")

    zero_shot_rec = recommender.generate_zero_shot_recommendation(user_pref_1, casual_candidates)
    print(f"\n[Zero-shot Recommendation]:\n{zero_shot_rec}\n")

    # --- Scenario 2: Chain-of-Thought recommendation for smart casual work attire ---
    user_pref_2 = "I'm looking for a smart casual outfit suitable for work."
    print(f"--- User Preference 2: {user_pref_2} ---")
    
    # Candidate Generation: Filter initial pool of items for smart casual
    smart_casual_candidates = catalog.filter_products(query="smart casual") + \
                              catalog.filter_products(query="formal", category="Tops") + \
                              catalog.filter_products(query="business casual", category="Footwear")
    print(f"[Candidate Generation] Found {len(smart_casual_candidates)} relevant candidates: {[p['name'] for p in smart_casual_candidates]}\n")

    cot_rec = recommender.generate_cot_recommendation(user_pref_2, smart_casual_candidates)
    print(f"\n[Chain-of-Thought Recommendation]:\n{cot_rec}\n")

    # --- Scenario 3: Zero-shot recommendation for an evening outfit ---
    user_pref_3 = "I need something elegant for an evening event."
    print(f"--- User Preference 3: {user_pref_3} ---")

    # Candidate Generation (can be broader if the user doesn't specify much initially)
    evening_candidates = catalog.filter_products(query="elegant") + \
                         catalog.filter_products(query="dress") + \
                         catalog.filter_products(query="skirt")
    print(f"[Candidate Generation] Found {len(evening_candidates)} relevant candidates: {[p['name'] for p in evening_candidates]}\n")

    zero_shot_rec_3 = recommender.generate_zero_shot_recommendation(user_pref_3, evening_candidates)
    print(f"\n[Zero-shot Recommendation]:\n{zero_shot_rec_3}\n")


if __name__ == "__main__":
    main()
