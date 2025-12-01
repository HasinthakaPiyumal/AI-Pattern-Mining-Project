class ProductDatabase:
    def __init__(self):
        self.products = {
            "P001": {"id": "P001", "name": "Wireless Bluetooth Headphones", "category": "Electronics", "description": "High-quality sound with noise cancellation.", "price": 79.99},
            "P002": {"id": "P002", "name": "Ergonomic Office Chair", "category": "Home & Office", "description": "Adjustable chair for maximum comfort.", "price": 199.99},
            "P003": {"id": "P003", "name": "Stainless Steel Water Bottle", "category": "Sports & Outdoors", "description": "Keeps drinks cold for 24 hours.", "price": 24.99},
            "P004": {"id": "P004", "name": "Smart LED Desk Lamp", "category": "Electronics", "description": "Dimmable with color temperature control.", "price": 49.99},
            "P005": {"id": "P005", "name": "Yoga Mat Deluxe", "category": "Sports & Outdoors", "description": "Non-slip surface for all types of yoga.", "price": 35.00},
            "P006": {"id": "P006", "name": "Coffee Maker Drip", "category": "Home & Office", "description": "Programmable 12-cup coffee maker.", "price": 59.99}
        }

    def get_product_details(self, product_id):
        return self.products.get(product_id)

class UserSimulator:
    def __init__(self):
        self.users = {
            "U001": {"id": "U001", "name": "Alice", "past_purchases": ["P001", "P004"], "browsing_history": ["P001", "P004", "P002"]},
            "U002": {"id": "U002", "name": "Bob", "past_purchases": ["P003"], "browsing_history": ["P003", "P005"]}
        }

    def get_user_profile(self, user_id):
        return self.users.get(user_id)

class RecommendationEngine:
    def __init__(self, product_db, user_simulator):
        self.product_db = product_db
        self.user_simulator = user_simulator

    def get_recommendations(self, user_id, num_recommendations=2):
        user_profile = self.user_simulator.get_user_profile(user_id)
        if not user_profile:
            return []

        user_categories = set()
        for prod_id in user_profile["past_purchases"] + user_profile["browsing_history"]:
            product = self.product_db.get_product_details(prod_id)
            if product: 
                user_categories.add(product["category"])

        recommended_products_with_reasons = []
        interacted_product_ids = set(user_profile["past_purchases"]) | set(user_profile["browsing_history"])

        for prod_id, product_details in self.product_db.products.items():
            if len(recommended_products_with_reasons) >= num_recommendations:
                break
            if prod_id not in interacted_product_ids and product_details["category"] in user_categories:
                reason = f"because you've shown interest in {product_details['category']} items."
                recommended_products_with_reasons.append((prod_id, reason))
            elif prod_id not in interacted_product_ids:
                reason = "a popular choice that might interest you."
                recommended_products_with_reasons.append((prod_id, reason))
        
        return recommended_products_with_reasons

class LLMExplainer:
    def generate_explanation(self, user_profile, product_details, reason):
        if not user_profile or not product_details:
            return "I cannot generate an explanation without sufficient details."

        explanation_template = (
            f"Hello {user_profile['name']}! Based on your recent activity, we think you'll love the "
            f"{product_details['name']} ({product_details['category']}). "
            f"We're recommending this because {reason} "
            f"This product, priced at ${product_details['price']:.2f}, offers {product_details['description']}."
        )
        return explanation_template

    def handle_interactive_query(self, user_query, product_details, current_explanation, context):
        user_query_lower = user_query.lower()
        product_name = product_details.get("name", "this product").lower()
        product_category = product_details.get("category", "its category").lower()

        if "why" in user_query_lower and product_name in user_query_lower:
            return f"You're interested in why we recommended the {product_details['name']}? " \
                   f"As mentioned, it's {context['reason_for_recommendation']} " \
                   f"Would you like to know more about its features?"
        elif "price" in user_query_lower:
            return f"The {product_details['name']} is priced at ${product_details['price']:.2f}. Is this within your budget?"
        elif "features" in user_query_lower or "what does it do" in user_query_lower:
            return f"The {product_details['name']} is known for: {product_details['description']}. " \
                   f"It's a great choice for {product_category} enthusiasts."
        elif "similar" in user_query_lower or "alternatives" in user_query_lower:
            return f"If you're looking for alternatives to {product_details['name']}, we can suggest other items in the {product_details['category']} category." \
                   f"However, this specific product is highly rated for its unique features."
        else:
            return f"I'm not sure how to answer that question about {product_details['name']}. " \
                   f"Could you please rephrase or ask something else?"

class CLIInterface:
    def __init__(self, recommendation_engine, llm_explainer, product_db, user_simulator):
        self.recommendation_engine = recommendation_engine
        self.llm_explainer = llm_explainer
        self.product_db = product_db
        self.user_simulator = user_simulator

    def run(self):
        print("\n--- Intelligent Product Explainer for E-commerce ---\n")
        user_id = input("Please enter your User ID (e.g., U001, U002): ").strip()
        user_profile = self.user_simulator.get_user_profile(user_id)

        if not user_profile:
            print(f"Error: User ID '{user_id}' not found. Exiting.")
            return
        
        print(f"\nWelcome, {user_profile['name']}! Getting your recommendations...\n")
        recommended_items_with_reasons = self.recommendation_engine.get_recommendations(user_id, num_recommendations=2)

        if not recommended_items_with_reasons:
            print("No new recommendations for you at the moment.")
            return

        print("--- Your Personalized Recommendations ---\n")
        recommendation_contexts = {}
        for i, (prod_id, reason) in enumerate(recommended_items_with_reasons):
            product_details = self.product_db.get_product_details(prod_id)
            if product_details:
                explanation = self.llm_explainer.generate_explanation(user_profile, product_details, reason)
                print(f"Recommendation {i+1}:\n")
                print(f"Product: {product_details['name']} ({product_details['category']})")
                print(f"Explanation: {explanation}\n")
                recommendation_contexts[str(i+1)] = {
                    "product_details": product_details,
                    "explanation": explanation,
                    "reason_for_recommendation": reason
                }
        
        while True:
            follow_up_choice = input(
                "Enter the number of a recommendation to ask a follow-up question (e.g., 1), "
                "or 'exit' to quit: "
            ).strip()

            if follow_up_choice.lower() == 'exit':
                print("Thank you for using the explainer. Goodbye!")
                break

            if follow_up_choice in recommendation_contexts:
                chosen_context = recommendation_contexts[follow_up_choice]
                product_details = chosen_context["product_details"]
                current_explanation = chosen_context["explanation"]
                context_for_llm = {"reason_for_recommendation": chosen_context["reason_for_recommendation"]}

                user_query = input(f"\nAsk a question about {product_details['name']}: ").strip()
                if user_query:
                    llm_response = self.llm_explainer.handle_interactive_query(
                        user_query, product_details, current_explanation, context_for_llm
                    )
                    print(f"LLM Response: {llm_response}\n")
                else:
                    print("No question entered. Returning to main menu.\n")
            else:
                print("Invalid choice. Please enter a valid recommendation number or 'exit'.\n")

if __name__ == "__main__":
    product_db = ProductDatabase()
    user_simulator = UserSimulator()
    recommendation_engine = RecommendationEngine(product_db, user_simulator)
    llm_explainer = LLMExplainer()

    cli = CLIInterface(recommendation_engine, llm_explainer, product_db, user_simulator)
    cli.run()