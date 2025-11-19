
# filename: recommender_system.py

# A mock Large Language Model service to simulate LLM interactions.
# In a real application, this would be an actual LLM API call (e.g., OpenAI, Google Gemini).
class LLMMockService:
    def get_recommendations(self, user_query: str) -> list:
        """Simulates LLM generating product recommendations based on user query."""
        if "laptop" in user_query.lower():
            return [
                {"id": "L101", "name": "Lightweight Ultrabook", "price": 1200, "category": "Electronics"},
                {"id": "L102", "name": "Gaming Laptop Pro", "price": 1800, "category": "Electronics"},
            ]
        elif "shirt" in user_query.lower():
            return [
                {"id": "S201", "name": "Cotton T-Shirt", "price": 25, "category": "Apparel"},
                {"id": "S202", "name": "Formal Dress Shirt", "price": 50, "category": "Apparel"},
            ]
        else:
            # Default recommendations
            return [
                {"id": "P001", "name": "Universal Gadget Charger", "price": 30, "category": "Accessories"},
                {"id": "P002", "name": "Ergonomic Office Chair", "price": 250, "category": "Furniture"},
            ]

    def get_explanation(self, product_name: str, user_query: str) -> str:
        """Simulates LLM generating an explanation for a recommendation."""
        return (f"The '{product_name}' is recommended because based on your interest in '{user_query}', "
                f"it aligns with high-demand features and positive user feedback in its category.")

def get_product_recommendations_with_explanation(user_query: str) -> list:
    """
    Generates product recommendations and personalized explanations using a mock LLM service.

    Args:
        user_query: A string representing the user's current interest or search query.

    Returns:
        A list of dictionaries, where each dictionary contains product details
        and a generated explanation for the recommendation.
    """
    llm_service = LLMMockService()
    recommendations = llm_service.get_recommendations(user_query)

    detailed_recommendations = []
    for product in recommendations:
        explanation = llm_service.get_explanation(product["name"], user_query)
        product_with_explanation = product.copy()
        product_with_explanation["explanation"] = explanation
        detailed_recommendations.append(product_with_explanation)

    return detailed_recommendations

if __name__ == "__main__":
    # Example Usage
    print("--- Recommendations for 'laptop' ---")
    laptop_recs = get_product_recommendations_with_explanation("I'm looking for a new laptop for work.")
    for rec in laptop_recs:
        print(f"Product: {rec['name']} (ID: {rec['id']}, Price: ${rec['price']})\n  Explanation: {rec['explanation']}\n")

    print("\n--- Recommendations for 'casual shirt' ---")
    shirt_recs = get_product_recommendations_with_explanation("I need a casual shirt.")
    for rec in shirt_recs:
        print(f"Product: {rec['name']} (ID: {rec['id']}, Price: ${rec['price']})\n  Explanation: {rec['explanation']}\n")

    print("\n--- General Recommendations ---")
    general_recs = get_product_recommendations_with_explanation("something useful")
    for rec in general_recs:
        print(f"Product: {rec['name']} (ID: {rec['id']}, Price: ${rec['price']})\n  Explanation: {rec['explanation']}\n")
