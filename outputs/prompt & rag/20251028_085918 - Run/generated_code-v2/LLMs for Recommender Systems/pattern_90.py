"""
This module simulates an external e-commerce recommendation system API.
In a real-world scenario, this would interact with a database and a recommendation engine.
"""

def get_product_recommendations(user_preferences: dict, query: str = None) -> list:
    """
    Simulates fetching product recommendations based on user preferences and an optional query.
    """
    print(f"[Tool Call] Getting recommendations for preferences: {user_preferences}, query: {query}")
    # This is a simplified simulation. In reality, this would involve complex logic.
    # Example static recommendations based on a few keywords
    if "electronics" in query.lower() or "laptop" in user_preferences.get("interests", "").lower():
        return [
            {"id": "prod101", "name": "Dell XPS 13 Laptop", "price": 1299.99, "category": "Electronics"},
            {"id": "prod102", "name": "Sony WH-1000XM4 Headphones", "price": 279.00, "category": "Electronics"},
        ]
    elif "books" in query.lower() or "reading" in user_preferences.get("interests", "").lower():
        return [
            {"id": "prod201", "name": "The Hitchhiker's Guide to the Galaxy", "price": 9.99, "category": "Books"},
            {"id": "prod202", "name": "Dune", "price": 12.50, "category": "Books"},
        ]
    elif "clothing" in query.lower() or "fashion" in user_preferences.get("interests", "").lower():
        return [
            {"id": "prod301", "name": "Classic Denim Jacket", "price": 59.00, "category": "Apparel"},
            {"id": "prod302", "name": "Comfort Fit T-Shirt", "price": 19.99, "category": "Apparel"},
        ]
    else:
        return [
            {"id": "prod901", "name": "Generic Product A", "price": 25.00, "category": "General"},
            {"id": "prod902", "name": "Generic Product B", "price": 50.00, "category": "General"},
        ]

def get_product_details(product_id: str) -> dict:
    """
    Simulates fetching details for a specific product ID.
    """
    print(f"[Tool Call] Getting details for product ID: {product_id}")
    # Placeholder for actual product database lookup
    products = {
        "prod101": {"id": "prod101", "name": "Dell XPS 13 Laptop", "price": 1299.99, "category": "Electronics", "description": "A high-performance ultrabook with a stunning display."},
        "prod102": {"id": "prod102", "name": "Sony WH-1000XM4 Headphones", "price": 279.00, "category": "Electronics", "description": "Industry-leading noise canceling Bluetooth headphones."},
        "prod201": {"id": "prod201", "name": "The Hitchhiker's Guide to the Galaxy", "price": 9.99, "category": "Books", "description": "A comedic science fiction series by Douglas Adams."},
        "prod202": {"id": "prod202", "name": "Dune", "price": 12.50, "category": "Books", "description": "An epic science fiction novel by Frank Herbert."},
        "prod301": {"id": "prod301", "name": "Classic Denim Jacket", "price": 59.00, "category": "Apparel", "description": "A timeless denim jacket for all seasons."},
        "prod302": {"id": "prod302", "name": "Comfort Fit T-Shirt", "price": 19.99, "category": "Apparel", "description": "Soft and breathable cotton t-shirt."},
        "prod901": {"id": "prod901", "name": "Generic Product A", "price": 25.00, "category": "General", "description": "A useful general purpose item."},
        "prod902": {"id": "prod902", "name": "Generic Product B", "price": 50.00, "category": "General", "description": "A high-quality general purpose item."},
    }
    return products.get(product_id)
