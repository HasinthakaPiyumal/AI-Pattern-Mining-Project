# Mock data to simulate an e-commerce backend
mock_orders = {
    "12345": {"status": "Shipped", "tracking_id": "TRK987654321", "items": [{"product_id": "P001", "quantity": 1}], "customer_id": "C001"},
    "67890": {"status": "Processing", "tracking_id": None, "items": [{"product_id": "P002", "quantity": 2}], "customer_id": "C002"},
}

mock_products = {
    "P001": {"name": "Wireless Headphones", "price": 99.99, "description": "High-quality wireless headphones with noise cancellation.", "return_policy": "30-day free returns."},
    "P002": {"name": "Smartwatch", "price": 199.99, "description": "Feature-rich smartwatch with health tracking.", "return_policy": "15-day free returns, must be in original packaging."},
    "P003": {"name": "USB-C Hub", "price": 29.99, "description": "Multi-port USB-C adapter.", "return_policy": "60-day returns with receipt."},
}

mock_faqs = {
    "shipping": "Standard shipping takes 5-7 business days. Express shipping takes 2-3 business days.",
    "returns": "Items can be returned within 30 days of purchase. Please visit our returns portal for more details.",
    "payment methods": "We accept Visa, MasterCard, American Express, PayPal, and Apple Pay.",
    "contact support": "You can reach our support team via email at support@ecommerce.com or by phone at 1-800-555-0123.",
}

def get_order_details(order_id: str) -> dict:
    """Retrieves details for a given order ID."""
    return mock_orders.get(order_id, {"error": "Order not found."})

def get_product_details(product_query: str) -> dict:
    """Searches for product details based on a query (name or ID)."""
    product_query = product_query.lower()
    for prod_id, details in mock_products.items():
        if product_query in details["name"].lower() or product_query == prod_id.lower():
            return details
    return {"error": "Product not found."}

def get_faq_answer(query: str) -> str:
    """Finds an answer to a common FAQ query."""
    query = query.lower()
    for keyword, answer in mock_faqs.items():
        if keyword in query:
            return answer
    return "I'm sorry, I couldn't find an answer to that in our FAQs. Please try rephrasing or contact support."
