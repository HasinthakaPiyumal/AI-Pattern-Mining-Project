class Config:
    # Mock LLM API Key (replace with your actual key)
    OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
    # Example LLM model name
    LLM_MODEL_NAME = "gpt-4-turbo"
    
    # Threshold for similarity in consistency checks (e.g., for vector embeddings)
    CONSISTENCY_THRESHOLD = 0.8
    
    # Ethical guidelines for Constitutional AI (simplified)
    ETHICAL_GUIDELINES = [
        "Ensure responses are always helpful and polite.",
        "Do not provide medical or legal advice.",
        "Avoid making definitive statements about uncertain order statuses.",
        "Prioritize customer privacy.",
        "Avoid discriminatory or biased language."
    ]

    # Mock product and order data (for demonstration)
    MOCK_PRODUCTS = {
        "P101": {"name": "Wireless Headphones", "price": 129.99, "description": "Noise-cancelling, Bluetooth 5.0, 20-hour battery life.", "stock": 150},
        "P102": {"name": "Smartwatch", "price": 249.00, "description": "Heart rate monitor, GPS, waterproof, 7-day battery.", "stock": 80},
        "P103": {"name": "Portable Charger", "price": 39.50, "description": "10000mAh, fast charging, USB-C compatible.", "stock": 300},
    }

    MOCK_ORDERS = {
        "ORD001": {"customer_id": "C001", "product_id": "P101", "status": "Shipped", "tracking": "TRK12345", "delivery_date": "2023-11-20"},
        "ORD002": {"customer_id": "C002", "product_id": "P103", "status": "Processing", "tracking": "N/A", "delivery_date": "N/A"},
        "ORD003": {"customer_id": "C001", "product_id": "P102", "status": "Delivered", "tracking": "TRK67890", "delivery_date": "2023-11-15"},
    }
