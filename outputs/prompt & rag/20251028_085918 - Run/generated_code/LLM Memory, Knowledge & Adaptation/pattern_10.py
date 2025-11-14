class DataLoader:
    def __init__(self):
        # Simulate a product database
        self.products = {
            "P001": {"name": "Laptop Pro X", "category": "Electronics", "price": 1200, "description": "High-performance laptop with 16GB RAM and 512GB SSD.", "stock": 10},
            "P002": {"name": "Wireless Headphones", "category": "Audio", "price": 150, "description": "Noise-cancelling headphones with 30-hour battery life.", "stock": 50},
            "P003": {"name": "Ergonomic Office Chair", "category": "Furniture", "price": 300, "description": "Adjustable chair for maximum comfort during long work hours.", "stock": 20},
            "P004": {"name": "Smartwatch Gen Z", "category": "Wearables", "price": 250, "description": "Fitness tracking, heart rate monitor, and notifications.", "stock": 30},
            "P005": {"name": "USB-C Hub", "category": "Accessories", "price": 40, "description": "Multi-port adapter for modern laptops.", "stock": 100}
        }

        # Simulate customer history/preferences
        self.customer_history = {
            "CUST001": {"name": "Alice Smith", "email": "alice@example.com", "past_orders": ["P001", "P002"], "preferences": ["Electronics", "Audio"]},
            "CUST002": {"name": "Bob Johnson", "email": "bob@example.com", "past_orders": ["P003"], "preferences": ["Furniture"]}
        }

        # Simulate a general knowledge base (FAQs)
        self.faq = [
            "What is your return policy? Our return policy allows returns within 30 days of purchase with original packaging.",
            "How can I track my order? You can track your order using the tracking number provided in your shipping confirmation email.",
            "Do you offer international shipping? Yes, we offer international shipping to most countries. Shipping fees and delivery times vary.",
            "What payment methods do you accept? We accept Visa, MasterCard, American Express, PayPal, and Apple Pay."
        ]

    def get_product_info(self, product_id):
        return self.products.get(product_id)

    def get_customer_info(self, customer_id):
        return self.customer_history.get(customer_id)

    def get_all_products(self):
        return self.products
    
    def get_faq(self):
        return self.faq


if __name__ == '__main__':
    loader = DataLoader()
    print("Product P001 Info:", loader.get_product_info("P001"))
    print("Customer CUST001 Info:", loader.get_customer_info("CUST001"))
    print("All FAQs:", loader.get_faq())
