class SimulatedEcommerceEnv:
    def __init__(self):
        self.products = {
            "P001": {"name": "Laptop Pro", "price": 1200, "stock": 10, "description": "High-performance laptop."},
            "P002": {"name": "Wireless Mouse", "price": 25, "stock": 50, "description": "Ergonomic wireless mouse."},
            "P003": {"name": "USB-C Hub", "price": 40, "stock": 20, "description": "Multi-port USB-C adapter."}
        }
        self.orders = {
            "ORD123": {"customer_id": "C001", "items": [{"product_id": "P001", "qty": 1}], "status": "shipped", "tracking": "TRK987654"},
            "ORD124": {"customer_id": "C002", "items": [{"product_id": "P002", "qty": 2}], "status": "pending", "tracking": "N/A"}
        }
        self.faqs = {
            "return_policy": "You can return items within 30 days of purchase. See our full policy for details.",
            "shipping_times": "Standard shipping takes 3-5 business days. Express options are available."
        }

    def get_product_details(self, product_id):
        return self.products.get(product_id)

    def get_order_status(self, order_id):
        order = self.orders.get(order_id)
        if order: 
            items_details = []
            for item in order["items"]:
                product_info = self.products.get(item["product_id"])
                if product_info:
                    items_details.append(f"{item["qty"]}x {product_info["name"]}")
                else:
                    items_details.append(f"{item["qty"]}x Unknown Product ({item["product_id"]})")
            return {"status": order["status"], "tracking": order["tracking"], "items": ", ".join(items_details)}
        return None

    def get_faq_answer(self, query):
        query = query.lower()
        if "return" in query or "policy" in query:
            return self.faqs["return_policy"]
        if "shipping" in query or "delivery" in query:
            return self.faqs["shipping_times"]
        return "I'm sorry, I don't have an answer to that specific FAQ."

    def search_database(self, query):
        # A more sophisticated search would involve NLP and vector databases
        if "order" in query:
            order_id_match = next((word for word in query.split() if word.startswith("ORD")), None)
            if order_id_match:
                return {"type": "order", "data": self.get_order_status(order_id_match)}
        elif "product" in query:
            product_id_match = next((word for word in query.split() if word.startswith("P")), None)
            if product_id_match:
                return {"type": "product", "data": self.get_product_details(product_id_match)}
        elif "faq" in query or "policy" in query or "shipping" in query or "return" in query:
            return {"type": "faq", "data": self.get_faq_answer(query)}
        return None
