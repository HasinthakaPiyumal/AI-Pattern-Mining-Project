class ECommerceEnvironment:
    """Simulates a simplified e-commerce platform environment."""

    def __init__(self):
        self.products = {
            "laptop": {"name": "Laptop Pro X", "price": "$1200", "description": "High-performance laptop for professionals.", "stock": 10},
            "mouse": {"name": "Wireless Mouse", "price": "$25", "description": "Ergonomic wireless mouse.", "stock": 50},
            "keyboard": {"name": "Mechanical Keyboard", "price": "$80", "description": "RGB mechanical keyboard with tactile switches.", "stock": 30},
            "monitor": {"name": "4K UHD Monitor", "price": "$350", "description": "27-inch 4K UHD monitor with HDR.", "stock": 15},
        }
        self.current_page = "home"
        self.search_results = []
        self.viewing_product = None

    def get_page_content(self):
        """Returns the textual content of the current simulated page."""
        if self.current_page == "home":
            return "Welcome to our E-commerce Store! \nSearch for products or browse our categories.\nAvailable products: Laptop, Mouse, Keyboard, Monitor."
        elif self.current_page == "search_results":
            if not self.search_results:
                return "No products found for your search query."
            content = "Search Results:\n"
            for i, product_name in enumerate(self.search_results):
                product = self.products[product_name.lower()]
                content += f"{i+1}. {product['name']} - {product['price']}\n"
            return content
        elif self.current_page == "product_detail" and self.viewing_product:
            product = self.products[self.viewing_product.lower()]
            return (f"Product Detail: {product['name']}\n" +
                    f"Price: {product['price']}\n" +
                    f"Description: {product['description']}\n" +
                    f"Stock: {product['stock']} units")
        return "Page not found or unknown state."

    def search_products(self, query):
        """Simulates searching for products."""
        self.search_results = []
        found = False
        for sku, product_info in self.products.items():
            if query.lower() in sku.lower() or query.lower() in product_info['name'].lower():
                self.search_results.append(sku)
                found = True
        self.current_page = "search_results"
        return found

    def navigate_to_product(self, product_name):
        """Simulates navigating to a product detail page."""
        if product_name.lower() in self.products:
            self.viewing_product = product_name
            self.current_page = "product_detail"
            return True
        return False

    def go_home(self):
        """Navigates back to the home page."""
        self.current_page = "home"
        self.search_results = []
        self.viewing_product = None
        return True
