import re

class EcommerceEnvironment:
    def __init__(self):
        self.products = [
            {"id": 101, "name": "Laptop Pro", "price": 1200.0, "description": "Powerful laptop for professionals"},
            {"id": 102, "name": "Gaming PC", "price": 1500.0, "description": "High-performance gaming desktop"},
            {"id": 103, "name": "Wireless Mouse", "price": 25.0, "description": "Ergonomic wireless mouse"},
            {"id": 104, "name": "Mechanical Keyboard", "price": 75.0, "description": "Durable mechanical keyboard"},
            {"id": 105, "name": "Monitor 27-inch", "price": 300.0, "description": "Full HD 27-inch monitor"}
        ]
        self.cart = {}

    def search_products(self, query: str) -> list:
        query_lower = query.lower()
        results = [p for p in self.products if query_lower in p["name"].lower() or query_lower in p["description"].lower()]
        return results

    def filter_products(self, min_price: float, max_price: float) -> list:
        results = [p for p in self.products if min_price <= p["price"] <= max_price]
        return results

    def add_to_cart(self, product_id: int, quantity: int = 1) -> str:
        product = next((p for p in self.products if p["id"] == product_id), None)
        if product:
            self.cart[product_id] = self.cart.get(product_id, 0) + quantity
            return f"Added {quantity} x {product['name']} to cart."
        return f"Product with ID {product_id} not found."

    def view_cart(self) -> list:
        cart_items = []
        for product_id, quantity in self.cart.items():
            product = next((p for p in self.products if p["id"] == product_id), None)
            if product:
                cart_items.append({"product": product, "quantity": quantity})
        return cart_items

    def proceed_to_checkout(self) -> str:
        if not self.cart:
            return "Your cart is empty. Nothing to checkout."
        total_price = sum(item["product"]["price"] * item["quantity"] for item in self.view_cart())
        self.cart = {}
        return f"Checkout successful! Total amount: ${total_price:.2f}. Your cart is now empty."

class EcommerceAgent:
    def __init__(self):
        self.environment = EcommerceEnvironment()

    def interpret_natural_language(self, user_request: str) -> str:
        user_request_lower = user_request.lower()

        if "search" in user_request_lower or "find" in user_request_lower:
            match = re.search(r"search (.+)|find (.+)", user_request_lower)
            if match:
                query = match.group(1) or match.group(2)
                return f"SEARCH_PRODUCT [{query.strip()}]"
        
        if "filter by price" in user_request_lower or "price between" in user_request_lower:
            match = re.search(r"(\$?\d+\.?\d*)\s*(to|-|and)\s*(\$?\d+\.?\d*)", user_request_lower)
            if match:
                min_price = float(re.sub(r"[^0-9.]", "", match.group(1)))
                max_price = float(re.sub(r"[^0-9.]", "", match.group(3)))
                return f"FILTER_BY_PRICE [{min_price} {max_price}]"

        if "add to cart" in user_request_lower:
            match = re.search(r"add (\d+) to cart", user_request_lower)
            if match:
                product_id = int(match.group(1))
                return f"ADD_TO_CART [{product_id}]"

        if "view cart" in user_request_lower or "show my cart" in user_request_lower:
            return "VIEW_CART []"

        if "checkout" in user_request_lower or "buy now" in user_request_lower:
            return "PROCEED_TO_CHECKOUT []"
            
        return "UNKNOWN_COMMAND []"

    def execute_structured_command(self, command_string: str) -> str:
        if not command_string.startswith("UNKNOWN_COMMAND"):
            command_parts = command_string.strip('[]').split(' ', 1)
            command_type = command_parts[0]
            args_str = command_parts[1] if len(command_parts) > 1 else ""

            if command_type == "SEARCH_PRODUCT":
                query = args_str.strip()
                results = self.environment.search_products(query)
                if results:
                    return "Search Results:\n" + "\n".join([f"ID: {p['id']}, Name: {p['name']}, Price: ${p['price']:.2f}" for p in results])
                return "No products found for your search."

            elif command_type == "FILTER_BY_PRICE":
                min_price_str, max_price_str = args_str.split(' ')
                min_price = float(min_price_str)
                max_price = float(max_price_str)
                results = self.environment.filter_products(min_price, max_price)
                if results:
                    return "Filtered Products:\n" + "\n".join([f"ID: {p['id']}, Name: {p['name']}, Price: ${p['price']:.2f}" for p in results])
                return "No products found in that price range."

            elif command_type == "ADD_TO_CART":
                product_id = int(args_str)
                return self.environment.add_to_cart(product_id)

            elif command_type == "VIEW_CART":
                cart_items = self.environment.view_cart()
                if cart_items:
                    return "Your Cart:\n" + "\n".join([f"Product: {item['product']['name']}, Quantity: {item['quantity']}, Price: ${item['product']['price']:.2f}" for item in cart_items])
                return "Your cart is empty."

            elif command_type == "PROCEED_TO_CHECKOUT":
                return self.environment.proceed_to_checkout()
        
        return "I didn't understand that command. Please try again."

    def interact(self, user_request: str) -> str:
        structured_command = self.interpret_natural_language(user_request)
        print(f"Agent interpreted: {structured_command}") # For debugging/demonstration
        response = self.execute_structured_command(structured_command)
        return response

if __name__ == "__main__":
    agent = EcommerceAgent()

    print("\n--- User Request: Search for laptops ---")
    print(agent.interact("I want to search for laptops"))

    print("\n--- User Request: Filter products between $1000 and $1500 ---")
    print(agent.interact("Show me products with price between $1000 and $1500"))

    print("\n--- User Request: Add product 101 to cart ---")
    print(agent.interact("Add product 101 to cart"))

    print("\n--- User Request: View my cart ---")
    print(agent.interact("Can I view my cart?"))

    print("\n--- User Request: Add product 104 to cart ---")
    print(agent.interact("add 104 to cart"))
    
    print("\n--- User Request: View my cart (again) ---")
    print(agent.interact("view cart"))

    print("\n--- User Request: Proceed to checkout ---")
    print(agent.interact("I want to checkout now"))

    print("\n--- User Request: View my cart (after checkout) ---")
    print(agent.interact("view cart"))

    print("\n--- User Request: Unknown command ---")
    print(agent.interact("What's the weather like?"))