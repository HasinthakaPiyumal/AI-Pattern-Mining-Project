import enum
import re

class Command(enum.Enum):
    SEARCH = "SEARCH"
    ADD_TO_CART = "ADD_TO_CART"
    VIEW_CART = "VIEW_CART"
    APPLY_COUPON = "APPLY_COUPON"
    CHECKOUT = "CHECKOUT"
    EXIT = "EXIT"
    UNKNOWN = "UNKNOWN"

class ECommerceEnvironment:
    def __init__(self):
        self.products = {
            "prod101": {"name": "Laptop Pro", "price": 1200.00, "stock": 5},
            "prod102": {"name": "Wireless Mouse", "price": 25.00, "stock": 50},
            "prod103": {"name": "Mechanical Keyboard", "price": 80.00, "stock": 20},
            "prod104": {"name": "Monitor 27 inch", "price": 300.00, "stock": 10},
            "prod105": {"name": "USB-C Hub", "price": 45.00, "stock": 30},
        }
        self.cart = {}
        self.coupons = {"SAVE10": 0.10, "FREESHIP": 0.05} # Example: 10% off, 5% off (simulated)
        self.applied_coupon = None

    def _get_product_details(self, product_id):
        return self.products.get(product_id)

    def search_products(self, query):
        found_products = []
        for prod_id, details in self.products.items():
            if query.lower() in details["name"].lower():
                found_products.append(details)
        return found_products

    def add_to_cart(self, product_id, quantity):
        product = self._get_product_details(product_id)
        if not product:
            return False, "Product not found."
        if product["stock"] < quantity:
            return False, f"Not enough stock for {product['name']}. Available: {product['stock']}"

        self.cart[product_id] = self.cart.get(product_id, 0) + quantity
        return True, f"{quantity} of {product['name']} added to cart."

    def view_cart(self):
        cart_summary = {"items": [], "total": 0.0}
        for prod_id, quantity in self.cart.items():
            product = self._get_product_details(prod_id)
            if product:
                item_total = product["price"] * quantity
                cart_summary["items"].append({
                    "product_name": product["name"],
                    "quantity": quantity,
                    "price_per_item": product["price"],
                    "item_total": item_total
                })
                cart_summary["total"] += item_total
        
        if self.applied_coupon:
            discount_percentage = self.coupons[self.applied_coupon]
            discount_amount = cart_summary["total"] * discount_percentage
            cart_summary["discount"] = discount_amount
            cart_summary["total"] -= discount_amount

        return cart_summary

    def apply_coupon(self, coupon_code):
        if coupon_code in self.coupons:
            self.applied_coupon = coupon_code
            return True, f"Coupon '{coupon_code}' applied successfully."
        return False, "Invalid coupon code."

    def checkout(self):
        if not self.cart:
            return False, "Your cart is empty. Nothing to checkout."
        
        total_amount = self.view_cart()["total"]
        for prod_id, quantity in self.cart.items():
            self.products[prod_id]["stock"] -= quantity
        
        self.cart = {}
        self.applied_coupon = None
        return True, f"Checkout successful! Total amount paid: ${total_amount:.2f}"

class LLMAgent:
    def __init__(self):
        pass # Simulated LLM, no model to load

    def process_input(self, user_input):
        user_input_lower = user_input.lower()

        if "search" in user_input_lower:
            match = re.search(r"search (.+)", user_input_lower)
            if match:
                query = match.group(1).strip()
                return Command.SEARCH, {"query": query}
            else:
                return Command.UNKNOWN, {"error": "Please specify a search query.", "help": "Example: search laptop"}
        
        elif "add" in user_input_lower and "cart" in user_input_lower:
            match = re.search(r"add (.+) to cart(?: quantity (\\d+))?", user_input_lower)
            if match:
                item_str = match.group(1).strip()
                quantity_str = match.group(2)
                
                # Simple mapping for product_id based on name keywords
                product_id_map = {
                    "laptop": "prod101",
                    "mouse": "prod102",
                    "keyboard": "prod103",
                    "monitor": "prod104",
                    "usb-c hub": "prod105"
                }
                product_id = ""
                for keyword, p_id in product_id_map.items():
                    if keyword in item_str:
                        product_id = p_id
                        break
                
                if not product_id:
                    return Command.UNKNOWN, {"error": f"Could not identify product from '{item_str}'.", "help": "Try: add laptop to cart or add wireless mouse to cart"}

                quantity = int(quantity_str) if quantity_str else 1
                return Command.ADD_TO_CART, {"product_id": product_id, "quantity": quantity}
            else:
                return Command.UNKNOWN, {"error": "Please specify item and optionally quantity to add to cart.", "help": "Example: add laptop to cart quantity 2"}

        elif "view cart" in user_input_lower:
            return Command.VIEW_CART, {}
        
        elif "apply coupon" in user_input_lower:
            match = re.search(r"apply coupon (.+)", user_input_lower)
            if match:
                coupon_code = match.group(1).strip().upper()
                return Command.APPLY_COUPON, {"coupon_code": coupon_code}
            else:
                return Command.UNKNOWN, {"error": "Please specify a coupon code.", "help": "Example: apply coupon SAVE10"}

        elif "checkout" in user_input_lower:
            return Command.CHECKOUT, {}
        
        elif user_input_lower == "exit" or user_input_lower == "quit":
            return Command.EXIT, {}

        return Command.UNKNOWN, {"error": "I didn't understand that command.", "help": "Try: search, add to cart, view cart, apply coupon, checkout, exit"}


def main():
    env = ECommerceEnvironment()
    agent = LLMAgent()

    print("Welcome to the E-commerce Shopping Assistant! Type 'exit' to quit.")
    print("Available products: Laptop Pro, Wireless Mouse, Mechanical Keyboard, Monitor 27 inch, USB-C Hub")
    print("Available coupons: SAVE10, FREESHIP")

    while True:
        user_input = input("\nYour command: ")
        command, args = agent.process_input(user_input)

        if command == Command.EXIT:
            print("Exiting. Goodbye!")
            break
        elif command == Command.SEARCH:
            query = args["query"]
            results = env.search_products(query)
            if results:
                print(f"Found products for '{query}':")
                for p in results:
                    print(f"  - {p['name']} (ID: {list(env.products.keys())[list(env.products.values()).index(p)]}, Price: ${p['price']:.2f}, Stock: {p['stock']})")
            else:
                print(f"No products found for '{query}'.")
        elif command == Command.ADD_TO_CART:
            success, message = env.add_to_cart(args["product_id"], args["quantity"])
            print(message)
        elif command == Command.VIEW_CART:
            cart_summary = env.view_cart()
            if cart_summary["items"]:
                print("\n--- Your Cart ---")
                for item in cart_summary["items"]:
                    print(f"  {item['product_name']} x {item['quantity']} (${item['price_per_item']:.2f} each) = ${item['item_total']:.2f}")
                if "discount" in cart_summary:
                    print(f"Discount Applied: ${cart_summary['discount']:.2f}")
                print(f"Total: ${cart_summary['total']:.2f}")
                print("-----------------")
            else:
                print("Your cart is empty.")
        elif command == Command.APPLY_COUPON:
            success, message = env.apply_coupon(args["coupon_code"])
            print(message)
        elif command == Command.CHECKOUT:
            success, message = env.checkout()
            print(message)
        elif command == Command.UNKNOWN:
            print(f"Error: {args.get('error', 'Unknown command.')}")
            if "help" in args:
                print(f"Hint: {args['help']}")

if __name__ == "__main__":
    main()