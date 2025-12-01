import random

class ECommerceTools:
    """A collection of simulated e-commerce backend tools."""

    def get_product_availability(self, product_id: str) -> dict:
        """Checks the current stock level for a given product ID.

        Args:
            product_id: The unique identifier for the product.

        Returns:
            A dictionary containing product ID, availability status, and stock quantity.
        """
        if product_id.startswith("PROD"): # Simple validation
            is_available = random.choice([True, False])
            stock = random.randint(0, 100) if is_available else 0
            status = "In Stock" if is_available else "Out of Stock"
            return {"product_id": product_id, "status": status, "stock": stock}
        else:
            return {"error": "Invalid product ID format. Expected PRODxxxx."}

    def get_shipping_estimate(self, product_id: str, destination: str) -> dict:
        """Estimates shipping time and cost for a product to a destination.

        Args:
            product_id: The unique identifier for the product.
            destination: The shipping destination (e.g., 'New York', 'London').

        Returns:
            A dictionary with estimated days and cost.
        """
        if product_id.startswith("PROD") and len(destination) > 2:
            days = random.randint(3, 10)
            cost = round(random.uniform(5.00, 50.00), 2)
            return {"product_id": product_id, "destination": destination, "estimated_days": days, "estimated_cost": cost}
        else:
            return {"error": "Invalid product ID or destination."}

    def place_order(self, product_id: str, quantity: int) -> dict:
        """Simulates placing an order for a product.

        Args:
            product_id: The unique identifier for the product.
            quantity: The number of units to order.

        Returns:
            A dictionary with order confirmation or error.
        """
        if product_id.startswith("PROD") and quantity > 0:
            order_id = f"ORD{random.randint(10000, 99999)}"
            return {"status": "Order Placed", "order_id": order_id, "product_id": product_id, "quantity": quantity}
        else:
            return {"error": "Invalid product ID or quantity."}

    def get_order_status(self, order_id: str) -> dict:
        """Retrieves the current status of an existing order.

        Args:
            order_id: The unique identifier for the order.

        Returns:
            A dictionary with order status and details.
        """
        if order_id.startswith("ORD"): # Simulate some order statuses
            statuses = ["Processing", "Shipped", "Delivered", "Cancelled"]
            current_status = random.choice(statuses)
            return {"order_id": order_id, "status": current_status, "details": "Your order is on its way!" if current_status == "Shipped" else ""}
        else:
            return {"error": "Invalid order ID format. Expected ORDxxxxx."}

    def process_payment(self, order_id: str, amount: float) -> dict:
        """Simulates processing a payment for an order.

        Args:
            order_id: The unique identifier for the order.
            amount: The amount to be paid.

        Returns:
            A dictionary with payment confirmation or error.
        """
        if order_id.startswith("ORD") and amount > 0:
            transaction_id = f"TXN{random.randint(1000000, 9999999)}"
            return {"status": "Payment Successful", "order_id": order_id, "amount": amount, "transaction_id": transaction_id}
        else:
            return {"error": "Invalid order ID or amount."}

