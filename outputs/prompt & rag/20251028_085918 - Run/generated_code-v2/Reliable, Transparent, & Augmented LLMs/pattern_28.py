import random

def get_order_status(order_id: str) -> dict:
    """Simulates fetching order status from an external system."""
    if order_id == "ORDER123":
        return {"status": "Shipped", "details": "Your order is on its way. Estimated delivery: 2-3 business days.", "order_id": order_id}
    elif order_id == "ORDER456":
        return {"status": "Processing", "details": "Your order is being prepared for shipment.", "order_id": order_id}
    elif order_id == "MALICIOUS_ORDER":
        # Simulate a potentially harmful output from a compromised tool
        return {"status": "Error", "details": "<script>alert('Malicious code injected!');</script> Your order could not be found.", "order_id": order_id}
    else:
        return {"status": "Not Found", "details": f"Order with ID {order_id} could not be found.", "order_id": order_id}

def process_refund(order_id: str) -> dict:
    """Simulates processing a refund for an external system."""
    if order_id == "ORDER123":
        return {"status": "Refund Initiated", "details": "Refund for ORDER123 has been initiated. Funds should appear in your account within 5-7 business days.", "order_id": order_id}
    elif order_id == "ORDER789":
        return {"status": "Refund Denied", "details": "Refund for ORDER789 cannot be processed. Item not eligible.", "order_id": order_id}
    else:
        return {"status": "Error", "details": f"Cannot process refund for order {order_id}. Order not found or invalid.", "order_id": order_id}

def get_product_info(product_name: str) -> dict:
    """Simulates fetching product information from an external catalog."""
    products = {
        "Laptop": {"price": "$1200", "description": "High-performance laptop with 16GB RAM and 512GB SSD.", "availability": "In Stock"},
        "Mouse": {"price": "$25", "description": "Ergonomic wireless mouse.", "availability": "In Stock"},
        "Keyboard": {"price": "$75", "description": "Mechanical gaming keyboard with RGB lighting.", "availability": "Low Stock"},
        "Malware_Gadget": {"price": "$9999", "description": "This device will steal your data. Do NOT buy!", "availability": "Critical Security Alert"}
    }
    product_info = products.get(product_name, {"description": f"No information found for {product_name}.", "availability": "N/A"})
    product_info["product_name"] = product_name
    return product_info