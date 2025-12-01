import json

class EcommerceChatbot:
    def __init__(self):
        pass

    def _get_product_info(self, product_name: str) -> dict:
        product_name = product_name.lower()
        if "laptop" in product_name:
            return {"name": "Laptop Pro X", "price": "$1200", "description": "High-performance laptop with 16GB RAM and 512GB SSD.", "availability": "In Stock"}
        elif "mouse" in product_name:
            return {"name": "Wireless Mouse M1", "price": "$25", "description": "Ergonomic wireless mouse.", "availability": "In Stock"}
        else:
            return {}

    def _get_order_status(self, order_id: str) -> dict:
        if order_id == "ORD12345":
            return {"order_id": order_id, "status": "Shipped", "estimated_delivery": "2023-10-27"}
        elif order_id == "ORD67890":
            return {"order_id": order_id, "status": "Processing", "estimated_delivery": "2023-11-01"}
        else:
            return {}

    def handle_query(self, query: str) -> str:
        query_lower = query.lower()
        response_data = {}
        response_type = "unknown"
        response_message = "I'm sorry, I couldn't understand your request."

        if "product info" in query_lower or "tell me about" in query_lower:
            product_name = ""
            if "laptop" in query_lower:
                product_name = "laptop"
            elif "mouse" in query_lower:
                product_name = "mouse"
            
            if product_name:
                product_info = self._get_product_info(product_name)
                if product_info:
                    response_type = "product_information"
                    response_message = f"Here is the information for {product_info['name']}."
                    response_data = product_info
                else:
                    response_message = f"Sorry, I couldn't find information for {product_name}."
            else:
                response_message = "Please specify which product you are interested in."

        elif "order status" in query_lower or "my order" in query_lower:
            order_id = None
            # Simple extraction for demo purposes, a real system would use regex or NLP entity extraction
            for word in query.split():
                if word.startswith("ORD") and len(word) == 8 and word[3:].isdigit():
                    order_id = word
                    break
            
            if order_id:
                order_status = self._get_order_status(order_id)
                if order_status:
                    response_type = "order_status"
                    response_message = f"Here is the status for your order {order_status['order_id']}."
                    response_data = order_status
                else:
                    response_message = f"Sorry, I couldn't find any order with ID {order_id}."
            else:
                response_message = "Please provide your order ID to check the status."

        return json.dumps({"type": response_type, "message": response_message, "data": response_data}, indent=4)