import json
import re

# --- 1. Define Mock External Tools ---

def get_faq_answer(question: str) -> str:
    """Simulates retrieving an answer from a FAQ knowledge base."""
    faqs = {
        "shipping cost": "Standard shipping within the US costs $5.99. Free shipping for orders over $50.",
        "return policy": "You can return products within 30 days of purchase with the original receipt.",
        "payment methods": "We accept Visa, MasterCard, American Express, PayPal, and Apple Pay.",
        "contact support": "You can reach our support team via email at support@ecommerce.com or call us at 1-800-123-4567."
    }
    question_lower = question.lower()
    for keyword, answer in faqs.items():
        if keyword in question_lower:
            return answer
    return "I'm sorry, I couldn't find an answer to that specific FAQ. Please try rephrasing or contact our live support."

def get_order_status(order_id: str) -> str:
    """Simulates querying an Order Management System for order status."""
    # In a real scenario, this would call an external API
    mock_orders = {
        "ORD12345": {"status": "Shipped", "tracking_number": "TRK67890", "estimated_delivery": "2024-07-20"},
        "ORD67890": {"status": "Processing", "estimated_delivery": "2024-07-25"},
        "ORD98765": {"status": "Delivered", "delivery_date": "2024-07-10"}
    }
    if order_id in mock_orders:
        return json.dumps(mock_orders[order_id])
    return f"Order {order_id} not found. Please double-check the order ID."

def process_product_return(order_id: str, product_id: str) -> str:
    """Simulates interacting with a Return Processing tool."""
    # In a real scenario, this would initiate a return process via an API
    if order_id.startswith("ORD") and product_id.startswith("PROD"):
        return f"Return for Order {order_id}, Product {product_id} has been initiated. You will receive an email with return instructions shortly."
    return "Invalid order ID or product ID for return processing."

def get_product_recommendations(user_id: str) -> str:
    """Simulates querying a Product Recommendation Engine."""
    # In a real scenario, this would call a recommendation API
    mock_recommendations = {
        "USER001": ["Laptop X", "Wireless Mouse", "Keyboard Cover"],
        "USER002": ["Running Shoes", "Fitness Tracker", "Water Bottle"],
        "USER003": ["Coffee Maker", "Espresso Beans", "Milk Frother"]
    }
    if user_id in mock_recommendations:
        return f"Here are some recommendations for you, {user_id}: {', '.join(mock_recommendations[user_id])}."
    return "I couldn't find personalized recommendations for you at this time."

# --- 2. LLM Orchestration (Simplified Agent Logic) ---

class SmartCustomerSupportAgent:
    def __init__(self):
        # Tools are defined as a dictionary for easy lookup by keyword
        self.tools = {
            "faq": get_faq_answer,
            "order_status": get_order_status,
            "return_processing": process_product_return,
            "product_recommendations": get_product_recommendations
        }

    def _decide_tool(self, query: str) -> tuple:
        """
        Simulates the LLM's decision-making process to choose a tool and extract arguments.
        In a real LLM-augmented system, the LLM would be prompted to output a tool call.
        """
        query_lower = query.lower()

        if "shipping" in query_lower or "return policy" in query_lower or "payment" in query_lower or "contact support" in query_lower:
            return "faq", {"question": query}
        elif "order status" in query_lower or "my order" in query_lower or "where is my order" in query_lower:
            match = re.search(r'(ord[0-9]{5})', query_lower)
            order_id = match.group(0).upper() if match else "UNKNOWN_ORDER"
            return "order_status", {"order_id": order_id}
        elif "return a product" in query_lower or "initiate return" in query_lower or "want to return" in query_lower:
            order_match = re.search(r'(ord[0-9]{5})', query_lower)
            product_match = re.search(r'(prod[0-9]{5})', query_lower)
            order_id = order_match.group(0).upper() if order_match else "UNKNOWN_ORDER"
            product_id = product_match.group(0).upper() if product_match else "UNKNOWN_PRODUCT"
            return "return_processing", {"order_id": order_id, "product_id": product_id}
        elif "recommendations" in query_lower or "suggest products" in query_lower or "what should i buy" in query_lower:
            user_match = re.search(r'(user[0-9]{3})', query_lower)
            user_id = user_match.group(0).upper() if user_match else "USER001" # Default user for demonstration
            return "product_recommendations", {"user_id": user_id}
        else:
            return None, {}

    def process_query(self, query: str) -> str:
        """
        Processes a customer query by deciding which tool to use and executing it.
        """
        tool_name, tool_args = self._decide_tool(query)

        if tool_name and tool_name in self.tools:
            print(f"[DEBUG] Agent decided to use tool: {tool_name} with arguments: {tool_args}")
            try:
                # Dynamically call the tool with its arguments
                return self.tools[tool_name](**tool_args)
            except TypeError as e:
                return f"Error executing tool {tool_name}: Missing or incorrect arguments. Details: {e}"
            except Exception as e:
                return f"An unexpected error occurred with tool {tool_name}: {e}"
        else:
            # Fallback for queries not handled by specific tools
            return "I'm a Smart Customer Support Agent. I can help with FAQs, order status, product returns, and recommendations. How can I assist you today?"

# --- Main Execution / Demonstration ---
if __name__ == "__main__":
    agent = SmartCustomerSupportAgent()

    print("--- Testing Smart Customer Support Agent ---")

    queries = [
        "What is your shipping cost?",
        "What is the status of my order ORD12345?",
        "I want to return a product. My order is ORD67890 and product is PROD001.",
        "Can you suggest some products for USER002?",
        "What is your return policy?",
        "Where is my order ORD67890?",
        "I'd like to return a product from ORD98765, product PROD005.",
        "Tell me about payment methods.",
        "How can I contact support?",
        "What is the weather like today?" # Unhandled query
    ]

    for i, query in enumerate(queries):
        print(f"\nCustomer Query {i+1}: '{query}'")
        response = agent.process_query(query)
        print(f"Agent Response: {response}")
        print("-" * 30)
