"""This module implements a two-stage AI-powered customer support agent to manage cognitive load. It separates information gathering from response generation.
"""

class InformationGatheringModule:
    def __init__(self):
        self.knowledge_base = {
            "product_faq": "Our products are made from high-quality, sustainable materials. Returns are accepted within 30 days.",
            "shipping_times": "Standard shipping takes 5-7 business days. Expedited shipping is available.",
            "billing_issues": "For billing discrepancies, please check your invoice or contact our finance department.",
            "account_setup": "To set up a new account, visit our website and click 'Sign Up'."
        }
        self.order_history_db = {
            "customer123": {"order_id_1": {"status": "shipped", "items": "Laptop", "date": "2023-01-15"}, "order_id_2": {"status": "processing", "items": "Mouse", "date": "2023-02-01"}},
            "customer456": {"order_id_3": {"status": "delivered", "items": "Keyboard", "date": "2023-01-20"}}
        }
        self.crm_db = {
            "customer123": {"name": "Alice Smith", "email": "alice@example.com", "account_type": "premium"},
            "customer456": {"name": "Bob Johnson", "email": "bob@example.com", "account_type": "standard"}
        }

    def _query_parser(self, query: str) -> list:
        keywords = []
        query_lower = query.lower()
        if "order status" in query_lower or "order id" in query_lower or "my order" in query_lower:
            keywords.append("order_details")
        if "return" in query_lower or "warranty" in query_lower:
            keywords.append("product_faq")
        if "billing" in query_lower or "invoice" in query_lower:
            keywords.append("billing_issues")
        if "account" in query_lower or "my details" in query_lower or "email" in query_lower:
            keywords.append("account_details")
        if "shipping" in query_lower or "delivery" in query_lower:
            keywords.append("shipping_times")
        return keywords

    def gather_information(self, customer_query: str, customer_id: str = None) -> dict:
        gathered_info = {"query": customer_query}
        parsed_keywords = self._query_parser(customer_query)

        for keyword in parsed_keywords:
            if keyword == "product_faq":
                gathered_info["product_faq"] = self.knowledge_base.get("product_faq", "No product FAQ found.")
            elif keyword == "shipping_times":
                gathered_info["shipping_times"] = self.knowledge_base.get("shipping_times", "No shipping information found.")
            elif keyword == "billing_issues":
                gathered_info["billing_issues"] = self.knowledge_base.get("billing_issues", "No billing information found.")
            elif keyword == "account_details" and customer_id:
                gathered_info["account_info"] = self.crm_db.get(customer_id, "Customer not found.")
            elif keyword == "order_details" and customer_id:
                gathered_info["order_history"] = self.order_history_db.get(customer_id, "No order history found.")
        
        return gathered_info


class ResponseGenerationModule:
    def _simulate_llm_response(self, query: str, gathered_info: dict) -> str:
        response_parts = []
        response_parts.append(f"Hello! Thank you for contacting support regarding your query: '{query}'. ")

        if "order_history" in gathered_info and isinstance(gathered_info["order_history"], dict):
            response_parts.append("Based on your order history, here are some details:")
            for order_id, details in gathered_info["order_history"].items():
                response_parts.append(f"  - Order {order_id}: Status is '{details.get('status')}' for '{details.get('items')}' placed on {details.get('date')}.")

        if "product_faq" in gathered_info:
            response_parts.append(f"Regarding product information, we can tell you that: {gathered_info['product_faq']}")
        
        if "shipping_times" in gathered_info:
            response_parts.append(f"For shipping, please note: {gathered_info['shipping_times']}")

        if "billing_issues" in gathered_info:
            response_parts.append(f"Concerning billing: {gathered_info['billing_issues']}")

        if "account_info" in gathered_info and isinstance(gathered_info["account_info"], dict):
            response_parts.append(f"Your account details indicate you are {gathered_info['account_info'].get('name')} with a {gathered_info['account_info'].get('account_type')} account.")
        
        if len(response_parts) == 1: # Only the initial greeting
            response_parts.append("I couldn't find specific information for your request at this moment. Could you please provide more details?")

        response_parts.append("Is there anything else I can assist you with today?")
        return "\n".join(response_parts)

    def generate_response(self, original_query: str, gathered_information: dict) -> str:
        return self._simulate_llm_response(original_query, gathered_information)


class CustomerSupportAgent:
    def __init__(self):
        self.info_gatherer = InformationGatheringModule()
        self.response_generator = ResponseGenerationModule()

    def handle_query(self, customer_query: str, customer_id: str = None) -> str:
        # Stage 1: Information Gathering
        gathered_info = self.info_gatherer.gather_information(customer_query, customer_id)
        
        # Stage 2: Response Generation
        final_response = self.response_generator.generate_response(customer_query, gathered_info)
        
        return final_response


if __name__ == "__main__":
    agent = CustomerSupportAgent()

    print("--- Test Case 1: Order Status Query ---")
    query1 = "What is the status of my recent order? My customer ID is customer123."
    response1 = agent.handle_query(query1, customer_id="customer123")
    print(response1)
    print("\n")

    print("--- Test Case 2: Product Return Policy ---")
    query2 = "Can I return a product? What's your policy?"
    response2 = agent.handle_query(query2)
    print(response2)
    print("\n")

    print("--- Test Case 3: Billing Question ---")
    query3 = "I have a question about my invoice."
    response3 = agent.handle_query(query3)
    print(response3)
    print("\n")

    print("--- Test Case 4: Account Details Query ---")
    query4 = "Can you tell me my account details? My ID is customer123."
    response4 = agent.handle_query(query4, customer_id="customer123")
    print(response4)
    print("\n")

    print("--- Test Case 5: Unrecognized Query ---")
    query5 = "Do you sell flying cars?"
    response5 = agent.handle_query(query5)
    print(response5)
    print("\n")