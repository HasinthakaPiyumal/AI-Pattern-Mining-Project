from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import FakeListLLM

# --- Simulated Databases ---
crm_data = {
    "cust_001": {"name": "Alice Smith", "email": "alice@example.com", "status": "Gold", "last_interactions": ["2023-10-26: inquired about order #12345", "2023-11-01: updated shipping address"]},
    "cust_002": {"name": "Bob Johnson", "email": "bob@example.com", "status": "Silver", "last_interactions": ["2023-11-05: placed new order #67890"]},
}

order_data = {
    "ord_12345": {"customer_id": "cust_001", "items": [{"product_id": "prod_A", "qty": 1}, {"product_id": "prod_B", "qty": 2}], "status": "Shipped", "shipping_address": "123 Main St", "payment_status": "Paid"},
    "ord_67890": {"customer_id": "cust_002", "items": [{"product_id": "prod_C", "qty": 1}], "status": "Processing", "shipping_address": "456 Oak Ave", "payment_status": "Pending"},
}

product_kb = {
    "prod_A": {"name": "Laptop Pro", "category": "Electronics", "price": 1200, "warranty": "1 year", "return_policy": "30 days unopened"},
    "prod_B": {"name": "Wireless Mouse", "category": "Accessories", "price": 25, "warranty": "90 days", "return_policy": "15 days"},
    "prod_C": {"name": "Ergonomic Keyboard", "category": "Accessories", "price": 75, "warranty": "1 year", "return_policy": "30 days unopened"},
    "shipping_policy": "Standard shipping takes 5-7 business days. Express shipping takes 1-2 business days. Free shipping on orders over $50.",
    "refund_process": "Refunds are processed within 3-5 business days after the returned item is received and inspected."
}

class InformationCollector:
    def __init__(self):
        pass

    def retrieve_crm_data(self, customer_id):
        return crm_data.get(customer_id, {})

    def retrieve_order_data(self, order_id):
        order_info = order_data.get(order_id, {})
        if order_info and "items" in order_info:
            order_info["item_details"] = []
            for item in order_info["items"]:
                product_id = item["product_id"]
                product_details = product_kb.get(product_id, {})
                order_info["item_details"].append({**item, **product_details})
        return order_info

    def search_product_kb(self, query):
        results = {}
        for key, value in product_kb.items():
            if isinstance(value, str) and query.lower() in value.lower():
                results[key] = value
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, str) and query.lower() in sub_value.lower():
                        if key not in results:
                            results[key] = value
                        break
        return results

    def collect_info(self, inquiry, customer_id=None, order_id=None):
        collected_data = {
            "original_inquiry": inquiry,
            "customer_info": {},
            "order_info": {},
            "product_kb_results": {},
        }

        if customer_id:
            collected_data["customer_info"] = self.retrieve_crm_data(customer_id)

        if order_id:
            collected_data["order_info"] = self.retrieve_order_data(order_id)
            if not customer_id and "customer_id" in collected_data["order_info"]:
                collected_data["customer_info"] = self.retrieve_crm_data(collected_data["order_info"]["customer_id"])

        # Simple keyword search for KB for now
        keywords = inquiry.lower().split()
        for keyword in keywords:
            kb_results = self.search_product_kb(keyword)
            if kb_results:
                collected_data["product_kb_results"].update(kb_results)

        return collected_data

class ResolutionPlanner:
    def __init__(self):
        # A simple fake LLM for demonstration. In a real app, replace with OpenAI, Gemini, etc.
        # This LLM will cycle through predefined responses.
        self.llm = FakeListLLM(responses=[
            "Problem: Customer wants to know order status. Plan: 1. Check order_info. 2. Provide status and estimated delivery. Response: Your order #{{order_id}} is currently {{order_status}} and is expected to arrive within 5-7 business days.",
            "Problem: Customer wants to return a product. Plan: 1. Check product_info for return policy. 2. Inform customer about policy and process. Response: For product {{product_name}}, our return policy is {{return_policy}}. To initiate a return, please visit our returns page.",
            "Problem: General inquiry about shipping policy. Plan: 1. Retrieve shipping policy from KB. 2. Explain policy. Response: Our standard shipping policy is as follows: {{shipping_policy}}.",
            "Problem: Issue requires escalation. Plan: 1. Gather all info. 2. Inform customer about escalation. Response: I understand this is important. I'm escalating your case to a specialist who will contact you within 24 hours.",
            "Problem: Customer wants to update address. Plan: 1. Check customer_info. 2. Explain how to update. Response: You can update your shipping address by logging into your account and navigating to the 'My Addresses' section.",
            "Problem: Unclear inquiry. Plan: 1. Ask for clarification. Response: Could you please provide more details about your inquiry? For example, an order ID or specific product name would be helpful."
        ])

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI customer support agent. Analyze the provided customer inquiry and collected information to formulate a clear plan of action and a concise customer-facing response. Prioritize helpfulness and accuracy. If you need more information, ask for it."),
            ("human", "Customer Inquiry: {inquiry}\n\nCollected Information: {collected_info}\n\nFormulate a plan and response based on this information.")
        ])
        self.output_parser = StrOutputParser()
        self.chain = {"inquiry": RunnablePassthrough(), "collected_info": RunnablePassthrough()} | self.prompt | self.llm | self.output_parser

    def plan_and_respond(self, inquiry, collected_info):
        # Format collected_info for the prompt
        formatted_info = "" 
        if collected_info["customer_info"]:
            formatted_info += f"Customer Info: {collected_info['customer_info']}\n"
        if collected_info["order_info"]:
            formatted_info += f"Order Info: {collected_info['order_info']}\n"
        if collected_info["product_kb_results"]:
            formatted_info += f"Product KB Results: {collected_info['product_kb_results']}\n"

        response_and_plan = self.chain.invoke({"inquiry": inquiry, "collected_info": formatted_info})

        # Post-process the fake LLM response to extract plan and response
        # For a real LLM, you'd use a more robust output parser (e.g., PydanticOutputParser)
        plan = "No specific plan extracted (fake LLM limitation)"
        response = response_and_plan
        if "Problem:" in response_and_plan:
            parts = response_and_plan.split("Plan:")
            if len(parts) > 1:
                plan_response_parts = parts[1].split("Response:")
                if len(plan_response_parts) > 1:
                    plan = plan_response_parts[0].strip()
                    response = plan_response_parts[1].strip()
                else:
                    plan = plan_response_parts[0].strip()
                    response = "Please see the plan for details."
            
            # Simple templating for fake LLM
            if collected_info["order_info"] and "order_id" in collected_info["order_info"]:
                response = response.replace("{{order_id}}", collected_info["order_info"]["order_id"])
                response = response.replace("{{order_status}}", collected_info["order_info"]["status"])
            if collected_info["product_kb_results"]:
                for prod_id, prod_details in collected_info["product_kb_results"].items():
                    if "name" in prod_details and "return_policy" in prod_details:
                        response = response.replace("{{product_name}}", prod_details["name"])
                        response = response.replace("{{return_policy}}", prod_details["return_policy"])
            if "shipping_policy" in collected_info["product_kb_results"]:
                response = response.replace("{{shipping_policy}}", collected_info["product_kb_results"]["shipping_policy"])


        return {"plan": plan, "response": response}


class CustomerSupportAgent:
    def __init__(self):
        self.info_collector = InformationCollector()
        self.resolution_planner = ResolutionPlanner()

    def handle_inquiry(self, inquiry, customer_id=None, order_id=None):
        collected_data = self.info_collector.collect_info(inquiry, customer_id, order_id)
        plan_and_response = self.resolution_planner.plan_and_respond(inquiry, collected_data)
        return plan_and_response

if __name__ == "__main__":
    agent = CustomerSupportAgent()

    print("--- Test Case 1: Order Status Inquiry ---")
    inquiry1 = "What is the status of my order #ord_12345?"
    response1 = agent.handle_inquiry(inquiry1, order_id="ord_12345")
    print(f"Inquiry: {inquiry1}")
    print(f"Plan: {response1['plan']}")
    print(f"Response: {response1['response']}")
    print("\n")

    print("--- Test Case 2: Product Return Policy Inquiry ---")
    inquiry2 = "Can I return a Laptop Pro?"
    response2 = agent.handle_inquiry(inquiry2)
    print(f"Inquiry: {inquiry2}")
    print(f"Plan: {response2['plan']}")
    print(f"Response: {response2['response']}")
    print("\n")

    print("--- Test Case 3: General Shipping Inquiry ---")
    inquiry3 = "Tell me about your shipping policy."
    response3 = agent.handle_inquiry(inquiry3)
    print(f"Inquiry: {inquiry3}")
    print(f"Plan: {response3['plan']}")
    print(f"Response: {response3['response']}")
    print("\n")

    print("--- Test Case 4: Complex Inquiry (update address) ---")
    inquiry4 = "I need to update my shipping address for my account."
    response4 = agent.handle_inquiry(inquiry4, customer_id="cust_001")
    print(f"Inquiry: {inquiry4}")
    print(f"Plan: {response4['plan']}")
    print(f"Response: {response4['response']}")
    print("\n")

    print("--- Test Case 5: Unclear Inquiry ---")
    inquiry5 = "I have a problem."
    response5 = agent.handle_inquiry(inquiry5)
    print(f"Inquiry: {inquiry5}")
    print(f"Plan: {response5['plan']}")
    print(f"Response: {response5['response']}")
    print("\n")