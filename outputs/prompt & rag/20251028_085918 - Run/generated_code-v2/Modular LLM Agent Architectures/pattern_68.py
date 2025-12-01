import os

class CRMSimulator:
    def get_customer_history(self, customer_id):
        if customer_id == "CUST001":
            return {"customer_id": "CUST001", "name": "Alice Smith", "last_interaction": "Product inquiry about X on 2023-10-26", "membership_level": "Gold"}
        return {"customer_id": customer_id, "name": "Unknown Customer", "last_interaction": "N/A", "membership_level": "Standard"}

class KnowledgeBaseSimulator:
    def search_faq(self, query):
        query = query.lower()
        if "reset password" in query:
            return "To reset your password, visit our website and click 'Forgot Password' in the login section. Follow the prompts to set a new one."
        elif "shipping policy" in query:
            return "Our standard shipping takes 3-5 business days. Expedited options are available at checkout. Please refer to our shipping policy page for full details."
        return "No direct FAQ found for your query. Please provide more details."

class OrderManagementSimulator:
    def get_order_details(self, order_id):
        if order_id == "ORD87654":
            return {"order_id": "ORD87654", "customer_id": "CUST001", "status": "Shipped", "item": "Product X", "tracking_number": "TN987654321"}
        return {"order_id": order_id, "status": "Not Found", "item": "N/A", "tracking_number": "N/A"}

class ExternalSearchSimulator:
    def search_web(self, query):
        return f"Simulated web search result for '{query}': Best practices usually suggest checking product manuals or official support forums."

class LLMSimulator:
    def generate_response(self, prompt, role="assistant"):
        if "summarize" in prompt.lower():
            return f"Summary based on data: {prompt.split('DATA:')[-1].strip()}"
        elif "plan solution" in prompt.lower():
            return f"Planned solution based on summary: {prompt.split('SUMMARY:')[-1].strip()}. Consider standard operating procedures."
        elif "generate response" in prompt.lower():
            return f"Customer response: We understand your concern. Based on the information... {prompt.split('SOLUTION:')[-1].strip()}"
        return "Simulated LLM response."

class InformationCollectorAgent:
    def __init__(self):
        self.crm = CRMSimulator()
        self.kb = KnowledgeBaseSimulator()
        self.oms = OrderManagementSimulator()
        self.external_search = ExternalSearchSimulator()
        self.llm = LLMSimulator()
        self.memory = [] # Simple list for memory

    def run(self, query, customer_id=None, order_id=None):
        self.memory.append(f"Customer query: {query}")
        collected_info = []

        # Try to extract IDs from query if not provided
        if not customer_id and "customer id" in query.lower():
            parts = query.lower().split("customer id")
            if len(parts) > 1: customer_id = parts[1].strip().split()[0]

        if not order_id and "order id" in query.lower():
            parts = query.lower().split("order id")
            if len(parts) > 1: order_id = parts[1].strip().split()[0]

        if customer_id:
            crm_data = self.crm.get_customer_history(customer_id)
            collected_info.append(f"CRM Data: {crm_data}")

        if order_id:
            oms_data = self.oms.get_order_details(order_id)
            collected_info.append(f"OMS Data: {oms_data}")

        kb_result = self.kb.search_faq(query)
        collected_info.append(f"Knowledge Base: {kb_result}")

        if not crm_data and not oms_data and "general" in query.lower():
             external_result = self.external_search.search_web(query)
             collected_info.append(f"External Search: {external_result}")

        self.memory.append(f"Collected Info: {collected_info}")
        return collected_info

class InformationSummarizer:
    def __init__(self):
        self.llm = LLMSimulator()

    def summarize(self, raw_info):
        raw_info_str = "\n".join(raw_info)
        prompt = f"Summarize the following raw customer data, highlighting key facts: DATA: {raw_info_str}"
        summary = self.llm.generate_response(prompt, role="summarizer")
        return summary

class PolicyAdherenceModule:
    def check_policy(self, solution_plan):
        if "refund" in solution_plan.lower() and "damaged item" not in solution_plan.lower():
            return "Policy Violation: Refunds usually require proof of damage or specific conditions."
        if "discount" in solution_plan.lower() and "gold membership" not in solution_plan.lower():
            return "Policy Note: Discounts are typically offered to Gold members or for specific promotions."
        return "Policy check: OK."

class ProblemAnalyzerPlannerAgent:
    def __init__(self):
        self.llm = LLMSimulator()
        self.policy_module = PolicyAdherenceModule()

    def run(self, summarized_info, original_query=""):
        prompt = f"Given the following summarized information and the original query '{original_query}', plan a solution, considering standard customer support protocols. SUMMARY: {summarized_info}"
        solution_plan = self.llm.generate_response(prompt, role="planner")

        policy_check_result = self.policy_module.check_policy(solution_plan)

        if "Policy Violation" in policy_check_result:
            # Agent might need to re-plan or escalate based on policy violation
            return f"Solution Plan: {solution_plan} (Policy Alert: {policy_check_result})"
        return f"Solution Plan: {solution_plan} (Policy check: {policy_check_result})"

class ResponseGeneratorAgent:
    def __init__(self):
        self.llm = LLMSimulator()

    def run(self, solution_plan, original_query):
        prompt = f"Based on the customer's original query '{original_query}' and the planned solution, generate a polite, comprehensive, and empathetic customer response. SOLUTION: {solution_plan}"
        customer_response = self.llm.generate_response(prompt, role="responder")
        return customer_response

class CustomerSupportOrchestrator:
    def __init__(self):
        self.info_collector = InformationCollectorAgent()
        self.summarizer = InformationSummarizer()
        self.planner = ProblemAnalyzerPlannerAgent()
        self.responder = ResponseGeneratorAgent()

    def handle_query(self, customer_query):
        print(f"\n--- Customer Query Received ---")
        print(f"Query: {customer_query}")

        # Stage 1: Information Gathering
        print(f"\n--- Stage 1: Information Gathering ---")
        raw_collected_info = self.info_collector.run(customer_query)
        print(f"Raw Collected Info: {raw_collected_info}")

        summarized_info = self.summarizer.summarize(raw_collected_info)
        print(f"Summarized Info: {summarized_info}")

        # Stage 2: Planning and Response Generation
        print(f"\n--- Stage 2: Planning and Response Generation ---")
        solution_plan = self.planner.run(summarized_info, customer_query)
        print(f"Solution Plan: {solution_plan}")

        final_response = self.responder.run(solution_plan, customer_query)
        print(f"\n--- Final Customer Response ---")
        print(f"Response: {final_response}")
        return final_response

if __name__ == "__main__":
    orchestrator = CustomerSupportOrchestrator()

    # Test Case 1: Product inquiry with customer ID
    orchestrator.handle_query("My Product X is not working. I'm customer CUST001.")

    # Test Case 2: Order status inquiry
    orchestrator.handle_query("What is the status of my order ORD87654?")

    # Test Case 3: General FAQ about password reset
    orchestrator.handle_query("How do I reset my password?")

    # Test Case 4: Query potentially leading to policy violation (refund without damage)
    orchestrator.handle_query("I want a refund for Product Y. It's not what I expected. I'm customer CUST001.")

    # Test Case 5: Query for an unknown customer and order
    orchestrator.handle_query("My gadget isn't turning on. My customer id is CUST999 and order id is ORD00000.")
