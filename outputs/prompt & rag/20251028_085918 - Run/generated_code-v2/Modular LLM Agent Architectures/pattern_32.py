class MockLLM:
    """A mock LLM to simulate responses for demonstration purposes."""
    def __init__(self, responses=None):
        self.responses = responses if responses is not None else {}
        self.call_count = 0

    def invoke(self, prompt, stop_sequences=None):
        self.call_count += 1
        print(f"--- LLM Call {self.call_count} ---")
        print(f"Prompt: {prompt[:200]}...") # Print a snippet of the prompt
        
        # Simulate dynamic responses based on keywords in the prompt
        if "information collection" in prompt.lower() or "clarifying questions" in prompt.lower():
            if "order status" in prompt.lower() and "#12345" in prompt:
                return "Identified need for order status. Order ID: #12345."
            elif "product details" in prompt.lower() and "wireless headphones" in prompt.lower():
                return "Identified need for product details. Product: Wireless Headphones."
            elif "refund policy" in prompt.lower():
                return "Identified need for refund policy information."
            else:
                return "Understood. I need to gather more information. What is the customer's primary concern?"
        elif "solution planning" in prompt.lower() or "formulate a plan" in prompt.lower():
            if "refund" in prompt.lower() and "#12345" in str(self.responses):
                return "Solution: Initiate a full refund for order #12345, provide return label and explain the 30-day refund policy."
            elif "product information" in prompt.lower() and "wireless headphones" in str(self.responses):
                return "Solution: Provide detailed specifications and a link to the user manual for the Wireless Headphones. Offer troubleshooting steps for common issues."
            else:
                return "Solution: Based on the gathered information, I recommend checking our FAQ for common issues related to this problem and offering further assistance."
        return "Generic LLM response."

class ExternalTools:
    """Simulates external systems for retrieving data like order history or product details."""
    def get_order_history(self, customer_id=None, order_id=None):
        print(f"--- Calling get_order_history(customer_id={customer_id}, order_id={order_id}) ---")
        if order_id == "#12345":
            return {"order_id": "#12345", "status": "shipped", "items": ["Wireless Headphones", "Charging Cable"], "purchase_date": "2023-10-26"}
        return {"error": "Order not found."}

    def get_product_details(self, product_name=None, sku=None):
        print(f"--- Calling get_product_details(product_name={product_name}, sku={sku}) ---")
        if product_name and "wireless headphones" in product_name.lower():
            return {"name": "Wireless Headphones", "SKU": "WH100", "price": 49.99, "features": ["Noise Cancelling", "Bluetooth 5.0", "20-hour battery"]}
        return {"error": "Product not found."}

    def get_faq_articles(self, query):
        print(f"--- Calling get_faq_articles(query='{query}') ---")
        if "refund policy" in query.lower() or "return policy" in query.lower():
            return "FAQ: Our refund policy allows returns within 30 days of purchase for a full refund, provided the item is in its original condition."
        return "FAQ: No relevant articles found for your query."

class DataGatheringModule:
    """Stage 1: Focuses on collecting and clarifying information from the customer and external tools."""
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools

    def process_query(self, customer_query: str) -> dict:
        print("\n--- Stage 1: Data Gathering Module ---")
        
        # Step 1: LLM for initial understanding and identifying data needs
        initial_analysis_prompt = f"""
        You are a customer support assistant. Your primary goal is to gather all necessary information to understand the customer's problem thoroughly.
        The customer's initial query is: "{customer_query}"

        Based on this, what specific pieces of information do you need to gather? 
        Think step-by-step to identify key entities (e.g., product names, order IDs) and determine which tools (get_order_history, get_product_details, get_faq_articles) might be relevant.
        After identifying the needs, list the identified entities and required data.
        """
        llm_analysis = self.llm.invoke(initial_analysis_prompt, responses=customer_query) # Pass query for mock LLM to use
        print(f"LLM Initial Analysis: {llm_analysis}")
        
        # Extract entities based on common patterns or simple keyword checks for this example
        extracted_order_id = next((word for word in customer_query.split() if word.startswith('#')), None)
        extracted_product_name = None
        if "headphones" in customer_query.lower():
            extracted_product_name = "Wireless Headphones"

        collected_data = {}
        if extracted_order_id:
            collected_data["order_history"] = self.tools.get_order_history(order_id=extracted_order_id)
        if extracted_product_name:
            collected_data["product_details"] = self.tools.get_product_details(product_name=extracted_product_name)
        if "refund" in customer_query.lower() or "return" in customer_query.lower():
            collected_data["faq_refund_policy"] = self.tools.get_faq_articles("refund policy")

        # Step 2: LLM to summarize collected information into a structured problem summary
        summary_prompt = f"""
        You have processed the customer query: "{customer_query}"
        And gathered the following relevant data:
        {collected_data}

        Please provide a concise, structured summary of the customer's problem, incorporating the collected data.
        This summary will be used by a solution planning module, so it should be clear and complete.
        Focus on identifying the core issue and any constraints or important context.

        Example Output Structure (you should output similar content, not JSON directly):
        "The customer's primary concern is X. Relevant order information: Y. Product details: Z. Current policy: A."
        """
        final_summary = self.llm.invoke(summary_prompt, responses=collected_data) # Pass collected data for mock LLM
        print(f"LLM Final Summary (Stage 1): {final_summary}")

        return {
            "customer_query": customer_query,
            "problem_statement_summary": final_summary, # Use LLM's summarized statement
            "collected_raw_data": collected_data,
            "identified_constraints": [] # In a real system, LLM would help populate this
        }

class ResolutionPlanningModule:
    """Stage 2: Takes the structured problem summary and formulates a solution plan."""
    def __init__(self, llm):
        self.llm = llm

    def plan_and_execute(self, structured_problem_summary: dict) -> str:
        print("\n--- Stage 2: Resolution Planning Module ---")
        
        plan_prompt = f"""
        You are a customer support agent. Your task is to formulate a clear and actionable resolution plan based on the following structured problem summary.
        This summary has already been thoroughly gathered and clarified.

        Structured Problem Summary: {structured_problem_summary['problem_statement_summary']}
        Relevant Raw Data: {structured_problem_summary['collected_raw_data']}

        Consider the following:
        1. Based on the summary and data, diagnose the root cause (if inferable).
        2. Identify potential solutions or next steps, considering any policies or product specifications.
        3. Formulate a step-by-step plan or direct answer for the customer.
        4. State the final proposed resolution clearly and professionally.

        Keep the response concise, customer-friendly, and actionable.
        """
        resolution = self.llm.invoke(plan_prompt, responses=structured_problem_summary) # Pass summary for mock LLM
        print(f"LLM Resolution Plan (Stage 2): {resolution}")
        return resolution

class CustomerSupportAgent:
    """Orchestrates the two-stage customer support agent workflow."""
    def __init__(self):
        self.mock_llm = MockLLM()
        self.external_tools = ExternalTools()
        self.data_gathering_module = DataGatheringModule(self.mock_llm, self.external_tools)
        self.resolution_planning_module = ResolutionPlanningModule(self.mock_llm)

    def handle_inquiry(self, customer_query: str) -> str:
        print(f"\n--- Handling New Inquiry: '{customer_query}' ---")
        # Stage 1: Information Collection
        structured_summary = self.data_gathering_module.process_query(customer_query)
        
        # Stage 2: Solution Planning and Execution
        final_resolution = self.resolution_planning_module.plan_and_execute(structured_summary)
        
        return f"Agent Final Response: {final_resolution}"

# Example Usage
if __name__ == "__main__":
    agent = CustomerSupportAgent()

    print("\n--- Test Case 1: Order Status Inquiry ---")
    response1 = agent.handle_inquiry("What is the status of my order #12345?")
    print(response1)

    print("\n--- Test Case 2: Product Information Inquiry ---")
    response2 = agent.handle_inquiry("Can you tell me more about the Wireless Headphones?")
    print(response2)

    print("\n--- Test Case 3: Refund Policy Inquiry with Order ID ---")
    response3 = agent.handle_inquiry("I want a refund for my order #12345. What's your policy?")
    print(response3)

    print("\n--- Test Case 4: General Inquiry ---")
    response4 = agent.handle_inquiry("My product isn't working.")
    print(response4)
