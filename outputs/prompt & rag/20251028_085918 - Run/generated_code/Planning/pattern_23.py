class KnowledgeBase:
    def __init__(self):
        self.policies = {
            "return_policy": "Items can be returned within 30 days of purchase, new condition, with receipt.",
            "exchange_policy": "Exchanges are subject to availability and item condition, within 30 days.",
            "warranty_info": "Most electronics have a 1-year manufacturer warranty."
        }
        self.orders = {
            "ORD123": {"customer_id": "CUST001", "items": [{"id": "PROD001", "name": "Smartwatch", "price": 199.99, "status": "delivered"}], "date": "2023-10-15"},
            "ORD124": {"customer_id": "CUST002", "items": [{"id": "PROD002", "name": "Headphones", "price": 99.00, "status": "shipped"}], "date": "2023-11-01"}
        }
        self.inventory = {
            "PROD001": {"name": "Smartwatch", "stock": 50},
            "PROD002": {"name": "Headphones", "stock": 120}
        }

    def get_policy(self, policy_name):
        return self.policies.get(policy_name, "Policy not found.")

    def get_order_details(self, order_id):
        return self.orders.get(order_id)

    def get_inventory(self, product_id):
        return self.inventory.get(product_id)


class LLMInterface:
    def __init__(self):
        pass

    def decompose_task(self, request):
        if "return" in request.lower():
            return ["Identify order details", "Check return policy", "Process return request"]
        elif "exchange" in request.lower():
            return ["Identify order details", "Check exchange policy", "Check new item availability", "Process exchange request"]
        else:
            return ["Understand request", "Provide general information"]

    def generate_plan(self, tasks):
        return {"steps": tasks, "status": "initial"}

    def introspect_plan(self, plan):
        if "Identify order details" in plan["steps"] and "Check return policy" in plan["steps"]:
            return "Plan looks reasonable for a return. Needs order ID and item details."
        return "Plan seems okay, but consider more context."

    def refine_plan_with_feedback(self, plan, feedback):
        if "missing order ID" in feedback.lower() and "Identify order details" in plan["steps"]:
            plan["steps"].insert(plan["steps"].index("Identify order details") + 1, "Ask customer for order ID")
            return plan
        return plan

    def identify_constraints(self, request, context):
        constraints = []
        if "return" in request.lower() and "30 days" not in context.get("policy", "").lower():
            constraints.append("Return window (e.g., 30 days) must be satisfied.")
        if "exchange" in request.lower() and "availability" not in context.get("policy", "").lower():
            constraints.append("New item availability is crucial for exchange.")
        return constraints

    def refine_prompt(self, prompt, context):
        if "order ID needed" in context.get("feedback", "").lower():
            return f"{prompt} Please provide your order ID."
        return prompt


class PlanningModule:
    def __init__(self, llm_interface, knowledge_base):
        self.llm_interface = llm_interface
        self.knowledge_base = knowledge_base

    def create_plan(self, request):
        tasks = self.llm_interface.decompose_task(request)
        plan = self.llm_interface.generate_plan(tasks)
        return plan

    def optimize_plan(self, plan):
        optimized_steps = []
        for step in plan["steps"]:
            if step == "Identify order details" and "order_id" in plan.get("context", {}):
                continue
            optimized_steps.append(step)
        plan["steps"] = optimized_steps
        return plan

    def execute_subtask(self, subtask, context):
        if subtask == "Identify order details":
            order_id = context.get("order_id")
            if order_id:
                order = self.knowledge_base.get_order_details(order_id)
                if order:
                    context["order_info"] = order
                    return f"Found order {order_id}. Items: {[item['name'] for item in order['items']]}.", "success"
                else:
                    return f"Order {order_id} not found.", "failure", {"feedback": "missing order ID details"}
            return "Please provide the order ID.", "pending", {"feedback": "missing order ID"}

        elif subtask == "Check return policy":
            policy = self.knowledge_base.get_policy("return_policy")
            context["policy"] = policy
            return f"Return Policy: {policy}", "success"

        elif subtask == "Check exchange policy":
            policy = self.knowledge_base.get_policy("exchange_policy")
            context["policy"] = policy
            return f"Exchange Policy: {policy}", "success"

        elif subtask == "Check new item availability":
            item_id = context.get("desired_exchange_item_id") or context.get("order_info", {}).get("items", [{}])[0].get("id") # Simplified for demo
            if item_id:
                inventory_info = self.knowledge_base.get_inventory(item_id)
                if inventory_info and inventory_info["stock"] > 0:
                    context["available"] = True
                    return f"Item {inventory_info['name']} is available. Stock: {inventory_info['stock']}.", "success"
                else:
                    context["available"] = False
                    return f"Item {item_id} is out of stock or not found.", "failure"
            return "Need item ID for availability check.", "pending"

        elif subtask == "Process return request":
            order_info = context.get("order_info")
            if order_info and "return_policy" in context.get("policy", "").lower() and "eligible" in context.get("status_check", "eligible").lower(): # Simplified check
                return f"Return initiated for order {order_info['items'][0]['name']}. Please follow instructions.", "success"
            return "Cannot process return. Insufficient information or policy violation.", "failure"

        elif subtask == "Process exchange request":
            order_info = context.get("order_info")
            if order_info and context.get("available") and "exchange_policy" in context.get("policy", "").lower(): # Simplified check
                return f"Exchange initiated for order {order_info['items'][0]['name']} for new item. Details sent to email.", "success"
            return "Cannot process exchange. Item not available or policy violation.", "failure"

        elif subtask == "Understand request":
            return "I understand you have a general query. How can I assist further?", "success"

        elif subtask == "Provide general information":
            return "I can help with returns, exchanges, and general product information.", "success"

        elif subtask == "Ask customer for order ID":
            return "Could you please provide your order ID?", "pending", {"feedback": "waiting for user input"}

        return f"Unknown subtask: {subtask}", "failure"


class CustomerSupportAgent:
    def __init__(self):
        self.knowledge_base = KnowledgeBase()
        self.llm_interface = LLMInterface()
        self.planning_module = PlanningModule(self.llm_interface, self.knowledge_base)
        self.conversation_history = []

    def handle_request(self, customer_request, user_context=None):
        self.conversation_history.append(f"Customer: {customer_request}")
        context = user_context if user_context is not None else {}

        print(f"Agent: Received request: {customer_request}")

        # 1. Task Decomposition & Initial Plan Generation
        initial_plan = self.planning_module.create_plan(customer_request)
        print(f"Agent: Initial Plan: {initial_plan['steps']}")

        # 2. Simulate Introspective Reasoning
        introspection_feedback = self.llm_interface.introspect_plan(initial_plan)
        print(f"Agent: Introspection: {introspection_feedback}")

        # 3. Simulate Constraint Identification
        constraints = self.llm_interface.identify_constraints(customer_request, context)
        if constraints:
            print(f"Agent: Identified constraints: {constraints}")

        current_plan = initial_plan
        response_messages = []
        status_overall = "pending"

        for step in list(current_plan["steps"]):
            print(f"Agent: Executing step: {step}")
            message, step_status, feedback_data = self.planning_module.execute_subtask(step, context)
            response_messages.append(f" - {message}")
            print(f"Agent: Step Result: {step_status} - {message}")

            if step_status == "pending":
                # Simulate waiting for user input or more data
                status_overall = "waiting_for_input"
                if feedback_data: # If an explicit request for more data is made
                    if "order ID" in feedback_data.get("feedback", "").lower() and "order_id" not in context:
                        print(f"Agent: Waiting for customer to provide Order ID.")
                        return "Please provide your Order ID to proceed."
                break

            elif step_status == "failure":
                print(f"Agent: Step failed. Refining plan with feedback.")
                current_plan = self.llm_interface.refine_plan_with_feedback(current_plan, feedback_data.get("feedback", ""))
                response_messages.append(f"Agent: Encountered an issue. Attempting to re-plan.")
                # For a simple demo, we'll just stop on first failure that needs re-planning for simplicity
                status_overall = "failed"
                break
            elif step_status == "success":
                # After a successful step, optimize the plan (e.g., remove completed tasks)
                current_plan = self.planning_module.optimize_plan(current_plan)


        if status_overall == "waiting_for_input":
            final_response = "\n".join(response_messages)
        elif status_overall == "failed":
            final_response = "\n".join(response_messages) + "\nAgent: I'm sorry, I encountered an issue and couldn't complete your request. Please try again or contact a human agent."
        else:
            status_overall = "completed"
            final_response = "\n".join(response_messages) + "\nAgent: Your request has been processed. Is there anything else I can help you with?"

        self.conversation_history.append(f"Agent: {final_response}")
        return final_response


def simulate_customer_interaction():
    agent = CustomerSupportAgent()
    print("\nWelcome to E-commerce Customer Support. How can I help you today?")

    # Scenario 1: Simple return request (needs order ID)
    print("\n--- Scenario 1: Return Request without Order ID ---")
    response = agent.handle_request("I want to return a product.")
    print(f"Customer Support Agent: {response}")

    # Simulate user providing order ID
    if "Order ID" in response:
        print("\nCustomer: My order ID is ORD123.")
        context_with_order = {"order_id": "ORD123"}
        response = agent.handle_request("My order ID is ORD123.", user_context=context_with_order)
        print(f"Customer Support Agent: {response}")

    # Scenario 2: Exchange request
    print("\n--- Scenario 2: Exchange Request ---")
    response = agent.handle_request("I want to exchange my smartwatch for a new one.", user_context={"order_id": "ORD123", "desired_exchange_item_id": "PROD001"})
    print(f"Customer Support Agent: {response}")

    # Scenario 3: General query
    print("\n--- Scenario 3: General Query ---")
    response = agent.handle_request("What can you do?")
    print(f"Customer Support Agent: {response}")

if __name__ == "__main__":
    simulate_customer_interaction()
