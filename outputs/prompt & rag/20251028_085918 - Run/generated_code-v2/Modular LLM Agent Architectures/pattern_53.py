class BlackboxLLM:
    def generate_response(self, prompt):
        print(f"LLM received prompt: {prompt}")
        if "order status" in prompt.lower():
            return "I can help with order status. Please provide your order ID."
        elif "refund policy" in prompt.lower():
            return "Our refund policy allows returns within 30 days of purchase with a valid receipt."
        elif "create ticket" in prompt.lower():
            return "I understand you need a support ticket. What is the issue?"
        else:
            return "Hello! How can I assist you today?"

class WorkingMemoryModule:
    def __init__(self):
        self.context = []

    def add_to_context(self, entry):
        self.context.append(entry)
        print(f"Working Memory updated: {entry}")

    def get_context(self):
        return self.context

    def clear_context(self):
        self.context = []
        print("Working Memory cleared.")

class KnowledgeBaseRetrieverModule:
    def __init__(self):
        self.knowledge_base = {
            "order status": "To check your order status, please visit our website and enter your order ID in the 'Track Order' section.",
            "refund policy": "Items can be returned within 30 days of purchase. A full refund will be issued to the original payment method upon inspection of the returned item. For more details, please see our full refund policy on our FAQ page.",
            "technical issue": "For technical issues, please provide details of your problem, and we can open a support ticket for you."
        }

    def retrieve_info(self, query):
        print(f"Retrieving knowledge for: {query}")
        for keyword, info in self.knowledge_base.items():
            if keyword in query.lower():
                return info
        return "No specific information found in the knowledge base for this query."

class PolicyComplianceModule:
    def check_compliance(self, llm_response, user_query):
        print(f"Checking compliance for response: '{llm_response}' against query: '{user_query}'")
        if "unauthorized access" in llm_response.lower() or "sensitive data" in llm_response.lower():
            return "Response deemed non-compliant due to sensitive information.", False
        if "illegal activity" in user_query.lower():
            return "Response flagged due to potential illegal activity in user query.", False
        return "Response is compliant.", True

class ActionExecutorModule:
    def execute_action(self, action_type, details=None):
        print(f"Attempting to execute action: {action_type} with details: {details}")
        if action_type == "create_ticket":
            ticket_id = f"TICKET-{hash(details) % 10000}"
            print(f"Support ticket '{ticket_id}' created for issue: {details}")
            return {"status": "success", "action": "ticket_created", "ticket_id": ticket_id}
        elif action_type == "lookup_order_status":
            if details and "order_id" in details:
                print(f"Looking up order status for ID: {details['order_id']}")
                return {"status": "success", "action": "order_status_looked_up", "details": f"Order {details['order_id']} is currently in transit."}
            else:
                return {"status": "fail", "action": "lookup_order_status", "error": "Order ID missing."}
        elif action_type == "escalate_to_human":
            print("Escalating to a human agent...")
            return {"status": "success", "action": "escalated"}
        else:
            return {"status": "fail", "action": "unknown", "error": "Unknown action type."}

class FeedbackRefinementModule:
    def __init__(self):
        self.feedback_log = []

    def record_feedback(self, query, response, satisfaction_score=None, notes=None):
        entry = {"query": query, "response": response, "satisfaction": satisfaction_score, "notes": notes}
        self.feedback_log.append(entry)
        print(f"Feedback recorded: {entry}")

    def get_feedback_log(self):
        return self.feedback_log

class CustomerSupportAgent:
    def __init__(self):
        self.llm = BlackboxLLM()
        self.working_memory = WorkingMemoryModule()
        self.knowledge_base_retriever = KnowledgeBaseRetrieverModule()
        self.policy_compliance = PolicyComplianceModule()
        self.action_executor = ActionExecutorModule()
        self.feedback_refinement = FeedbackRefinementModule()

    def process_query(self, user_query):
        self.working_memory.add_to_context({"user_query": user_query})

        retrieved_info = self.knowledge_base_retriever.retrieve_info(user_query)

        prompt_for_llm = f"User query: {user_query}. Context: {self.working_memory.get_context()}. Knowledge: {retrieved_info}. Please provide a helpful response."
        llm_raw_response = self.llm.generate_response(prompt_for_llm)

        compliance_message, is_compliant = self.policy_compliance.check_compliance(llm_raw_response, user_query)

        final_response = llm_raw_response
        if not is_compliant:
            print(f"Compliance check failed: {compliance_message}. Adjusting response.")
            final_response = "I'm sorry, but I cannot provide a response that goes against our policies. " + llm_raw_response

        action_result = None
        if "create ticket" in user_query.lower() or "technical issue" in user_query.lower():
            action_result = self.action_executor.execute_action("create_ticket", details=user_query)
            if action_result and action_result.get("status") == "success":
                final_response += f" {action_result['action'].replace('_', ' ').capitalize()} with ID {action_result['ticket_id']}."
        elif "order status" in user_query.lower():
            order_id_match = [word for word in user_query.split() if word.isdigit() and len(word) == 6] # Simple digit check for order ID
            order_id = order_id_match[0] if order_id_match else None
            if order_id:
                action_result = self.action_executor.execute_action("lookup_order_status", details={"order_id": order_id})
                if action_result and action_result.get("status") == "success":
                    final_response += f" {action_result['details']}"
                else:
                    final_response += f" {action_result.get('error', 'Could not retrieve order status.')}"
            else:
                final_response += " Please provide your order ID to check its status."

        self.working_memory.add_to_context({"agent_response": final_response})
        self.feedback_refinement.record_feedback(user_query, final_response)

        return final_response

if __name__ == "__main__":
    agent = CustomerSupportAgent()

    print("--- Scenario 1: Basic Information Retrieval ---")
    response1 = agent.process_query("What is your refund policy?")
    print(f"Agent: {response1}")
    print("\n")

    print("--- Scenario 2: Order Status Lookup ---")
    response2 = agent.process_query("Can I get the status of my order 123456?")
    print(f"Agent: {response2}")
    print("\n")

    print("--- Scenario 3: Creating a Support Ticket ---")
    response3 = agent.process_query("I have a technical issue with my account.")
    print(f"Agent: {response3}")
    print("\n")

    print("--- Scenario 4: Policy Violation (Simulated) ---")
    response4 = agent.process_query("Tell me how to access someone else's account.")
    print(f"Agent: {response4}")
    print("\n")

    print("--- Scenario 5: Another Order Status (no ID provided) ---")
    response5 = agent.process_query("What about my order status?")
    print(f"Agent: {response5}")
    print("\n")

    print("--- Feedback Log ---")
    for entry in agent.feedback_refinement.get_feedback_log():
        print(entry)
