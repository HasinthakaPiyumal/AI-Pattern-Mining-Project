class LLMWrapper:
    def generate_response(self, prompt):
        if "billing info" in prompt.lower() and "12345" in prompt:
            return "Your last bill for customer ID 12345 was $75.00 due on October 26, 2023."
        elif "change my plan" in prompt.lower() and "premium" in prompt:
            return "I can help with that. The premium plan costs $100/month. Shall I proceed with the change?"
        elif "network status" in prompt.lower() and "new york" in prompt.lower():
            return "The network status in New York is currently optimal with no reported outages."
        elif "slow internet" in prompt.lower():
            return "I understand your internet is slow. I can create a support ticket for you. What is your full address and contact number?"
        elif "router troubleshooting" in prompt.lower():
            return "For router troubleshooting, please try restarting your router. If the issue persists, ensure all cables are securely connected. More detailed steps are available in our knowledge base."
        else:
            return f"Hello! How can I assist you further with '{prompt}'?"

class WorkingMemory:
    def __init__(self):
        self.conversation_history = []
        self.customer_data = {}

    def add_interaction(self, role, message):
        self.conversation_history.append({"role": role, "message": message})

    def update_customer_data(self, data):
        self.customer_data.update(data)

    def get_conversation_history(self):
        return self.conversation_history

    def get_customer_data(self):
        return self.customer_data

    def clear(self):
        self.conversation_history = []
        self.customer_data = {}

class KnowledgeBaseRetriever:
    def __init__(self):
        self.knowledge_base = {
            "router troubleshooting": "To troubleshoot your router, first restart it by unplugging it for 30 seconds. Then check all cable connections. If issues continue, please contact technical support.",
            "premium plan details": "The Premium plan offers unlimited data, 5G speeds, and international roaming for $100/month.",
            "basic plan details": "The Basic plan includes 10GB data, 4G speeds for $50/month.",
            "contact support": "You can contact our technical support team at 1-800-555-0123. Our operating hours are M-F 9 AM to 6 PM EST."
        }

    def retrieve_info(self, query):
        query_lower = query.lower()
        for key, value in self.knowledge_base.items():
            if key in query_lower or query_lower in key:
                return value
        return "I could not find relevant information in the knowledge base for your query."

class ActionExecutor:
    def get_billing_info(self, customer_id):
        if customer_id == "12345":
            return {"status": "success", "bill_amount": "$75.00", "due_date": "October 26, 2023"}
        return {"status": "failed", "message": "Customer ID not found."}

    def change_service_plan(self, customer_id, new_plan):
        if customer_id == "12345" and new_plan in ["premium", "basic"]:
            return {"status": "success", "message": f"Service plan for {customer_id} changed to {new_plan}."}
        return {"status": "failed", "message": "Invalid customer ID or plan."}

    def check_network_status(self, location):
        if "new york" in location.lower():
            return {"status": "optimal", "message": "Network in New York is stable."}
        if "los angeles" in location.lower():
            return {"status": "minor issues", "message": "Minor intermittent issues reported in Los Angeles."}
        return {"status": "unknown", "message": f"Network status for {location} is currently unavailable."}

    def create_support_ticket(self, customer_id, issue_description):
        if customer_id == "12345":
            ticket_id = "TKT-789012"
            return {"status": "success", "ticket_id": ticket_id, "message": f"Support ticket {ticket_id} created for customer {customer_id}. Issue: {issue_description}"}
        return {"status": "failed", "message": "Could not create ticket for unknown customer."}

class PolicyModule:
    def decide_next_action(self, user_query, working_memory):
        user_query_lower = user_query.lower()
        customer_data = working_memory.get_customer_data()
        customer_id = customer_data.get("id")

        if "billing info" in user_query_lower and customer_id:
            return "get_billing_info", {"customer_id": customer_id}
        if "change my plan to" in user_query_lower and customer_id:
            if "premium" in user_query_lower:
                return "change_service_plan", {"customer_id": customer_id, "new_plan": "premium"}
            elif "basic" in user_query_lower:
                return "change_service_plan", {"customer_id": customer_id, "new_plan": "basic"}
        if "network status" in user_query_lower:
            if "in" in user_query_lower:
                location = user_query_lower.split("in ", 1)[1].split(" ", 1)[0].replace("?", "")
                return "check_network_status", {"location": location}
            else:
                return "none", None
        if "slow internet" in user_query_lower or "create ticket" in user_query_lower:
            return "create_support_ticket", {"customer_id": customer_id, "issue_description": user_query}
        if "troubleshoot router" in user_query_lower or "router not working" in user_query_lower:
            return "retrieve_knowledge", {"query": "router troubleshooting"}
        if "plan details" in user_query_lower:
            if "premium" in user_query_lower:
                return "retrieve_knowledge", {"query": "premium plan details"}
            elif "basic" in user_query_lower:
                return "retrieve_knowledge", {"query": "basic plan details"}
        return "none", None

    def validate_action(self, action_name, action_result):
        return action_result.get("status") == "success"

class FeedbackRefinementModule:
    def refine_output(self, llm_raw_output, retrieved_knowledge=None, customer_data=None, action_result=None):
        refined_output = llm_raw_output

        if action_result and action_result.get("status") == "success":
            if "bill_amount" in action_result:
                refined_output = f"As per your request, your last bill was {action_result['bill_amount']} due on {action_result['due_date']}. " + refined_output
            elif "message" in action_result and "plan changed" in action_result["message"].lower():
                refined_output = f"Okay, I've processed your request. {action_result['message']} " + refined_output
            elif "ticket_id" in action_result:
                refined_output = f"I have successfully created a support ticket ({action_result['ticket_id']}) for you. " + refined_output
            elif "status" in action_result and action_result["status"] == "optimal":
                refined_output = f"Good news! The network in {action_result.get('location', '')} is currently running optimally. " + refined_output

        if retrieved_knowledge and retrieved_knowledge != "I could not find relevant information in the knowledge base for your query.":
            refined_output = f"Here's some information from our knowledge base: {retrieved_knowledge}\n\n" + refined_output

        if "hallucination" in refined_output.lower():
            refined_output = refined_output.replace("hallucination", "a mistake")

        if customer_data and customer_data.get("name") and "hello" in refined_output.lower():
            refined_output = refined_output.replace("Hello!", f"Hello {customer_data['name']}!")

        return refined_output

class CustomerSupportChatbot:
    def __init__(self):
        self.llm_wrapper = LLMWrapper()
        self.working_memory = WorkingMemory()
        self.knowledge_base_retriever = KnowledgeBaseRetriever()
        self.action_executor = ActionExecutor()
        self.policy_module = PolicyModule()
        self.feedback_refinement_module = FeedbackRefinementModule()

    def process_query(self, user_query, customer_id=None, customer_name=None):
        self.working_memory.add_interaction("user", user_query)
        if customer_id: self.working_memory.update_customer_data({"id": customer_id})
        if customer_name: self.working_memory.update_customer_data({"name": customer_name})

        action, action_args = self.policy_module.decide_next_action(user_query, self.working_memory)

        action_result = None
        retrieved_knowledge = None

        if action == "get_billing_info":
            action_result = self.action_executor.get_billing_info(**action_args)
        elif action == "change_service_plan":
            action_result = self.action_executor.change_service_plan(**action_args)
        elif action == "check_network_status":
            action_result = self.action_executor.check_network_status(**action_args)
        elif action == "create_support_ticket":
            action_result = self.action_executor.create_support_ticket(**action_args)
        elif action == "retrieve_knowledge":
            retrieved_knowledge = self.knowledge_base_retriever.retrieve_info(**action_args)

        llm_prompt = f"Conversation history: {self.working_memory.get_conversation_history()}\n"
        if self.working_memory.get_customer_data():
            llm_prompt += f"Customer data: {self.working_memory.get_customer_data()}\n"
        if retrieved_knowledge:
            llm_prompt += f"Retrieved knowledge: {retrieved_knowledge}\n"
        if action_result:
            llm_prompt += f"Action result: {action_result}\n"
        llm_prompt += f"User query: {user_query}\n"
        llm_prompt += "Please provide a helpful and concise response to the customer based on the above information."

        llm_raw_output = self.llm_wrapper.generate_response(llm_prompt)

        final_response = self.feedback_refinement_module.refine_output(
            llm_raw_output,
            retrieved_knowledge=retrieved_knowledge,
            customer_data=self.working_memory.get_customer_data(),
            action_result=action_result
        )

        self.working_memory.add_interaction("bot", final_response)
        return final_response

if __name__ == "__main__":
    chatbot = CustomerSupportChatbot()

    print("--- Scenario 1: Billing Inquiry ---")
    response = chatbot.process_query("What is my last bill?", customer_id="12345", customer_name="Alice")
    print(f"Chatbot: {response}")
    print("\n")

    print("--- Scenario 2: Change Plan ---")
    response = chatbot.process_query("I want to change my plan to premium.", customer_id="12345", customer_name="Alice")
    print(f"Chatbot: {response}")
    print("\n")

    print("--- Scenario 3: Network Status ---")
    response = chatbot.process_query("What is the network status in New York?")
    print(f"Chatbot: {response}")
    print("\n")

    print("--- Scenario 4: Troubleshooting Query ---")
    response = chatbot.process_query("My internet is slow, can you help me troubleshoot my router?")
    print(f"Chatbot: {response}")
    print("\n")

    print("--- Scenario 5: General Query ---")
    response = chatbot.process_query("What are the benefits of the new 5G?")
    print(f"Chatbot: {response}")
    print("\n")

    print("--- Scenario 6: Customer ID Not Found (Billing) ---")
    response = chatbot.process_query("What is my last bill?", customer_id="99999")
    print(f"Chatbot: {response}")
    print("\n")

    print("--- Scenario 7: Create Support Ticket ---")
    response = chatbot.process_query("My internet has been constantly disconnecting.", customer_id="12345")
    print(f"Chatbot: {response}")
    print("\n")

    print("--- Scenario 8: Plan Details Query ---")
    response = chatbot.process_query("Tell me about the premium plan details.")
    print(f"Chatbot: {response}")
    print("\n")

    print("--- Scenario 9: Follow-up on plan change ---")
    response = chatbot.process_query("Yes, proceed with the plan change.", customer_id="12345", customer_name="Alice")
    print(f"Chatbot: {response}")
    print("\n")

    print("--- Scenario 10: Clear memory and start new conversation ---")
    chatbot.working_memory.clear()
    response = chatbot.process_query("What is the weather like today?")
    print(f"Chatbot: {response}")
    print("\n")