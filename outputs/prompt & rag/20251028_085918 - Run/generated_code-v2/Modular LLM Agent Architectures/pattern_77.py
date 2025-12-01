class SimulatedLLM:
    def generate_response(self, prompt, context):
        combined_input = f"Customer Query: {prompt}\nContext: {context}"
        if "order status" in prompt.lower():
            return "Simulated LLM response: I can help with that! Let me check your order status. What is your order ID?"
        elif "reset password" in prompt.lower():
            return "Simulated LLM response: To reset your password, please visit our website and click 'Forgot Password'."
        elif "product details" in prompt.lower() or "tell me about" in prompt.lower():
            return "Simulated LLM response: Please specify which product you are interested in, and I can retrieve details from our knowledge base."
        elif "technical issue" in prompt.lower() or "bug" in prompt.lower():
            return "Simulated LLM response: I understand you are experiencing a technical issue. I can create a support ticket for you."
        elif "escalate" in prompt.lower() or "human agent" in prompt.lower():
            return "Simulated LLM response: It seems this requires human intervention. I will escalate this to a human agent."
        return f"Simulated LLM response: I'm processing your request based on the following: {combined_input}. How else can I assist you?"

class WorkingMemoryModule:
    def __init__(self, customer_db, product_db):
        self.customer_db = customer_db
        self.product_db = product_db

    def get_context(self, customer_id):
        customer_info = self.customer_db.get(customer_id, {})
        conversation_history = customer_info.get("conversation_history", [])
        return {
            "customer_info": customer_info,
            "conversation_history": conversation_history,
        }

    def update_context(self, customer_id, new_entry):
        if customer_id not in self.customer_db:
            self.customer_db[customer_id] = {"id": customer_id, "conversation_history": []}
        self.customer_db[customer_id]["conversation_history"].append(new_entry)

class PolicyModule:
    def __init__(self, rules):
        self.rules = rules

    def apply_policy(self, llm_response, context):
        refined_response = llm_response
        suggested_action = None

        if "escalate to a human agent" in llm_response.lower() or "human intervention" in llm_response.lower():
            suggested_action = {"type": "escalate_to_human"}
            refined_response = "I have detected a need for human assistance. Initiating escalation to a human agent."
        
        for rule in self.rules:
            if rule["condition"] in refined_response.lower():
                refined_response = rule["action"](refined_response, context) 
                if "suggested_action" in rule:
                    suggested_action = rule["suggested_action"]

        return refined_response, suggested_action

class ActionExecutorModule:
    def __init__(self, crm_api, knowledge_base_api, ticketing_system_api):
        self.crm_api = crm_api
        self.knowledge_base_api = knowledge_base_api
        self.ticketing_system_api = ticketing_system_api

    def _check_order_status(self, order_id):
        print(f"Simulating CRM API: Checking order status for {order_id}")
        if order_id == "ORDER123":
            return "Order ORDER123 is currently in transit and expected by Friday."
        return "Order not found or invalid ID."

    def _search_faq(self, query):
        print(f"Simulating Knowledge Base API: Searching FAQ for '{query}'")
        if "password" in query.lower():
            return "FAQ: To reset your password, visit our account settings page and click 'Forgot Password'."
        return "No relevant FAQ found for your query."
    
    def _create_ticket(self, customer_id, issue_description):
        print(f"Simulating Ticketing System API: Creating ticket for customer {customer_id} with issue: {issue_description}")
        ticket_id = f"TICKET{hash(customer_id + issue_description) % 10000}"
        return f"Support ticket {ticket_id} has been created for your issue. A specialist will contact you shortly."

    def execute_action(self, action_type, **kwargs):
        if action_type == "check_order_status":
            return self._check_order_status(kwargs.get("order_id"))
        elif action_type == "search_faq":
            return self._search_faq(kwargs.get("query"))
        elif action_type == "create_ticket":
            return self._create_ticket(kwargs.get("customer_id"), kwargs.get("issue_description"))
        else:
            return f"Unknown action type: {action_type}"

class UtilityModule:
    def evaluate_response(self, llm_response, context):
        confidence_score = 0.95
        sentiment = "neutral"
        flags = []

        if "apologies for the confusion" in llm_response.lower() or "i'm not sure" in llm_response.lower():
            confidence_score = 0.6
        
        if "fictional account" in llm_response.lower(): # Simple hallucination detection
            flags.append("potential_hallucination")
            confidence_score = 0.4
        
        if "happy to help" in llm_response.lower():
            sentiment = "positive"
        elif "unacceptable" in llm_response.lower():
            sentiment = "negative"

        return {"confidence": confidence_score, "sentiment": sentiment, "flags": flags}

    def identify_intent(self, query):
        if "order status" in query.lower():
            return "check_order_status"
        elif "password reset" in query.lower() or "forgot password" in query.lower():
            return "password_reset"
        elif "product info" in query.lower() or "details about" in query.lower():
            return "get_product_info"
        elif "technical issue" in query.lower() or "bug" in query.lower():
            return "report_technical_issue"
        elif "speak to human" in query.lower() or "escalate" in query.lower():
            return "escalate_to_human"
        return "general_query"

class LLMAgent:
    def __init__(self, llm_model, working_memory_module, policy_module, action_executor_module, utility_module):
        self.llm_model = llm_model
        self.working_memory = working_memory_module
        self.policy_module = policy_module
        self.action_executor = action_executor_module
        self.utility_module = utility_module

    def process_query(self, customer_id, query):
        self.working_memory.update_context(customer_id, {"speaker": "customer", "text": query})
        context = self.working_memory.get_context(customer_id)

        prompt_for_llm = f"You are a customer support agent. Here is the customer's query and their history:\nCustomer Query: {query}\nConversation History: {context['conversation_history'][-5:]}\nCustomer Info: {context['customer_info']}\nGenerate a helpful response and suggest any necessary actions."

        initial_llm_response = self.llm_model.generate_response(prompt_for_llm, context)
        print(f"LLM Raw Response: {initial_llm_response}")

        refined_response, suggested_action = self.policy_module.apply_policy(initial_llm_response, context)
        print(f"Policy Refined Response: {refined_response}, Suggested Action: {suggested_action}")

        action_result = None
        if suggested_action:
            if suggested_action["type"] == "check_order_status":
                # This would ideally extract order_id from the query or context
                order_id_from_query = "ORDER123" # Placeholder
                action_result = self.action_executor.execute_action("check_order_status", order_id=order_id_from_query)
                refined_response = f"{refined_response}\nAction Result: {action_result}"
            elif suggested_action["type"] == "create_ticket":
                action_result = self.action_executor.execute_action("create_ticket", customer_id=customer_id, issue_description=query)
                refined_response = f"{refined_response}\nAction Result: {action_result}"
            elif suggested_action["type"] == "escalate_to_human":
                action_result = self.action_executor.execute_action("create_ticket", customer_id=customer_id, issue_description=f"Escalated issue from LLM: {query}")
                refined_response = f"I am escalating your request to a human agent. {action_result}"
        
        evaluation = self.utility_module.evaluate_response(refined_response, context)
        print(f"Utility Evaluation: {evaluation}")

        final_response = f"{refined_response}\n(Confidence: {evaluation['confidence'] * 100:.0f}%, Sentiment: {evaluation['sentiment']}{' Flags: ' + ', '.join(evaluation['flags']) if evaluation['flags'] else ''})"
        self.working_memory.update_context(customer_id, {"speaker": "agent", "text": final_response})
        
        return final_response


if __name__ == "__main__":
    # Simulate external systems/databases
    customer_db = {
        "cust_001": {
            "id": "cust_001",
            "name": "Alice Smith",
            "email": "alice@example.com",
            "conversation_history": []
        },
        "cust_002": {
            "id": "cust_002",
            "name": "Bob Johnson",
            "email": "bob@example.com",
            "conversation_history": []
        }
    }
    product_db = {"prod_001": {"name": "Laptop X", "specs": "16GB RAM, 512GB SSD"}}

    # Initialize modules
    simulated_llm = SimulatedLLM()
    working_memory = WorkingMemoryModule(customer_db, product_db)
    
    policy_rules = [
        {"condition": "order status", "action": lambda res, ctx: res, "suggested_action": {"type": "check_order_status"}},
        {"condition": "technical issue", "action": lambda res, ctx: res, "suggested_action": {"type": "create_ticket"}}
    ]
    policy_module = PolicyModule(policy_rules)

    # Simulate APIs for ActionExecutor
    simulated_crm_api = lambda x: print(f"CRM API called with: {x}")
    simulated_knowledge_base_api = lambda x: print(f"KB API called with: {x}")
    simulated_ticketing_system_api = lambda x: print(f"Ticketing API called with: {x}")
    action_executor = ActionExecutorModule(simulated_crm_api, simulated_knowledge_base_api, simulated_ticketing_system_api)
    
    utility_module = UtilityModule()

    # Initialize LLMAgent
    agent = LLMAgent(simulated_llm, working_memory, policy_module, action_executor, utility_module)

    print("--- Customer 1: Checking order status ---")
    response = agent.process_query("cust_001", "Hey, what's the status of my recent order?")
    print(f"Final Agent Response: {response}\n")

    print("--- Customer 2: Reporting a technical issue ---")
    response = agent.process_query("cust_002", "I'm having a technical issue with my account, can you help?")
    print(f"Final Agent Response: {response}\n")

    print("--- Customer 1: Asking about product details ---")
    response = agent.process_query("cust_001", "Can you tell me more about Laptop X?")
    print(f"Final Agent Response: {response}\n")

    print("--- Customer 2: Requesting escalation ---")
    response = agent.process_query("cust_002", "This is not helping, I need to speak to a human agent!")
    print(f"Final Agent Response: {response}\n")

    print("--- Customer 1: General query ---")
    response = agent.process_query("cust_001", "Hello, how are you today?")
    print(f"Final Agent Response: {response}\n")