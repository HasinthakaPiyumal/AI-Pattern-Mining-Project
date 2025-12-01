import random

class WorkingMemory:
    def __init__(self):
        self.user_query = None
        self.conversation_history = []
        self.customer_info = None
        self.knowledge_base_results = []
        self.service_status = None
        self.llm_candidate_response = None
        self.current_action = None
        self.has_customer_id = False
        self.has_issue_keyword = False
        self.has_outage_keyword = False

    def update_query(self, query):
        self.user_query = query
        self.conversation_history.append(f"User: {query}")
        self.has_customer_id = any(word.isdigit() for word in query.split() if len(word) > 4) # Simple check for a number
        self.has_issue_keyword = any(k in query.lower() for k in ["problem", "issue", "not working", "slow"])
        self.has_outage_keyword = any(k in query.lower() for k in ["outage", "down", "no service"])

    def update_customer_info(self, info):
        self.customer_info = info
        self.conversation_history.append(f"Agent: Retrieved customer info: {info}")

    def update_knowledge_results(self, results):
        self.knowledge_base_results = results
        self.conversation_history.append(f"Agent: Knowledge base results: {results}")

    def update_service_status(self, status):
        self.service_status = status
        self.conversation_history.append(f"Agent: Service status: {status}")

    def update_llm_response(self, response):
        self.llm_candidate_response = response
        self.conversation_history.append(f"Agent: LLM candidate response: {response}")

    def reset_for_new_query(self):
        self.user_query = None
        self.customer_info = None
        self.knowledge_base_results = []
        self.service_status = None
        self.llm_candidate_response = None
        self.current_action = None
        self.has_customer_id = False
        self.has_issue_keyword = False
        self.has_outage_keyword = False

class MockLLMClient:
    def query(self, prompt):
        # Simulate LLM response based on prompt
        if "customer details" in prompt:
            return "Based on the customer information, their current plan is 'Premium Fiber' and their last bill was $75.00."
        elif "knowledge base findings" in prompt:
            return "The knowledge base suggests troubleshooting steps for slow internet include restarting the router and checking cable connections."
        elif "service status" in prompt:
            return "Current service status indicates no widespread outages in your area. If the issue persists, a technician visit might be needed."
        else:
            return f"I understand you're asking about '{prompt[:50]}...'. Let me get more information."

class MockCustomerDB:
    def get_customer_details(self, customer_id):
        if customer_id == "123456789":
            return {"id": customer_id, "name": "Alice Smith", "plan": "Premium Fiber", "address": "123 Main St", "status": "Active"}
        return None

class MockKnowledgeBase:
    def search_articles(self, query):
        if "slow internet" in query.lower():
            return ["Troubleshooting slow internet guide", "Optimizing Wi-Fi performance"]
        elif "billing question" in query.lower():
            return ["Understanding your bill", "Payment options"]
        return ["General FAQ"]

class MockServiceStatusAPI:
    def check_status(self, area=None):
        if random.random() < 0.1: # 10% chance of an outage
            return {"status": "Outage", "details": "Planned maintenance in your area.", "eta": "2 hours"}
        return {"status": "Operational", "details": "All services are running normally.", "area": area}

class AgenticPolicy:
    def decide_action(self, memory: WorkingMemory):
        if memory.llm_candidate_response and len(memory.llm_candidate_response) > 50 and not memory.has_issue_keyword:
            memory.current_action = "SEND_FINAL_RESPONSE"
            return "SEND_FINAL_RESPONSE"
        
        if memory.has_outage_keyword and not memory.service_status:
            memory.current_action = "CHECK_SERVICE_STATUS"
            return "CHECK_SERVICE_STATUS"

        if memory.has_customer_id and not memory.customer_info:
            memory.current_action = "RETRIEVE_CUSTOMER_INFO"
            return "RETRIEVE_CUSTOMER_INFO"
        
        if memory.has_issue_keyword and not memory.knowledge_base_results and not memory.service_status:
            memory.current_action = "SEARCH_KNOWLEDGE_BASE"
            return "SEARCH_KNOWLEDGE_BASE"
        
        if (memory.customer_info or memory.knowledge_base_results or memory.service_status) and not memory.llm_candidate_response:
            memory.current_action = "QUERY_LLM_FOR_RESPONSE"
            return "QUERY_LLM_FOR_RESPONSE"
        
        memory.current_action = "QUERY_LLM_FOR_RESPONSE" # Fallback to LLM if no specific rule matches
        return "QUERY_LLM_FOR_RESPONSE"

class CustomerSupportAgent:
    def __init__(self):
        self.memory = WorkingMemory()
        self.llm = MockLLMClient()
        self.customer_db = MockCustomerDB()
        self.knowledge_base = MockKnowledgeBase()
        self.service_api = MockServiceStatusAPI()
        self.policy = AgenticPolicy()

    def _execute_action(self, action):
        if action == "RETRIEVE_CUSTOMER_INFO":
            customer_id = next((word for word in self.memory.user_query.split() if len(word) > 4 and word.isdigit()), None)
            if customer_id:
                info = self.customer_db.get_customer_details(customer_id)
                self.memory.update_customer_info(info)
                print(f"Agent executing: Retrieved customer info for ID {customer_id}.")
            else:
                print("Agent executing: Could not find customer ID in query.")

        elif action == "SEARCH_KNOWLEDGE_BASE":
            results = self.knowledge_base.search_articles(self.memory.user_query)
            self.memory.update_knowledge_results(results)
            print(f"Agent executing: Searched knowledge base for '{self.memory.user_query}'.")

        elif action == "CHECK_SERVICE_STATUS":
            status = self.service_api.check_status()
            self.memory.update_service_status(status)
            print(f"Agent executing: Checked service status.")

        elif action == "QUERY_LLM_FOR_RESPONSE":
            prompt_parts = [f"User query: {self.memory.user_query}"]
            if self.memory.conversation_history: prompt_parts.append(f"Conversation history: {' '.join(self.memory.conversation_history[-3:])}")
            if self.memory.customer_info: prompt_parts.append(f"Customer details: {self.memory.customer_info}")
            if self.memory.knowledge_base_results: prompt_parts.append(f"Knowledge base findings: {', '.join(self.memory.knowledge_base_results)}")
            if self.memory.service_status: prompt_parts.append(f"Service status: {self.memory.service_status}")
            
            prompt = " ".join(prompt_parts) + "\nBased on the above, generate a helpful and concise response to the user."
            llm_response = self.llm.query(prompt)
            self.memory.update_llm_response(llm_response)
            print(f"Agent executing: Queried LLM with gathered info.")

        elif action == "SEND_FINAL_RESPONSE":
            print(f"Agent: {self.memory.llm_candidate_response}")
            print("--- Conversation Ended ---")
            self.memory.reset_for_new_query()

        else:
            print(f"Agent: Unknown action: {action}")
    
    def run_conversation(self, user_input):
        self.memory.update_query(user_input)
        print(f"User: {user_input}")

        while self.memory.current_action != "SEND_FINAL_RESPONSE":
            action = self.policy.decide_action(self.memory)
            print(f"Policy decided action: {action}")
            self._execute_action(action)
            if self.memory.current_action == "SEND_FINAL_RESPONSE":
                break
            # Prevent infinite loops in case of policy issues or lack of clear resolution
            if not self.memory.llm_candidate_response and action == "QUERY_LLM_FOR_RESPONSE":
                break # Give up if LLM cannot provide a response after being queried once


if __name__ == "__main__":
    agent = CustomerSupportAgent()

    print("\n--- Scenario 1: Basic query requiring KB ---")
    agent.run_conversation("My internet is really slow, what should I do?")

    print("\n--- Scenario 2: Query with customer ID ---")
    agent.run_conversation("I'm Alice Smith, my customer ID is 123456789. Can you check my plan?")

    print("\n--- Scenario 3: Query about an outage ---")
    agent.run_conversation("Is there an outage in my area? My service is completely down.")

    print("\n--- Scenario 4: General query ---")
    agent.run_conversation("What are your business hours?")
