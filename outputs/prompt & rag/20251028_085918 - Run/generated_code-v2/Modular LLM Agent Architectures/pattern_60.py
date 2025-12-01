class AgenticWorkingMemory:
    def __init__(self):
        self._state = {
            "q": None, 
            "e": {},
            "o": [], 
            "u": [], 
            "f": None, 
            "h": []
        }

    def update_query(self, query):
        self._state["q"] = query

    def add_external_evidence(self, evidence):
        self._state["e"].update(evidence)

    def add_llm_response(self, response, score):
        self._state["o"].append(response)
        self._state["u"].append(score)

    def update_feedback(self, feedback):
        self._state["f"] = feedback

    def add_to_history(self, user_msg, agent_response):
        self._state["h"].append({"user": user_msg, "agent": agent_response})

    def get_state(self):
        return self._state

class LLM_Interface:
    def generate_response(self, prompt):
        if "product" in prompt.lower():
            return "Based on your query, the XYZ product is currently in stock." 
        elif "order" in prompt.lower():
            return "Your order #12345 is currently being processed and expected to ship tomorrow." 
        return "I am an AI assistant. How can I help you today?"

class ExternalDataReader:
    def retrieve_product_info(self, product_id):
        if product_id == "XYZ":
            return {"product_id": product_id, "name": "XYZ Wireless Headphones", "price": 99.99, "stock": "In Stock"}
        return {}

    def retrieve_order_status(self, order_id):
        if order_id == "12345":
            return {"order_id": order_id, "status": "Processing", "delivery_date": "2023-11-15"}
        return {}

class PromptEngine:
    def construct_prompt(self, working_memory_state):
        prompt_parts = []
        if working_memory_state["h"]:
            prompt_parts.append("Dialog History:")
            for turn in working_memory_state["h"]:
                prompt_parts.append(f"User: {turn['user']}")
                prompt_parts.append(f"Agent: {turn['agent']}")
        
        if working_memory_state["e"]:
            prompt_parts.append("External Evidence:")
            for key, value in working_memory_state["e"].items():
                prompt_parts.append(f"{key}: {value}")
        
        if working_memory_state["q"]:
            prompt_parts.append(f"Current User Query: {working_memory_state['q']}")
        
        if working_memory_state["f"]:
            prompt_parts.append(f"Agent Feedback: {working_memory_state['f']}")

        return "\n".join(prompt_parts)

class PolicyModule:
    def decide_action(self, working_memory_state):
        if working_memory_state["o"]:
            # For simplicity, just pick the first candidate response
            chosen_response = working_memory_state["o"][0]
            feedback = "" # No explicit feedback for this simple policy
            return chosen_response, feedback
        return "I'm not sure how to respond to that.", ""

class CustomerSupportAgent:
    def __init__(self):
        self.working_memory = AgenticWorkingMemory()
        self.llm_interface = LLM_Interface()
        self.external_data_reader = ExternalDataReader()
        self.prompt_engine = PromptEngine()
        self.policy_module = PolicyModule()

    def handle_query(self, user_query):
        self.working_memory.update_query(user_query)

        # Mock external data retrieval
        external_evidence = {}
        if "product" in user_query.lower():
            product_id = "XYZ" # Mock product ID
            product_info = self.external_data_reader.retrieve_product_info(product_id)
            if product_info: 
                external_evidence.update(product_info)
        elif "order" in user_query.lower():
            order_id = "12345" # Mock order ID
            order_status = self.external_data_reader.retrieve_order_status(order_id)
            if order_status: 
                external_evidence.update(order_status)
                
        if external_evidence:
            self.working_memory.add_external_evidence(external_evidence)

        prompt = self.prompt_engine.construct_prompt(self.working_memory.get_state())
        
        llm_candidate_response = self.llm_interface.generate_response(prompt)
        
        # Mock utility score
        utility_score = 0.8 
        self.working_memory.add_llm_response(llm_candidate_response, utility_score)

        chosen_response, feedback = self.policy_module.decide_action(self.working_memory.get_state())
        
        if feedback:
            self.working_memory.update_feedback(feedback)

        self.working_memory.add_to_history(user_query, chosen_response)
        
        return chosen_response

if __name__ == "__main__":
    agent = CustomerSupportAgent()

    print("\n--- Turn 1 ---")
    response1 = agent.handle_query("What is the status of product XYZ?")
    print(f"User: What is the status of product XYZ?")
    print(f"Agent: {response1}")
    # print("Memory State:", agent.working_memory.get_state())
    
    print("\n--- Turn 2 ---")
    response2 = agent.handle_query("And what about my order 12345?")
    print(f"User: And what about my order 12345?")
    print(f"Agent: {response2}")
    # print("Memory State:", agent.working_memory.get_state())

    print("\n--- Turn 3 ---")
    response3 = agent.handle_query("Thanks!")
    print(f"User: Thanks!")
    print(f"Agent: {response3}")
    print("\nFinal Memory State:", agent.working_memory.get_state())
