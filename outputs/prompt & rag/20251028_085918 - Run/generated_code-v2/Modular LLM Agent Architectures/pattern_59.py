class LLMWrapper:
    def generate_response(self, prompt: str) -> str:
        # Simulate LLM response
        if "product information" in prompt.lower():
            return "The product XYZ features a 108MP camera and a 5000mAh battery. For more details, please visit our website."
        elif "delivery status" in prompt.lower():
            return "I can help with delivery status. Could you please provide your order number?"
        elif "issue with my order" in prompt.lower():
            return "I understand you're having an issue. Please describe it in more detail, and I'll see if I can find a solution or escalate it."
        return f"Hello! I'm a simulated LLM. You asked: '{prompt}'. How else can I assist you today?"

class WorkingMemoryModule:
    def __init__(self):
        self._conversation_history = []
        self._customer_data = {}

    def add_message(self, sender: str, message: str):
        self._conversation_history.append({"sender": sender, "message": message})

    def get_conversation_history(self) -> list:
        return self._conversation_history

    def update_customer_data(self, data: dict):
        self._customer_data.update(data)

    def get_customer_data(self) -> dict:
        return self._customer_data

class PolicyModule:
    def decide_action(self, current_state: dict, llm_response: str) -> str:
        if "order number" in llm_response.lower() and "customer_id" not in current_state.get("customer_data", {}):
            return "request_customer_id"
        if "more details" in llm_response.lower() and "escalate" not in llm_response.lower():
            return "ask_for_clarification"
        if "escalate" in llm_response.lower() or "complex issue" in current_state.get("problem_description", "").lower():
            return "create_support_ticket"
        if "product information" in llm_response.lower():
            return "provide_product_info"
        if "delivery status" in llm_response.lower() and "order_number" in current_state.get("customer_data", {}):
            return "check_delivery_status"
        return "respond_to_user"

class ActionExecutorModule:
    def search_faq(self, query: str) -> str:
        # Simulate FAQ search
        if "return policy" in query.lower():
            return "Our return policy allows returns within 30 days of purchase with a valid receipt."
        return f"No direct FAQ found for '{query}'."

    def retrieve_customer_info(self, customer_id: str) -> dict:
        # Simulate CRM lookup
        if customer_id == "CUST123":
            return {"name": "Alice Smith", "email": "alice@example.com", "last_order": "ORD456"}
        return {}

    def create_support_ticket(self, details: dict) -> str:
        # Simulate ticket creation
        return f"Support ticket created with details: {details.get('issue', 'N/A')}. Reference ID: TKT789."

class UtilityModule:
    def evaluate_response(self, llm_response: str, context: dict) -> dict:
        evaluation = {"score": 1.0, "feedback": "Good response.", "action_needed": None}
        if "hallucination" in llm_response.lower() or "made-up" in llm_response.lower():
            evaluation["score"] = 0.2
            evaluation["feedback"] = "Potentially hallucinated content detected."
            evaluation["action_needed"] = "human_review"
        if len(llm_response) < 10:
            evaluation["score"] = 0.5
            evaluation["feedback"] = "Response is too short, might be unhelpful."
        if "sensitive_info_request" in context.get("flags", []) and "SSN" in llm_response:
            evaluation["score"] = 0.1
            evaluation["feedback"] = "Sensitive information leakage detected."
            evaluation["action_needed"] = "censor_response"
        return evaluation

class CustomerSupportAssistant:
    def __init__(self):
        self.llm_wrapper = LLMWrapper()
        self.working_memory = WorkingMemoryModule()
        self.policy_module = PolicyModule()
        self.action_executor = ActionExecutorModule()
        self.utility_module = UtilityModule()

    def process_query(self, user_query: str) -> str:
        self.working_memory.add_message("user", user_query)
        conversation_history = self.working_memory.get_conversation_history()
        customer_data = self.working_memory.get_customer_data()

        prompt_parts = [
            "You are a helpful customer support assistant.",
            "Conversation history:",
        ]
        for msg in conversation_history:
            prompt_parts.append(f"{msg['sender'].capitalize()}: {msg['message']}")
        if customer_data:
            prompt_parts.append(f"Customer data: {customer_data}")
        prompt_parts.append(f"User's current query: {user_query}")
        prompt_parts.append("Please provide a concise and helpful response.")

        llm_prompt = "\n".join(prompt_parts)
        raw_llm_response = self.llm_wrapper.generate_response(llm_prompt)

        current_state = {
            "conversation_history": conversation_history,
            "customer_data": customer_data,
            "user_query": user_query,
            "problem_description": user_query # Simple for demo
        }

        evaluation = self.utility_module.evaluate_response(raw_llm_response, current_state)
        if evaluation["action_needed"] == "human_review" or evaluation["action_needed"] == "censor_response":
            return f"[ASSISTANT - EVALUATION FAILED: {evaluation['feedback']}] Please wait while I connect you to a human agent or rephrase your query."

        action = self.policy_module.decide_action(current_state, raw_llm_response)
        final_response = raw_llm_response # Start with LLM's response

        if action == "request_customer_id":
            final_response = "Could you please provide your customer ID so I can look up your details?"
        elif action == "ask_for_clarification":
            final_response = "Could you please provide more details about your issue?"
        elif action == "create_support_ticket":
            ticket_details = {"issue": user_query, "customer_id": customer_data.get("id", "unknown"), "priority": "medium"}
            ticket_confirmation = self.action_executor.create_support_ticket(ticket_details)
            final_response = f"I've created a support ticket for you. {ticket_confirmation}"
        elif action == "provide_product_info":
            # Assuming LLM already provided, but can enhance here if needed from external source
            pass 
        elif action == "check_delivery_status":
            order_num = customer_data.get("order_number", "unknown")
            # In a real scenario, this would call an external API with order_num
            final_response = f"Checking delivery status for order {order_num}. It appears your order is out for delivery and expected today."

        self.working_memory.add_message("assistant", final_response)
        return final_response

# --- Example Usage --- 
if __name__ == "__main__":
    assistant = CustomerSupportAssistant()

    print("\n--- Scenario 1: Basic Query ---")
    response = assistant.process_query("Hi, I'd like to know about your new phone.")
    print(f"Assistant: {response}")

    print("\n--- Scenario 2: Requesting Customer ID (Policy Module in action) ---")
    response = assistant.process_query("What is the delivery status of my order?")
    print(f"Assistant: {response}")

    print("\n--- Scenario 3: Providing Customer ID and checking delivery ---")
    assistant.working_memory.update_customer_data({"order_number": "ORD12345"})
    response = assistant.process_query("My order number is ORD12345. What's the status?")
    print(f"Assistant: {response}")

    print("\n--- Scenario 4: Simulating an issue leading to ticket creation ---")
    response = assistant.process_query("I have a complex issue with my recent purchase where it arrived broken.")
    print(f"Assistant: {response}")

    print("\n--- Scenario 5: Simulating a short, potentially unhelpful response from LLM (Utility Module in action) ---")
    # Manually setting a context flag for this scenario demonstration
    assistant.utility_module.evaluate_response_mock_flag = True 
    assistant.llm_wrapper.generate_response_mock_short = True
    response = assistant.process_query("Tell me about the return policy.")
    print(f"Assistant: {response}")
    del assistant.utility_module.evaluate_response_mock_flag
    del assistant.llm_wrapper.generate_response_mock_short

    print("\n--- Scenario 6: Search FAQ (Action Executor Module) ---")
    response = assistant.process_query("What is your return policy?")
    print(f"Assistant: {assistant.action_executor.search_faq('return policy')}")
    self_response = assistant.process_query("What is your return policy?") # Process again to update history
    print(f"Assistant (after policy): {self_response}")
