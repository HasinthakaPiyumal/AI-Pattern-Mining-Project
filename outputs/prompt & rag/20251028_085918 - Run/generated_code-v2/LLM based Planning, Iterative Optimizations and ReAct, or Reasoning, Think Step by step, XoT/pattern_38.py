class DummyLLM:
    def __init__(self, responses):
        self.responses = responses
        self.call_count = 0

    def generate(self, prompt, error=None):
        if error and "error" in prompt.lower():
            correction_prompt_idx = min(self.call_count, len(self.responses) - 1)
            self.call_count += 1
            return self.responses[correction_prompt_idx]
        
        initial_prompt_idx = min(self.call_count, len(self.responses) - 1)
        self.call_count += 1
        return self.responses[initial_prompt_idx]

class Tool:
    def __init__(self, name, description, func):
        self.name = name
        self.description = description
        self.func = func

    def run(self, **kwargs):
        return self.func(**kwargs)

def check_order_status_tool(order_id):
    if order_id == "ORDER123":
        return {"status": "delivered", "eta": "N/A"}
    elif order_id == "ERROR404":
        raise ValueError("Order not found")
    else:
        return {"status": "processing", "eta": "2 days"}

def create_ticket_tool(customer_id, issue_description):
    if not customer_id or not issue_description:
        raise ValueError("Customer ID and issue description are required")
    if "payment" in issue_description.lower():
        return {"ticket_id": "TKT789", "status": "escalated to finance"}
    return {"ticket_id": "TKT456", "status": "created"}

class CustomerSupportAgent:
    def __init__(self, llm, tools, max_attempts=3):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.max_attempts = max_attempts
        self.history = []

    def _identify_intent_and_tool(self, prompt, error=None):
        # Simplified: LLM decides intent and tool based on prompt
        # In a real scenario, this would involve more sophisticated LLM parsing
        if error:
            rationale = self.llm.generate(f"Analyze the previous error '{error}' and user query: '{prompt}'. Propose a correction or alternative tool use.", error=error)
        else:
            rationale = self.llm.generate(f"User query: '{prompt}'. Determine the best tool and parameters.")
        
        # Extract tool name and arguments from rationale (simplified)
        if "use check_order_status" in rationale.lower():
            order_id = "ORDER123" # Placeholder for LLM argument extraction
            if "error404" in rationale.lower():
                order_id = "ERROR404"
            elif "unknown order" in rationale.lower():
                order_id = "UNKNOWN_ORDER"
            return "check_order_status", {"order_id": order_id}, rationale
        elif "use create_ticket" in rationale.lower():
            customer_id = "CUST001" # Placeholder
            issue = "general inquiry" # Placeholder
            if "payment issue" in rationale.lower():
                issue = "payment issue"
            elif "no customer id" in rationale.lower():
                customer_id = ""
            return "create_ticket", {"customer_id": customer_id, "issue_description": issue}, rationale
        return None, {}, rationale

    def chat(self, user_query):
        current_query = user_query
        attempts = 0

        while attempts < self.max_attempts:
            attempts += 1
            print(f"\n--- Attempt {attempts} ---")
            
            tool_name, tool_args, rationale = self._identify_intent_and_tool(current_query, self.history[-1]['error'] if self.history and 'error' in self.history[-1] else None)
            print(f"Agent Rationale: {rationale}")
            self.history.append({"query": current_query, "rationale": rationale})

            if tool_name and tool_name in self.tools:
                try:
                    print(f"Executing tool: {tool_name} with args: {tool_args}")
                    tool_output = self.tools[tool_name].run(**tool_args)
                    print(f"Tool Output: {tool_output}")
                    
                    # LLM-based reflection on tool output and generation of final response
                    final_response_rationale = self.llm.generate(f"Based on user query '{user_query}' and tool output '{tool_output}', formulate a helpful response.")
                    print(f"Agent Response: {final_response_rationale}")
                    self.history[-1]['response'] = final_response_rationale
                    return final_response_rationale

                except Exception as e:
                    print(f"Tool Error: {e}")
                    self.history[-1]['error'] = str(e)
                    current_query = f"User asked: '{user_query}'. Previous attempt failed with error: '{e}'. Try to correct or find an alternative approach."
                    print("Agent is attempting self-correction...")
            else:
                # If no tool was identified or it was an unknown tool
                error_msg = f"Could not identify a suitable tool or the identified tool '{tool_name}' is not available."
                print(f"Agent Error: {error_msg}")
                self.history[-1]['error'] = error_msg
                current_query = f"User asked: '{user_query}'. Previous attempt failed: '{error_msg}'. Re-evaluate the intent."
                print("Agent is attempting self-correction...")

        final_fallback_response = "I'm sorry, I'm having trouble assisting you with that request after multiple attempts. Would you like to speak to a human agent?"
        print(f"Max attempts reached. Agent Response: {final_fallback_response}")
        self.history.append({"query": user_query, "response": final_fallback_response})
        return final_fallback_response


# --- Example Usage ---

# Dummy LLM responses to simulate different stages of interaction
llm_responses = [
    "User query: 'What is the status of my order?' Determine the best tool and parameters. I should use check_order_status with order_id ORDER123.",
    "Analyze the previous error 'Order not found' and user query: 'What is the status of my order?'. Propose a correction or alternative tool use. The order ID might be wrong. Let's try to ask user for the order ID. Or if it's an error, assume it was ERROR404 to trigger an error for demonstration.",
    "Based on user query 'What is the status of my order?' and tool output '{'status': 'delivered', 'eta': 'N/A'}', formulate a helpful response. Your order ORDER123 has been delivered. No ETA is available.",
    "Based on user query 'What is the status of my order?' and tool output '{'status': 'processing', 'eta': '2 days'}', formulate a helpful response. Your order UNKNOWN_ORDER is currently processing and is expected in 2 days.",
    "Analyze the previous error 'Order not found' and user query: 'I have a payment issue.'. Propose a correction or alternative tool use. This is a payment issue. I should use create_ticket with customer_id CUST001 and issue payment issue.",
    "Based on user query 'I have a payment issue.' and tool output '{'ticket_id': 'TKT789', 'status': 'escalated to finance'}', formulate a helpful response. Your payment issue has been logged with ticket ID TKT789 and escalated to the finance department.",
    "User query: 'I need help'. Determine the best tool and parameters. This is a general inquiry. I should use create_ticket with customer_id CUST001 and issue general inquiry.",
    "Based on user query 'I need help' and tool output '{'ticket_id': 'TKT456', 'status': 'created'}', formulate a helpful response. I have created a general support ticket for you with ID TKT456. Someone will get back to you shortly.",
    "Analyze the previous error 'Customer ID and issue description are required' and user query: 'I need to create a ticket but don't have a customer ID'. Propose a correction or alternative tool use. The agent failed to provide customer ID. I need to explicitly state no customer id for create_ticket. Let's try to use create_ticket with no customer id for demonstration."
]
dummy_llm = DummyLLM(llm_responses)

# Define tools
check_order_tool = Tool("check_order_status", "Checks the status of a customer order.", check_order_status_tool)
create_ticket_tool_obj = Tool("create_ticket", "Creates a support ticket for a customer issue.", create_ticket_tool)

agent = CustomerSupportAgent(dummy_llm, [check_order_tool, create_ticket_tool_obj])

print("Scenario 1: Successful order status check")
agent.chat("What is the status of my order?")

print("\nScenario 2: Order not found error and implicit self-correction (simulated)")
# Reset LLM for next scenario demonstration
dummy_llm = DummyLLM([
    "User query: 'What is the status of my order?'. Determine the best tool and parameters. I should use check_order_status with order_id ERROR404.",
    "Analyze the previous error 'Order not found' and user query: 'What is the status of my order?'. Propose a correction or alternative tool use. The previous order ID ERROR404 was incorrect. Let's try a different order ID, say UNKNOWN_ORDER.",
    "Based on user query 'What is the status of my order?' and tool output '{'status': 'processing', 'eta': '2 days'}', formulate a helpful response. Your order UNKNOWN_ORDER is currently processing and is expected in 2 days.",
    "Based on user query 'What is the status of my order?' and tool output '{'status': 'delivered', 'eta': 'N/A'}', formulate a helpful response. Your order ORDER123 has been delivered. No ETA is available."
])
agent = CustomerSupportAgent(dummy_llm, [check_order_tool, create_ticket_tool_obj])
agent.chat("What is the status of my order?")

print("\nScenario 3: Creating a ticket with missing info (simulated error and correction)")
# Reset LLM for next scenario demonstration
dummy_llm = DummyLLM([
    "User query: 'I need to create a ticket but don't have a customer ID'. Determine the best tool and parameters. I should use create_ticket with customer_id and issue general inquiry.",
    "Analyze the previous error 'Customer ID and issue description are required' and user query: 'I need to create a ticket but don't have a customer ID'. Propose a correction or alternative tool use. The previous attempt failed because customer ID was not passed. I need to explicitly pass an empty string for customer_id if it's missing for demonstration of error handling. Let's try create_ticket with customer_id '' and issue general inquiry.",
    "Based on user query 'I need to create a ticket but don't have a customer ID' and tool output '{'ticket_id': 'TKT456', 'status': 'created'}', formulate a helpful response. I have created a general support ticket for you with ID TKT456. Someone will get back to you shortly."
])
agent = CustomerSupportAgent(dummy_llm, [check_order_tool, create_ticket_tool_obj])
agent.chat("I need to create a ticket but don't have a customer ID")
