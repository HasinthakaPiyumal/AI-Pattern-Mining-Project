import json

def _knowledge_base_search(query: str) -> str:
    """Simulates searching a knowledge base for answers."""
    print(f"[Tool Call] KnowledgeBaseSearch with query: '{query}'")
    if "billing issue" in query.lower():
        return "Solution for billing issue: Please check the customer's payment history and subscription plan. Advise them to update payment method if expired."
    elif "product feature" in query.lower():
        return "Product X has feature Y. Refer to the user manual for details."
    else:
        return "No relevant information found in the knowledge base."

def _crm_lookup(customer_id: str) -> str:
    """Simulates looking up customer details in a CRM system."""
    print(f"[Tool Call] CRMLookup with customer_id: '{customer_id}'")
    if customer_id == "CUST001":
        return json.dumps({"id": "CUST001", "name": "Alice Smith", "status": "Active", "recent_issues": ["billing issue"]})
    elif customer_id == "CUST002":
        return json.dumps({"id": "CUST002", "name": "Bob Johnson", "status": "Inactive", "recent_issues": []})
    else:
        return "Customer not found."

def _order_management(order_id: str, action: str = "status") -> str:
    """Simulates interacting with an order management system."""
    print(f"[Tool Call] OrderManagement with order_id: '{order_id}', action: '{action}'")
    if order_id == "ORD87654":
        if action == "status":
            return "Order ORD87654 status: Shipped on 2023-10-26. Tracking: TRK12345."
        elif action == "cancel":
            return "Order cancellation for ORD87654 initiated. Refund in 3-5 business days."
    else:
        return "Order not found."

# --- Tool Registry ---
tools = {
    "KnowledgeBaseSearch": _knowledge_base_search,
    "CRMLookup": _crm_lookup,
    "OrderManagement": _order_management,
}

# --- LLM Simulation --- 
def _simulate_llm_response(history: list[str], current_observation: str) -> str:
    """Simulates an LLM's reasoning and tool call/final answer generation.
    
    The LLM's response format: 'THOUGHT: ... (TOOL_CALL: ToolName(arg=value))' or 'FINAL_ANSWER: ...'
    """
    full_context = "\n".join(history) + (f"\nOBSERVATION: {current_observation}" if current_observation else "")
    print(f"\n[LLM Input Context]:\n{full_context}\n")

    # Simple rule-based simulation based on current observation and history
    if "customer_id" in current_observation and "billing issue" in full_context:
        if "Solution for billing issue" in full_context:
             return "FINAL_ANSWER: I have identified the customer's billing issue and found a solution in the knowledge base. The solution is: Please check the customer's payment history and subscription plan and advise them to update payment method if expired. Customer details: CUST001, Alice Smith."
        return "THOUGHT: I have identified the customer and their recent issue. Now I need to find a solution for the billing issue in the knowledge base. TOOL_CALL: KnowledgeBaseSearch(query='billing issue CUST001')"
    elif "Order status for ORD87654" in full_context or "track order ORD87654" in full_context:
        if "Order ORD87654 status: Shipped" in full_context:
            return "FINAL_ANSWER: The order ORD87654 has been shipped on 2023-10-26 with tracking number TRK12345."
        return "THOUGHT: The user is asking about an order. I should check the order status. TOOL_CALL: OrderManagement(order_id='ORD87654', action='status')"
    elif "customer_id" in current_observation and "Customer not found" in current_observation:
        return "FINAL_ANSWER: I could not find the customer with the provided ID. Please double-check the customer ID."
    elif "customer_id" in full_context and not ("billing issue" in full_context or "Order status" in full_context):
        return f"FINAL_ANSWER: I found customer details for {json.loads(current_observation).get('name')}. How can I further assist them?"
    elif "CRM lookup" in full_context and "customer ID" in full_context.lower() and not "TOOL_CALL" in full_context:
        # Initial step to find customer ID if not directly provided in current_observation
        # This would usually come from NLU, but for simulation, we'll infer from query
        customer_id_match = next((c for c in ["CUST001", "CUST002"] if c in full_context), None)
        if customer_id_match:
            return f"THOUGHT: The query seems to involve a customer. I should look up customer details. TOOL_CALL: CRMLookup(customer_id='{customer_id_match}')"
        else:
             return "FINAL_ANSWER: I need a customer ID to look up customer details. Please provide one."
    elif "cancel order" in full_context.lower() and "ORD87654" in full_context:
        if "Order cancellation for ORD87654 initiated" in full_context:
             return "FINAL_ANSWER: The cancellation for order ORD87654 has been initiated and a refund will be processed in 3-5 business days."
        return "THOUGHT: The user wants to cancel an order. I should attempt to cancel it. TOOL_CALL: OrderManagement(order_id='ORD87654', action='cancel')"

    # Default catch-all
    return "FINAL_ANSWER: I am sorry, I cannot fully resolve this specific query with my current tools and simulated logic. Please try a different query related to billing issues, order status, or customer lookup."


def handle_customer_query(query: str, max_iterations: int = 5) -> str:
    """Manages the adaptive agent's interaction loop to resolve a customer query.

    The agent integrates reasoning with tools, processes feedback, and self-corrects.
    """
    print(f"\n--- Handling Customer Query: '{query}' ---")
    history = [f"User Query: {query}"]
    current_observation = ""
    
    for i in range(max_iterations):
        print(f"\n[Iteration {i+1}/{max_iterations}]")
        llm_response = _simulate_llm_response(history, current_observation)
        history.append(f"LLM Response: {llm_response}")

        if llm_response.startswith("FINAL_ANSWER:"):
            final_answer = llm_response.replace("FINAL_ANSWER: ", "")
            print(f"\n--- Agent Final Answer: ---\n{final_answer}")
            return final_answer
        elif "TOOL_CALL:" in llm_response:
            try:
                # Extract tool name and arguments
                tool_call_str = llm_response.split("TOOL_CALL:")[1].strip()
                tool_name_end = tool_call_str.find('(')
                tool_name = tool_call_str[:tool_name_end].strip()
                args_str = tool_call_str[tool_name_end+1:-1]
                
                # Parse arguments (basic parsing, assumes key=value pairs or simple strings)
                # This part is simplified and might need robust parsing for real-world scenarios
                kwargs = {}
                for arg_pair in args_str.split(','):
                    if '=' in arg_pair:
                        key, value = arg_pair.split('=', 1)
                        kwargs[key.strip()] = value.strip().strip("'\"") # Remove quotes
                    else: # Handle positional args if any (not used in current tools)
                        pass

                if tool_name in tools:
                    tool_func = tools[tool_name]
                    tool_output = tool_func(**kwargs)
                    current_observation = f"Tool Output ({tool_name}): {tool_output}"
                    history.append(current_observation)
                else:
                    current_observation = f"Tool Error: Unknown tool '{tool_name}'"
                    history.append(current_observation)
            except Exception as e:
                current_observation = f"Tool Call Error: {e}. Raw LLM output: {llm_response}"
                history.append(current_observation)
                print(current_observation)
        else:
            current_observation = f"LLM did not provide a recognized TOOL_CALL or FINAL_ANSWER format: {llm_response}"
            history.append(current_observation)
            print(current_observation)

    final_fallback = "Agent could not resolve the query within the maximum iterations. Please refine your request."
    print(f"\n--- Agent Final Answer (Fallback): ---\n{final_fallback}")
    return final_fallback

# --- Example Usage ---
if __name__ == "__main__":
    # Example 1: Resolving a billing issue
    print("\n========================================")
    handle_customer_query("I have a billing issue with my account CUST001.")
    print("\n========================================")

    # Example 2: Checking an order status
    print("\n========================================")
    handle_customer_query("What is the status of my order ORD87654?")
    print("\n========================================")

    # Example 3: Customer not found scenario
    print("\n========================================")
    handle_customer_query("Can you look up details for customer CUST999?")
    print("\n========================================")

    # Example 4: Cancelling an order
    print("\n========================================")
    handle_customer_query("I want to cancel order ORD87654.")
    print("\n========================================")

    # Example 5: Unresolvable query (demonstrates fallback)
    print("\n========================================")
    handle_customer_query("What is the meaning of life?")
    print("\n========================================")

    # Example 6: Query that only needs CRM lookup
    print("\n========================================")
    handle_customer_query("Tell me about customer CUST001.")
    print("\n========================================")
