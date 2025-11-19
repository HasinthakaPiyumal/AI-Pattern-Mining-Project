class ComposableAIAgent:
    def __init__(self):
        self.conversation_history = []
        self.customer_context = {}
        self.tools = {
            "crm_lookup": self._crm_lookup,
            "order_status_check": self._order_status_check,
            "update_delivery_address": self._update_delivery_address,
            "retrieve_from_kb": self._retrieve_from_kb,
        }
        print("Composable AI Agent initialized with basic tools and memory.")

    def _call_llm(self, prompt, context=None):
        """Simulates an LLM call for planning and response generation."""
        # In a real scenario, this would interface with a large language model API.
        # For this example, we'll use simple rule-based responses or string manipulation.
        
        full_prompt = ""
        if context:
            full_prompt += f"Context: {context}\n"
        full_prompt += f"Inquiry: {prompt}\n"
        full_prompt += "Based on the above, what action should be taken or what is the response?"

        print(f"[LLM Mock Call] Processing prompt: {full_prompt[:100]}...")

        # Simple mock LLM logic for demonstration
        if "order status" in prompt.lower() or "where is my order" in prompt.lower():
            return "ACTION: order_status_check"
        elif "change delivery address" in prompt.lower():
            return "ACTION: update_delivery_address"
        elif "customer history" in prompt.lower() or "who is this customer" in prompt.lower():
            return "ACTION: crm_lookup"
        elif "product information" in prompt.lower() or "faq" in prompt.lower():
            return "ACTION: retrieve_from_kb"
        else:
            return "LLM Response: I am processing your request. If I can't find a specific tool, I'll try to provide a general answer or ask for more details."

    def _crm_lookup(self, customer_id="unknown"): # Simplified for demo
        """Mocks a CRM system lookup."""
        print(f"[Tool Use] Performing CRM lookup for customer ID: {customer_id}")
        # In a real system, this would call an external CRM API.
        if customer_id == "cust123":
            self.customer_context = {"id": "cust123", "name": "Alice Smith", "tier": "Gold"}
            return {"status": "success", "data": self.customer_context}
        else:
            return {"status": "failure", "message": "Customer not found."}

    def _order_status_check(self, order_id="unknown"): # Simplified for demo
        """Mocks an order management system check."""
        print(f"[Tool Use] Checking order status for order ID: {order_id}")
        # In a real system, this would call an external order management API.
        if order_id == "ORD456":
            return {"status": "success", "data": {"order_id": "ORD456", "status": "Shipped", "estimated_delivery": "2023-12-25"}}
        else:
            return {"status": "failure", "message": "Order not found."}

    def _update_delivery_address(self, order_id="unknown", new_address="unknown"): # Simplified for demo
        """Mocks updating a delivery address in the order management system."""
        print(f"[Tool Use] Attempting to update address for order {order_id} to {new_address}")
        # In a real system, this would call an external order management API.
        if order_id == "ORD456" and new_address != "unknown":
            return {"status": "success", "message": f"Delivery address for order {order_id} updated to {new_address}."}
        else:
            return {"status": "failure", "message": "Failed to update address."}

    def _retrieve_from_kb(self, query): # Simplified for demo
        """Mocks retrieving information from a RAG knowledge base."""
        print(f"[Tool Use] Retrieving from knowledge base for query: {query}")
        # In a real RAG system, this would involve embedding the query, 
        # searching a vector database, and potentially re-ranking results.
        if "return policy" in query.lower():
            return {"status": "success", "data": "Our return policy allows returns within 30 days for a full refund."}
        elif "product warranty" in query.lower():
            return {"status": "success", "data": "All products come with a one-year manufacturer's warranty."}
        else:
            return {"status": "failure", "message": "No relevant information found in KB."}

    def _plan_and_execute(self, inquiry):
        """Orchestrates the LLM and tool usage based on the inquiry."""
        current_context = {
            "conversation_history": self.conversation_history,
            "customer_context": self.customer_context,
            "current_inquiry": inquiry
        }
        
        # Step 1: LLM determines the initial action or response
        llm_decision = self._call_llm(inquiry, context=current_context)
        print(f"[Planning] LLM initial decision: {llm_decision}")

        response = ""
        if llm_decision.startswith("ACTION:"):
            action = llm_decision.split(":")[1].strip()
            tool_func = self.tools.get(action)
            if tool_func:
                # Simplified parameter extraction based on inquiry
                params = {}
                if action == "order_status_check":
                    order_id = next((word for word in inquiry.split() if word.startswith("ORD")), "ORD456") # Mock ID
                    params = {"order_id": order_id}
                elif action == "crm_lookup":
                    customer_id = next((word for word in inquiry.split() if word.startswith("cust")), "cust123") # Mock ID
                    params = {"customer_id": customer_id}
                elif action == "update_delivery_address":
                    order_id = next((word for word in inquiry.split() if word.startswith("ORD")), "ORD456")
                    # Very basic address extraction mock
                    if "to" in inquiry.lower():
                        parts = inquiry.lower().split("to")
                        new_address = parts[1].strip() if len(parts) > 1 else "123 Main St"
                    else:
                        new_address = "123 Main St, Anytown"
                    params = {"order_id": order_id, "new_address": new_address}
                elif action == "retrieve_from_kb":
                    params = {"query": inquiry}

                tool_result = tool_func(**params)
                response = f"[Tool Result] {tool_result}"
                
                # After tool use, LLM might generate a more natural response
                follow_up_prompt = f"Based on the inquiry '{inquiry}' and tool result '{tool_result}', provide a customer-friendly response."
                final_llm_response = self._call_llm(follow_up_prompt, context=tool_result)
                if not final_llm_response.startswith("ACTION:"):
                    response = final_llm_response.replace("LLM Response: ", "")

            else:
                response = f"[Error] Unknown action requested: {action}"
        else:
            response = llm_decision.replace("LLM Response: ", "")
        
        return response

    def handle_inquiry(self, inquiry):
        """Main method to handle a customer inquiry."""
        print(f"\n--- Handling new inquiry: '{inquiry}' ---")

        # Update short-term memory (conversation history)
        self.conversation_history.append({"role": "user", "content": inquiry})

        # Plan and execute the necessary actions
        agent_response = self._plan_and_execute(inquiry)
        
        # Update conversation history with agent's response
        self.conversation_history.append({"role": "agent", "content": agent_response})

        print(f"--- Agent Response: {agent_response} ---")
        return agent_response

# --- Demo Usage --- 
if __name__ == "__main__":
    agent = ComposableAIAgent()

    # Test Case 1: Order status inquiry
    agent.handle_inquiry("What is the status of my order ORD456?")

    # Test Case 2: Customer information lookup
    agent.handle_inquiry("Can you tell me about customer cust123?")

    # Test Case 3: Update delivery address (requires planning and tool use)
    agent.handle_inquiry("I need to change the delivery address for order ORD456 to 789 Oak Ave, Cityville.")

    # Test Case 4: Knowledge Base query
    agent.handle_inquiry("What is your return policy?")
    
    # Test Case 5: A more general inquiry
    agent.handle_inquiry("Hello, I have a general question.")

    print("\n--- Final Conversation History ---")
    for entry in agent.conversation_history:
        print(f"{entry['role'].capitalize()}: {entry['content']}")

    print("\n--- Final Customer Context ---")
    print(agent.customer_context)
