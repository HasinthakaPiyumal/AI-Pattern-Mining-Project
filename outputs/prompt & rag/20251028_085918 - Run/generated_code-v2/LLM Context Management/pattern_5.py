class AdaptiveCustomerSupportAgent:
    def __init__(self):
        self.memory = []

    def _retrieve_from_memory(self, query):
        relevant_memories = []
        for interaction in self.memory:
            if query.lower() in interaction["query"].lower() or \
               query.lower() in interaction["response"].lower() or \
               (interaction.get("learned_solution") and query.lower() in interaction["learned_solution"].lower()):
                relevant_memories.append(interaction)
        return relevant_memories

    def _simulate_llm_response(self, query, retrieved_memory):
        base_response = f"Hello! How can I help you with '{query}' today?"
        if retrieved_memory:
            memory_context = "\n".join([f"- Past Query: {m['query']}, Past Response: {m['response']}{f', Solution: {m["learned_solution"]}' if m.get('learned_solution') else ''}" for m in retrieved_memory])
            base_response = f"Based on our past interactions and solutions related to '{query}', I can tell you:\n{memory_context}\n\n" + base_response
            
            # Simple adaptation: if a solution exists in memory, try to use it directly
            for mem in retrieved_memory:
                if mem.get("learned_solution") and "solution" in query.lower() or "how to fix" in query.lower():
                    return f"It seems you are looking for a solution related to '{query}'. Based on previous cases, the solution is: {mem['learned_solution']}. Is there anything else I can help you with?"

        return base_response

    def _update_memory(self, query, response, learned_solution=None):
        self.memory.append({"query": query, "response": response, "learned_solution": learned_solution})

    def handle_query(self, query):
        retrieved_memory = self._retrieve_from_memory(query)
        llm_response = self._simulate_llm_response(query, retrieved_memory)
        
        # For demonstration, let's assume some queries lead to a 'learned solution'
        learned_solution = None
        if "reset password" in query.lower():
            learned_solution = "Instructions to reset password sent to your registered email."
        elif "check order status" in query.lower():
            learned_solution = "Please visit our 'My Orders' section on the website and enter your order ID."

        self._update_memory(query, llm_response, learned_solution)
        return llm_response

if __name__ == "__main__":
    agent = AdaptiveCustomerSupportAgent()
    print("Adaptive AI Customer Support Agent (Type 'exit' to quit)")

    while True:
        user_query = input("You: ")
        if user_query.lower() == "exit":
            break
        
        response = agent.handle_query(user_query)
        print(f"Agent: {response}")
        # print(f"\nAgent Memory (for debugging): {agent.memory}\n") # Uncomment to see memory state