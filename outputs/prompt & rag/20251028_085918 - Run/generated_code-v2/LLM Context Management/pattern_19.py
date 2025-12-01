import collections

class MockLLM:
    def generate_response(self, prompt):
        if "shipping policy" in prompt.lower():
            return "Our standard shipping takes 3-5 business days. Expedited options are available at checkout."
        elif "return policy" in prompt.lower():
            return "You can return most items within 30 days of purchase with a valid receipt."
        elif "product A" in prompt.lower():
            return "Product A is a high-quality electronic gadget known for its durability and sleek design."
        elif "product B" in prompt.lower():
            return "Product B is a best-seller, ideal for home use, with excellent user reviews."
        elif "past interaction" in prompt.lower() and "order status" in prompt.lower():
            return "Based on your past interaction, you inquired about order #12345. It is currently in transit."
        elif "thank you" in prompt.lower() or "bye" in prompt.lower():
            return "You're welcome! Feel free to ask if you have more questions. Goodbye!"
        return "I'm here to help with your inquiries. How can I assist you further today?"

class ShortTermMemory:
    def __init__(self, max_context_length=5):
        self.max_context_length = max_context_length
        self.conversation_history = collections.deque(maxlen=max_context_length)

    def add_message(self, speaker, message):
        self.conversation_history.append((speaker, message))

    def get_history(self):
        return list(self.conversation_history)

    def clear(self):
        self.conversation_history.clear()

class LongTermMemory:
    def __init__(self):
        self.knowledge_base = {
            "shipping policy": "Our standard shipping takes 3-5 business days. Expedited options are available at checkout.",
            "return policy": "You can return most items within 30 days of purchase with a valid receipt. Items must be in original condition.",
            "product A description": "Product A is a high-quality electronic gadget featuring a 10-inch screen, 128GB storage, and a long-lasting battery. Ideal for professionals.",
            "product B description": "Product B is a popular household appliance, known for its energy efficiency and user-friendly interface. Comes with a 2-year warranty.",
            "customer service hours": "Our customer service is available Monday to Friday, 9 AM to 5 PM EST."
        }
        self.customer_interaction_history = {}

    def retrieve_knowledge(self, query):
        retrieved_info = []
        query_lower = query.lower()
        for key, value in self.knowledge_base.items():
            if any(word in query_lower for word in key.split()) or any(word in query_lower for word in value.lower().split()):
                retrieved_info.append(value)
        return "\n".join(retrieved_info)

    def summarize_conversation(self, conversation_history):
        if not conversation_history:
            return "No conversation to summarize."
        
        last_user_query = ""
        last_agent_response = ""
        
        for speaker, message in reversed(conversation_history):
            if speaker == "User" and not last_user_query:
                last_user_query = message
            elif speaker == "Agent" and not last_agent_response:
                last_agent_response = message
            if last_user_query and last_agent_response:
                break

        if last_user_query and last_agent_response:
            return f"User inquired about: '{last_user_query}'. Agent responded: '{last_agent_response}'."
        elif last_user_query:
            return f"User inquired about: '{last_user_query}'. No agent response yet."
        else:
            return "Conversation summary: Unclear or initial interaction."

    def store_customer_summary(self, customer_id, summary):
        if customer_id not in self.customer_interaction_history:
            self.customer_interaction_history[customer_id] = []
        self.customer_interaction_history[customer_id].append(summary)

    def get_customer_history(self, customer_id):
        return self.customer_interaction_history.get(customer_id, [])


class CustomerSupportAgent:
    def __init__(self):
        self.llm = MockLLM()
        self.short_term_memory = ShortTermMemory()
        self.long_term_memory = LongTermMemory()
        self.current_customer_id = None

    def start_conversation(self, customer_id):
        self.current_customer_id = customer_id
        self.short_term_memory.clear()
        print(f"--- Starting new conversation for Customer ID: {customer_id} ---")

    def converse(self, user_query):
        if not self.current_customer_id:
            print("Error: Conversation not started. Call start_conversation first.")
            return ""

        self.short_term_memory.add_message("User", user_query)

        # Retrieve relevant information from long-term memory
        knowledge_context = self.long_term_memory.retrieve_knowledge(user_query)
        customer_history = self.long_term_memory.get_customer_history(self.current_customer_id)
        customer_history_str = "\n".join(customer_history) if customer_history else "No past interactions."

        # Construct prompt for the LLM
        prompt_parts = [
            "You are a helpful customer support agent for an e-commerce platform.",
            "Current conversation:",
            "\n".join([f"{s}: {m}" for s, m in self.short_term_memory.get_history()]),
            "\nRelevant knowledge base information:",
            knowledge_context if knowledge_context else "No specific knowledge found.",
            "\nCustomer's past interactions:",
            customer_history_str,
            f"\nUser's current query: {user_query}",
            "\nAgent:"
        ]
        full_prompt = "\n".join(prompt_parts)
        
        # Get response from LLM
        agent_response = self.llm.generate_response(full_prompt)
        
        self.short_term_memory.add_message("Agent", agent_response)
        return agent_response

    def end_conversation(self):
        if not self.current_customer_id:
            print("Error: No active conversation to end.")
            return
        
        summary = self.long_term_memory.summarize_conversation(self.short_term_memory.get_history())
        self.long_term_memory.store_customer_summary(self.current_customer_id, summary)
        self.short_term_memory.clear()
        print(f"--- Conversation for Customer ID: {self.current_customer_id} ended. Summary stored. ---")
        print(f"Stored summary: {summary}")
        self.current_customer_id = None

if __name__ == "__main__":
    agent = CustomerSupportAgent()

    # Scenario 1: New customer, asking about policies and products
    print("\n--- Scenario 1: New Customer, general inquiries ---")
    agent.start_conversation("customer_001")
    print(f"Agent: {agent.converse('What is your shipping policy?')}")
    print(f"Agent: {agent.converse('And your return policy?')}")
    print(f"Agent: {agent.converse('Tell me about product A.')}")
    agent.end_conversation()

    # Scenario 2: Returning customer, asking about a product, then a past order
    print("\n--- Scenario 2: Returning Customer, product inquiry and past history ---")
    agent.start_conversation("customer_001")
    print(f"Agent: {agent.converse('Hello again, I was looking at product B.')}")
    print(f"Agent: {agent.converse('Can you tell me if my last order, number 12345, has shipped yet? (simulating retrieval of past interaction details)')}")
    print(f"Agent: {agent.converse('Thanks, that helps!')}")
    agent.end_conversation()

    # Scenario 3: Another customer, asking about product A again
    print("\n--- Scenario 3: Another Customer, asking about product A ---")
    agent.start_conversation("customer_002")
    print(f"Agent: {agent.converse('I need information on product A.')}")
    agent.end_conversation()

    # Demonstrate retrieving stored history for customer_001
    print("\n--- Customer 001's Full Interaction History ---")
    for summary in agent.long_term_memory.get_customer_history("customer_001"):
        print(f"- {summary}")

    print("\n--- Customer 002's Full Interaction History ---")
    for summary in agent.long_term_memory.get_customer_history("customer_002"):
        print(f"- {summary}")
