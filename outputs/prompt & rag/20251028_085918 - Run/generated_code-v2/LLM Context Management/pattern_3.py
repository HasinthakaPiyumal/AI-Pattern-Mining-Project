class ShortTermMemory:
    def __init__(self):
        self.memory = []

    def add_to_memory(self, interaction):
        self.memory.append(interaction)

    def get_context(self):
        return self.memory

    def clear_memory(self):
        self.memory = []

class LongTermMemory:
    def __init__(self):
        self.product_knowledge_base = {
            "internet": "Our internet plans offer speeds up to 1Gbps. Check our website for current promotions.",
            "mobile": "We have a range of mobile plans, including unlimited data options. You can port your existing number.",
            "tv": "Our TV packages include sports, movies, and family channels. You can also add premium subscriptions."
        }
        self.customer_history_db = {}

    def retrieve_product_info(self, keyword):
        return self.product_knowledge_base.get(keyword.lower(), "I couldn't find information on that specific product.")

    def retrieve_customer_history(self, customer_id):
        return self.customer_history_db.get(customer_id, "No prior interaction history found for this customer.")

    def update_customer_history(self, customer_id, new_interaction_summary):
        if customer_id not in self.customer_history_db:
            self.customer_history_db[customer_id] = []
        self.customer_history_db[customer_id].append(new_interaction_summary)

class MemoryManager:
    def __init__(self):
        self.short_term_memory = ShortTermMemory()
        self.long_term_memory = LongTermMemory()

    def process_query(self, user_query, customer_id=None):
        context = self.short_term_memory.get_context()
        response_parts = []

        if context:
            response_parts.append(f"From our recent chat: {' '.join([i['user_query'] for i in context[-2:] if 'user_query' in i])}. ")

        # Simulate keyword-based long-term memory retrieval
        if "product" in user_query.lower() or "service" in user_query.lower():
            for keyword in self.long_term_memory.product_knowledge_base.keys():
                if keyword in user_query.lower():
                    product_info = self.long_term_memory.retrieve_product_info(keyword)
                    response_parts.append(f"Regarding {keyword}: {product_info} ")
                    break
        
        if customer_id and ("history" in user_query.lower() or "past interactions" in user_query.lower()):
            customer_history = self.long_term_memory.retrieve_customer_history(customer_id)
            response_parts.append(f"Your past interactions: {customer_history} ")

        if not response_parts:
            return f"I received your query: '{user_query}'. How can I assist you further?"
        else:
            return "".join(response_parts).strip() + f"\nAlso, I received your query: '{user_query}'."

    def record_interaction(self, user_query, chatbot_response, customer_id=None):
        self.short_term_memory.add_to_memory({"user_query": user_query, "chatbot_response": chatbot_response})
        
        # Simulate periodically summarizing and updating long-term customer history
        # In a real system, this would involve NLP for summarization and a more sophisticated trigger
        if customer_id and len(self.short_term_memory.get_context()) % 3 == 0: # Update every 3 turns for example
            session_summary = f"Customer inquired about: {user_query}. Chatbot responded: {chatbot_response}"
            self.long_term_memory.update_customer_history(customer_id, session_summary)


class TelecomChatbot:
    def __init__(self):
        self.memory_manager = MemoryManager()

    def converse(self, user_query, customer_id=None):
        simulated_llm_response = self.memory_manager.process_query(user_query, customer_id)
        self.memory_manager.record_interaction(user_query, simulated_llm_response, customer_id)
        return simulated_llm_response

if __name__ == "__main__":
    chatbot = TelecomChatbot()

    print("--- Starting Chatbot Conversation ---")

    # Interaction 1: General query, no customer ID
    user_input_1 = "Hello, what internet plans do you offer?"
    print(f"User: {user_input_1}")
    response_1 = chatbot.converse(user_input_1)
    print(f"Chatbot: {response_1}\n")

    # Interaction 2: Follow-up on product, with customer ID
    user_input_2 = "Tell me more about the mobile services. My customer ID is CUST001."
    print(f"User: {user_input_2}")
    response_2 = chatbot.converse(user_input_2, customer_id="CUST001")
    print(f"Chatbot: {response_2}\n")

    # Interaction 3: Ask about history, with customer ID
    user_input_3 = "What were my past interactions about? My customer ID is CUST001."
    print(f"User: {user_input_3}")
    response_3 = chatbot.converse(user_input_3, customer_id="CUST001")
    print(f"Chatbot: {response_3}\n")

    # Interaction 4: New customer, product query
    user_input_4 = "I'm new here. Can you tell me about your TV packages?"
    print(f"User: {user_input_4}")
    response_4 = chatbot.converse(user_input_4, customer_id="CUST002")
    print(f"Chatbot: {response_4}\n")
    
    # Interaction 5: General query, new customer to trigger long-term memory update
    user_input_5 = "How do I check my bill?"
    print(f"User: {user_input_5}")
    response_5 = chatbot.converse(user_input_5, customer_id="CUST001")
    print(f"Chatbot: {response_5}\n")

    print("--- End of Conversation ---")

    print("\n--- Memory States After Conversation ---")
    print(f"Short-Term Memory (last interaction): {chatbot.memory_manager.short_term_memory.get_context()[-1] if chatbot.memory_manager.short_term_memory.get_context() else 'Empty'}")
    print(f"Long-Term Memory (Customer CUST001 history): {chatbot.memory_manager.long_term_memory.retrieve_customer_history('CUST001')}")
    print(f"Long-Term Memory (Customer CUST002 history): {chatbot.memory_manager.long_term_memory.retrieve_customer_history('CUST002')}")
