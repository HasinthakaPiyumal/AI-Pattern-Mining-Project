import collections

class ShortTermMemory:
    def __init__(self, max_history_len=10):
        self.conversation_history = collections.deque(maxlen=max_history_len)
        self.explicit_notes = {}

    def add_message(self, role, content):
        self.conversation_history.append({"role": role, "content": content})

    def get_history(self):
        return list(self.conversation_history)

    def record_note(self, key, value):
        self.explicit_notes[key] = value

    def get_note(self, key):
        return self.explicit_notes.get(key)

    def clear(self):
        self.conversation_history.clear()
        self.explicit_notes.clear()

class CustomerProfileDB:
    def __init__(self):
        self.profiles = {}

    def get_customer_profile(self, customer_id):
        if customer_id not in self.profiles:
            self.profiles[customer_id] = {"history": [], "preferences": {}}
        return self.profiles[customer_id]

    def update_customer_history(self, customer_id, summary):
        profile = self.get_customer_profile(customer_id)
        profile["history"].append(summary)
        self.profiles[customer_id] = profile

    def update_customer_preferences(self, customer_id, preferences):
        profile = self.get_customer_profile(customer_id)
        profile["preferences"].update(preferences)
        self.profiles[customer_id] = profile

class KnowledgeBase:
    def __init__(self):
        self.articles = {
            "product_warranty": "Our product warranty covers manufacturing defects for 1 year.",
            "return_policy": "Returns are accepted within 30 days with original receipt.",
            "troubleshooting_steps": "Please try restarting your device, checking connections, or reinstalling software.",
            "account_setup": "To set up your account, visit our website and click 'Sign Up'."
        }

    def query(self, topic):
        return self.articles.get(topic.lower(), "No information found for that topic in our knowledge base.")

class LongTermMemory:
    def __init__(self, customer_db, knowledge_base):
        self.customer_db = customer_db
        self.knowledge_base = knowledge_base

    def retrieve_knowledge(self, query):
        return self.knowledge_base.query(query)

    def retrieve_customer_history(self, customer_id):
        profile = self.customer_db.get_customer_profile(customer_id)
        return profile["history"]

    def summarize_conversation(self, conversation_history):
        # In a real application, this would involve an LLM call to summarize.
        # For simulation, we'll just create a simple summary.
        last_user_message = "N/A"
        for msg in reversed(conversation_history):
            if msg["role"] == "user":
                last_user_message = msg["content"]
                break
        return f"Interaction Summary: Customer last asked about '{last_user_message}'."

    def update_customer_history(self, customer_id, summary):
        self.customer_db.update_customer_history(customer_id, summary)

class LLMAgent:
    def __init__(self, customer_id, short_term_memory, long_term_memory):
        self.customer_id = customer_id
        self.stm = short_term_memory
        self.ltm = long_term_memory

    def _decide_action(self, user_input):
        if "warranty" in user_input.lower() or "return" in user_input.lower():
            return "retrieve_knowledge"
        if "past issue" in user_input.lower() or "previous interaction" in user_input.lower():
            return "retrieve_customer_history"
        if "account number" in self.stm.explicit_notes:
            return "use_short_term_note"
        return "general_response"

    def process_input(self, user_input):
        self.stm.add_message("user", user_input)
        print(f"\nCustomer ({self.customer_id}): {user_input}")

        action = self._decide_action(user_input)
        agent_response = ""

        if action == "retrieve_knowledge":
            topic = user_input.split("about ")[-1] if "about " in user_input.lower() else user_input # Basic extraction
            knowledge = self.ltm.retrieve_knowledge(topic)
            agent_response = f"I looked up some information for you: {knowledge}"
        elif action == "retrieve_customer_history":
            history = self.ltm.retrieve_customer_history(self.customer_id)
            if history:
                agent_response = f"Based on your past interactions, I see: {'. '.join(history)}"
            else:
                agent_response = "I couldn't find any specific past interactions for you."
        elif action == "use_short_term_note":
            account_num = self.stm.get_note("account_number")
            agent_response = f"Thank you for providing your account number {account_num}. How can I help you with that?"
        else:
            # Simulate LLM processing for general conversation
            if "hello" in user_input.lower():
                agent_response = "Hello! How can I assist you today?"
            elif "account" in user_input.lower() and "number" in user_input.lower():
                agent_response = "Could you please provide your account number? I'll record it for this session."
                # Simulate recording in STM
                self.stm.record_note("account_number", "123456789") # Placeholder for actual input
            elif "thanks" in user_input.lower() or "thank you" in user_input.lower():
                agent_response = "You're welcome! Is there anything else?"
            else:
                agent_response = f"I'm processing your request regarding '{user_input}'. Please bear with me."

        self.stm.add_message("agent", agent_response)
        print(f"Agent: {agent_response}")
        return agent_response

    def end_session(self):
        conversation_summary = self.ltm.summarize_conversation(self.stm.get_history())
        self.ltm.update_customer_history(self.customer_id, conversation_summary)
        self.stm.clear()
        print(f"\nSession ended for customer {self.customer_id}. Conversation summarized and added to long-term memory.\n")

# --- Example Usage ---
if __name__ == "__main__":
    # Initialize memory components
    customer_db = CustomerProfileDB()
    knowledge_base = KnowledgeBase()
    long_term_memory = LongTermMemory(customer_db, knowledge_base)

    # Simulate a customer session
    customer_id_1 = "cust_001"
    short_term_memory_1 = ShortTermMemory()
    agent_1 = LLMAgent(customer_id_1, short_term_memory_1, long_term_memory)

    print("--- Starting Customer 1 Session 1 ---")
    agent_1.process_input("Hello, I have a question about my account.")
    agent_1.process_input("Can you tell me about the return policy?")
    agent_1.process_input("Also, what's my account number?")
    agent_1.process_input("My account number is 98765. I'd like to check my past issue with product X.") # Simulating agent recording '98765'
    agent_1.end_session()

    # Simulate another session with the same customer
    print("--- Starting Customer 1 Session 2 ---")
    short_term_memory_2 = ShortTermMemory() # New short-term memory for new session
    agent_2 = LLMAgent(customer_id_1, short_term_memory_2, long_term_memory)
    agent_2.process_input("Hello again, I need help with a new issue.")
    agent_2.process_input("What was my previous interaction about?")
    agent_2.end_session()

    # Simulate a new customer
    customer_id_2 = "cust_002"
    short_term_memory_3 = ShortTermMemory()
    agent_3 = LLMAgent(customer_id_2, short_term_memory_3, long_term_memory)

    print("--- Starting Customer 2 Session 1 ---")
    agent_3.process_input("I want to know about product warranty.")
    agent_3.process_input("Thanks!")
    agent_3.end_session()

    print("\n--- Final Customer Profiles ---")
    print(f"Customer {customer_id_1} Profile: {customer_db.get_customer_profile(customer_id_1)}")
    print(f"Customer {customer_id_2} Profile: {customer_db.get_customer_profile(customer_id_2)}")