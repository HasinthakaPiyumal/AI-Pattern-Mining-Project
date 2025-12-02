
class MockLLM:
    def invoke(self, prompt):
        if "escalate" in prompt.lower() or "complex issue" in prompt.lower():
            return "This seems like a complex issue requiring escalation. Please provide further details to a human agent."
        elif "past interactions" in prompt.lower() and "customer ID" in prompt.lower():
            return f"Based on your customer ID, your last interaction was about X and the issue was resolved by Y. How can I help you today?" # Simplified
        elif "product" in prompt.lower() and "feature" in prompt.lower():
            return "Let me check the product knowledge base for details on that feature."
        return "I am a simulated AI assistant. How can I assist you with your query?"

class MockOpenAIEmbeddings:
    def embed_documents(self, texts):
        return [[0.1] * 10 for _ in texts] # Dummy embeddings

    def embed_query(self, text):
        return [0.2] * 10 # Dummy embedding

class MockDocument:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}

class MockRetriever:
    def __init__(self, documents):
        self.documents = documents

    def invoke(self, query):
        # Simple keyword-based retrieval for demonstration
        relevant_docs = []
        for doc in self.documents:
            if any(keyword.lower() in doc.page_content.lower() for keyword in query.split()):
                relevant_docs.append(doc)
        return relevant_docs

class MockChroma:
    def __init__(self):
        self.documents = []

    def from_documents(self, documents, embedding):
        self.documents.extend(documents)
        return self

    def as_retriever(self):
        return MockRetriever(self.documents)

class MockConversationBufferWindowMemory:
    def __init__(self, k=5):
        self.k = k
        self.buffer = []

    def save_context(self, inputs, outputs):
        self.buffer.append({"human": inputs["input"], "ai": outputs["output"]})
        if len(self.buffer) > self.k:
            self.buffer = self.buffer[-self.k:]

    def load_memory_variables(self, inputs=None):
        history_str = ""
        for turn in self.buffer:
            history_str += f"Human: {turn['human']}\nAI: {turn['ai']}\n"
        return {"history": history_str.strip()}

class CustomerSupportAgent:
    def __init__(self):
        self.llm = MockLLM()
        self.embeddings = MockOpenAIEmbeddings()
        self.short_term_memory = MockConversationBufferWindowMemory(k=3)

        self.customer_interaction_history = {}

        self.product_knowledge_base = MockChroma().from_documents(
            [
                MockDocument("FAQ: How to reset your password? Go to settings -> security -> reset password.", {"source": "faq"}),
                MockDocument("Feature Guide: Data Export functionality. Supports CSV and JSON formats.", {"source": "guide"}),
                MockDocument("Troubleshooting: Login issues. Clear browser cache or try a different browser.", {"source": "troubleshooting"}),
            ],
            self.embeddings
        )

        self.internal_playbooks = MockChroma().from_documents(
            [
                MockDocument("Escalation Procedure: For critical outages, contact team lead on Slack channel #critical-support.", {"type": "procedure"}),
                MockDocument("Refund Policy: Refunds are processed within 5-7 business days after approval from finance.", {"type": "policy"}),
            ],
            self.embeddings
        )

    def _extract_customer_id(self, query):
        import re
        match = re.search(r"customer ID: ([a-zA-Z0-9-]+)", query, re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    def _extract_keywords(self, query):
        return query.lower().split()

    def _retrieve_customer_history(self, customer_id):
        return self.customer_interaction_history.get(customer_id, "No prior interactions found.")

    def _retrieve_from_knowledge_base(self, query):
        docs = self.product_knowledge_base.as_retriever().invoke(query)
        return "\n".join([doc.page_content for doc in docs]) if docs else "No relevant product information found."

    def _retrieve_from_playbooks(self, query):
        docs = self.internal_playbooks.as_retriever().invoke(query)
        return "\n".join([doc.page_content for doc in docs]) if docs else "No relevant internal notes found."

    def process_query(self, user_query, customer_id=None):
        initial_customer_id = customer_id or self._extract_customer_id(user_query)

        # Retrieve short-term memory (before updating for the current turn)
        st_memory = self.short_term_memory.load_memory_variables()["history"]

        customer_history_info = ""
        if initial_customer_id:
            customer_history_info = self._retrieve_customer_history(initial_customer_id)
            if customer_history_info == "No prior interactions found.":
                customer_history_info = f"Customer ID {initial_customer_id} is new or has no recorded history."
            else:
                customer_history_info = f"Customer ID {initial_customer_id} prior interactions: {customer_history_info}"

        knowledge_base_info = self._retrieve_from_knowledge_base(user_query)
        playbook_info = self._retrieve_from_playbooks(user_query)

        # Construct the augmented prompt
        augmented_prompt = f"""
        You are an AI Customer Support Agent for a B2B SaaS product. Respond to the user's query comprehensively.

        Current Conversation History (Short-Term Memory):
        {st_memory}

        Customer History:
        {customer_history_info}

        Product Knowledge Base Information:
        {knowledge_base_info}

        Internal Notes/Playbooks:
        {playbook_info}

        User Query: {user_query}
        AI Assistant:
        """

        llm_response = self.llm.invoke(augmented_prompt)

        # Update short-term memory with the current turn
        self.short_term_memory.save_context({"input": user_query}, {"output": llm_response})

        # Update customer interaction history
        if initial_customer_id:
            current_interaction = f"Human: {user_query}\nAI: {llm_response}"
            if initial_customer_id not in self.customer_interaction_history:
                self.customer_interaction_history[initial_customer_id] = []
            self.customer_interaction_history[initial_customer_id].append(current_interaction)

        return llm_response


if __name__ == "__main__":
    agent = CustomerSupportAgent()

    print("--- Scenario 1: New Customer, Product Query ---")
    response1 = agent.process_query("How do I export data from your platform?")
    print(f"Agent: {response1}")
    print("\n")

    print("--- Scenario 2: Existing Customer, Troubleshooting Login ---")
    response2 = agent.process_query("I can't log in. My customer ID is CUST-001.")
    print(f"Agent: {response2}")
    print("\n")

    print("--- Scenario 3: Follow-up on previous issue for CUST-001 (demonstrates short-term memory) ---")
    response3 = agent.process_query("What about clearing my cache, will that help?", customer_id="CUST-001")
    print(f"Agent: {response3}")
    print("\n")

    print("--- Scenario 4: Query requiring escalation ---")
    response4 = agent.process_query("We have a critical system outage, all services are down!")
    print(f"Agent: {response4}")
    print("\n")

    print("--- Scenario 5: Another query for CUST-001 (demonstrates long-term customer history) ---")
    response5 = agent.process_query("What was my last issue about, customer ID: CUST-001?")
    print(f"Agent: {response5}")
    print("\n")

    print("--- Scenario 6: General product FAQ ---")
    response6 = agent.process_query("How do I reset my password?")
    print(f"Agent: {response6}")
    print("\n")
