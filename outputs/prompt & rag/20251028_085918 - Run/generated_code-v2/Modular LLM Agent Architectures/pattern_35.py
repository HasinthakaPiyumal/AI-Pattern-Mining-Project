
class SimulatedLLM:
    """Simulates a blackbox LLM with basic rule-based responses."""
    def __init__(self):
        pass

    def generate_response(self, prompt, context):
        # Simulate LLM processing
        if "escalate" in prompt.lower() or "complex issue" in prompt.lower():
            return "LLM Suggestion: This seems like a complex issue. Consider escalating to a human agent. Also, I found some potential information in the knowledge base related to this topic." # Simulate LLM suggesting escalation
        elif "order status" in prompt.lower() and "tracking number" in context.get("knowledge", "").lower():
            return f"LLM Suggestion: Based on tracking number {context['tracking_number']}, the order status is 'shipped' and expected by tomorrow. Do you want me to update the CRM?" # Simulate LLM using retrieved knowledge
        elif "product warranty" in prompt.lower() and "warranty policy" in context.get("knowledge", "").lower():
            return "LLM Suggestion: The warranty for this product is 2 years. Please refer to the full warranty policy document for details." # Simulate LLM using retrieved knowledge
        elif "refund policy" in prompt.lower():
            return "LLM Suggestion: Our refund policy allows returns within 30 days for a full refund." # Simple LLM response
        else:
            return "LLM Suggestion: I'm not sure, but I can check our knowledge base for more information or escalate if needed."


class WorkingMemory:
    """Stores and manages the conversation history and current context."""
    def __init__(self):
        self.history = []
        self.current_context = {}

    def add_turn(self, speaker, text):
        self.history.append({"speaker": speaker, "text": text})

    def update_context(self, key, value):
        self.current_context[key] = value

    def get_full_history(self):
        return "\n".join([f"{turn['speaker']}: {turn['text']}" for turn in self.history])

    def get_context(self):
        return self.current_context

    def clear_context(self):
        self.current_context = {}


class KnowledgeRetriever:
    """Retrieves relevant information from a simulated knowledge base."""
    def __init__(self, faqs, docs):
        self.faqs = faqs
        self.docs = docs

    def retrieve_knowledge(self, query):
        relevant_info = []
        # Simple keyword matching for demonstration
        for q, a in self.faqs.items():
            if any(word in query.lower() for word in q.lower().split()):
                relevant_info.append(f"FAQ: {q} - {a}")
        for doc_name, doc_content in self.docs.items():
            if any(word in query.lower() for word in doc_name.lower().split()) or \
               any(word in query.lower() for word in doc_content.lower().split()):
                relevant_info.append(f"Document ({doc_name}): {doc_content}")
        
        if not relevant_info:
            return "No specific knowledge found."
        return "\n".join(relevant_info)


class PolicyManager:
    """Decides the next action based on LLM output and context."""
    def __init__(self):
        pass

    def decide_action(self, llm_suggestion, current_context):
        action = {"type": "respond", "details": llm_suggestion.replace("LLM Suggestion: ", "")} # Default action

        if "escalate to a human agent" in llm_suggestion.lower():
            action["type"] = "escalate"
            action["details"] = "Escalating to human agent as suggested by LLM."
        elif "update the crm" in llm_suggestion.lower() and "tracking_number" in current_context:
            action["type"] = "perform_crm_update"
            action["details"] = {"customer_id": current_context.get("customer_id", "N/A"), 
                                 "tracking_number": current_context["tracking_number"], 
                                 "status": "order_inquiry_handled"}
        elif "refer to the full warranty policy document" in llm_suggestion.lower():
            action["type"] = "provide_document_link"
            action["details"] = "[Link to Warranty Policy Document]"

        return action


class ActionExecutor:
    """Executes external actions (e.g., CRM updates, sending emails)."""
    def __init__(self):
        pass

    def execute_action(self, action_type, action_details):
        if action_type == "perform_crm_update":
            print(f"Executing CRM Update: Customer ID: {action_details['customer_id']}, Tracking: {action_details['tracking_number']}, Status: {action_details['status']}")
            return "CRM updated successfully."
        elif action_type == "escalate":
            print(f"Escalating issue. Details: {action_details}")
            return "Issue escalated to human agent."
        elif action_type == "provide_document_link":
            print(f"Providing document link: {action_details}")
            return f"Here is the link: {action_details}"
        elif action_type == "respond":
            # This is handled by the main agent loop, not an external system action
            return ""
        else:
            return f"Unknown action type: {action_type}"


class CustomerSupportAgent:
    """Main orchestrator for the Intelligent Customer Support Agent."""
    def __init__(self, llm, working_memory, knowledge_retriever, policy_manager, action_executor):
        self.llm = llm
        self.working_memory = working_memory
        self.knowledge_retriever = knowledge_retriever
        self.policy_manager = policy_manager
        self.action_executor = action_executor

    def process_customer_query(self, query, customer_id="unknown", tracking_number=None):
        # 1. Store user query in working memory
        self.working_memory.add_turn("Customer", query)
        self.working_memory.update_context("customer_id", customer_id)
        if tracking_number:
            self.working_memory.update_context("tracking_number", tracking_number)

        # 2. Retrieve relevant knowledge
        retrieved_knowledge = self.knowledge_retriever.retrieve_knowledge(query)
        self.working_memory.update_context("knowledge", retrieved_knowledge)
        print(f"[Debug] Retrieved Knowledge: {retrieved_knowledge}")

        # 3. Formulate prompt for LLM
        full_history = self.working_memory.get_full_history()
        llm_prompt = f"Conversation History:\n{full_history}\n\nRelevant Knowledge:\n{retrieved_knowledge}\n\nCustomer Query: {query}\nBased on the above, provide a response or suggest an action to resolve the customer's issue."
        
        # 4. Get response/suggestion from LLM
        llm_context = self.working_memory.get_context()
        llm_response = self.llm.generate_response(llm_prompt, llm_context)
        self.working_memory.add_turn("LLM", llm_response)
        print(f"[Debug] LLM Raw Response: {llm_response}")

        # 5. Policy Manager decides action
        action = self.policy_manager.decide_action(llm_response, llm_context)
        print(f"[Debug] Decided Action: {action}")

        # 6. Execute action or respond
        if action["type"] == "respond":
            agent_response = action["details"]
        else:
            action_result = self.action_executor.execute_action(action["type"], action["details"])
            agent_response = f"I have initiated the requested action: {action['type']}. {action_result}. Additionally, {action['details']} "
            # If the LLM's suggestion had a response part, we can include it after the action
            if action["type"] != "escalate" and "LLM Suggestion:" in llm_response:
                agent_response += f"\nMy response: {llm_response.replace('LLM Suggestion: ', '')}"
            
        self.working_memory.add_turn("Agent", agent_response)
        self.working_memory.clear_context() # Clear context for next independent query if desired
        return agent_response


# --- Demonstration --- 
if __name__ == "__main__":
    # Simulated Knowledge Base
    sample_faqs = {
        "What is your return policy?": "You can return items within 30 days of purchase for a full refund.",
        "How do I track my order?": "You will receive a tracking number via email once your order ships. Use it on our tracking page.",
        "Do you offer international shipping?": "Yes, we ship to over 100 countries worldwide. Shipping fees apply."
    }
    sample_docs = {
        "Warranty Policy": "Our products come with a 2-year manufacturer's warranty covering defects.",
        "Product Manual X1": "Detailed instructions for Product X1 setup and troubleshooting."
    }

    # Initialize modules
    sim_llm = SimulatedLLM()
    memory = WorkingMemory()
    kb_retriever = KnowledgeRetriever(sample_faqs, sample_docs)
    policy = PolicyManager()
    executor = ActionExecutor()

    # Initialize the Customer Support Agent
    agent = CustomerSupportAgent(sim_llm, memory, kb_retriever, policy, executor)

    print("\n--- Scenario 1: Simple Information Request with KB Retrieval ---")
    response = agent.process_customer_query("What is your refund policy?", customer_id="C101")
    print(f"Customer: What is your refund policy?")
    print(f"Agent: {response}")
    print(f"\nConversation History:\n{memory.get_full_history()}")

    print("\n--- Scenario 2: Order Status Inquiry with Context and Action ---")
    memory.clear_context() # Start fresh for next scenario demonstration
    memory.history = [] # Clear history for new scenario
    response = agent.process_customer_query("What is the status of my order? My tracking number is TRK12345.", customer_id="C102", tracking_number="TRK12345")
    print(f"Customer: What is the status of my order? My tracking number is TRK12345.")
    print(f"Agent: {response}")
    print(f"\nConversation History:\n{memory.get_full_history()}")

    print("\n--- Scenario 3: Complex Issue leading to Escalation ---")
    memory.clear_context() # Start fresh
    memory.history = []
    response = agent.process_customer_query("I have a complex issue with my product that your FAQs don't cover. I need to speak to someone.", customer_id="C103")
    print(f"Customer: I have a complex issue with my product that your FAQs don't cover. I need to speak to someone.")
    print(f"Agent: {response}")
    print(f"\nConversation History:\n{memory.get_full_history()}")

    print("\n--- Scenario 4: Warranty Information Request ---")
    memory.clear_context() # Start fresh
    memory.history = []
    response = agent.process_customer_query("Can you tell me about the warranty for Product X1?", customer_id="C104")
    print(f"Customer: Can you tell me about the warranty for Product X1?")
    print(f"Agent: {response}")
    print(f"\nConversation History:\n{memory.get_full_history()}")
