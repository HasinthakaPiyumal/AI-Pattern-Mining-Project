
# customer_support_agent.py

import random
import time

# --- Mock External Tools --- 

# Simulate a CRM system
def mock_crm_lookup(customer_id: str) -> dict:
    """Simulates looking up customer information in a CRM.
    Returns a dictionary with customer details or an empty dict if not found.
    """
    print(f"[Tool Call: CRM Lookup] Searching for customer ID: {customer_id}")
    time.sleep(0.5) # Simulate network latency
    mock_data = {
        "CUST001": {"name": "Alice Smith", "email": "alice@example.com", "plan": "Premium", "recent_tickets": ["TKT789", "TKT101"]},
        "CUST002": {"name": "Bob Johnson", "email": "bob@example.com", "plan": "Standard", "recent_tickets": ["TKT202"]},
    }
    return mock_data.get(customer_id.upper(), {})

# Simulate a Knowledge Base
def mock_knowledge_base_search(query: str) -> list:
    """Simulates searching a knowledge base for relevant articles.
    Returns a list of article titles/summaries.
    """
    print(f"[Tool Call: Knowledge Base] Searching for: '{query}'")
    time.sleep(0.7)
    mock_articles = {
        "password reset": ["How to Reset Your Password", "Troubleshooting Login Issues"],
        "billing": ["Understanding Your Bill", "Payment Methods", "Subscription Plans FAQs"],
        "product features": ["New Features in Version X.Y", "Getting Started Guide"],
        "shipping": ["Shipping Policies", "Order Tracking Information"],
        "return policy": ["Our Return and Refund Policy"],
        "default": ["General FAQs", "Contact Support"]
    }
    # Simple keyword matching for demonstration
    results = []
    for keyword, articles in mock_articles.items():
        if keyword in query.lower():
            results.extend(articles)
    if not results:
        results = mock_articles["default"]
    return list(set(results)) # Remove duplicates

# Simulate a Ticketing System
def mock_create_ticket(customer_id: str, issue_description: str) -> str:
    """Simulates creating a new support ticket.
    Returns a new ticket ID.
    """
    print(f"[Tool Call: Ticketing System] Creating ticket for {customer_id}: {issue_description}")
    time.sleep(0.3)
    new_ticket_id = f"TKT{random.randint(1000, 9999)}"
    return new_ticket_id

# --- LLM Simulation --- 

class LLMSimulator:
    """Simulates an LLM's response generation, reasoning, and confidence.
    This stand-in replaces actual LLM inference for demonstration.
    """
    def __init__(self):
        self.abstention_threshold = 0.6 # Confidence below which the agent abstains

    def _analyze_query(self, query: str, context: str) -> dict:
        """Internal method to simulate LLM understanding and intent.
        Generates a simulated response, reasoning, and confidence.
        """
        lower_query = query.lower()
        response = ""
        reasoning = ""
        confidence = random.uniform(0.7, 0.95) # Default high confidence
        tool_calls = []
        
        if "password" in lower_query or "login issue" in lower_query:
            response = "It sounds like you're having trouble with your password or logging in. I can help with that."
            reasoning = "Identified keywords related to password/login; suggesting KB search for solution."
            tool_calls.append({"tool": "knowledge_base", "query": "password reset"})
            if "customer id" in lower_query:
                response += " Could you provide your customer ID so I can look up your account?"
                tool_calls.append({"tool": "crm_lookup", "query": "customer_id_placeholder"})
        elif "bill" in lower_query or "payment" in lower_query or "subscription" in lower_query:
            response = "I understand you have questions about your billing or subscription."
            reasoning = "Keywords indicate a billing or subscription inquiry; suggesting KB search and CRM lookup if customer ID available."
            tool_calls.append({"tool": "knowledge_base", "query": "billing"})
            if "customer id" in lower_query:
                response += " To assist you further, please provide your customer ID."
                tool_calls.append({"tool": "crm_lookup", "query": "customer_id_placeholder"})
        elif "issue" in lower_query or "problem" in lower_query or "not working" in lower_query:
            response = "I'm sorry to hear you're experiencing an issue. Can you describe it in more detail?"
            reasoning = "Detected general problem keywords; prompting for more details or suggesting ticket creation."
            confidence = random.uniform(0.5, 0.8) # Lower confidence for vague issues
            tool_calls.append({"tool": "create_ticket_if_unresolved", "description": query})
        elif "thanks" in lower_query or "thank you" in lower_query:
            response = "You're welcome! Is there anything else I can assist you with today?"
            reasoning = "Acknowledging gratitude."
            confidence = 1.0
        elif "who are you" in lower_query or "what can you do" in lower_query:
            response = "I am an intelligent customer support agent designed to help you with common queries, look up information, and create support tickets if needed. I aim to be transparent in my reasoning."
            reasoning = "Providing self-introduction."
            confidence = 1.0
        else:
            response = "I'm not entirely sure how to help with that specific request. Could you rephrase it or provide more details?"
            reasoning = "Query was too vague or outside known scope. Recommending clarification or abstention."
            confidence = random.uniform(0.4, 0.65) # Lower confidence for unknown queries
            if confidence < self.abstention_threshold: # Simulate LLM saying "I don't know"
                response = "I apologize, but I don't have enough information or certainty to provide a precise answer for that at the moment. Would you like me to connect you with a human agent or search our knowledge base for general topics?"
                reasoning = "Abstained due to low confidence."

        return {"response": response, "reasoning": reasoning, "confidence": confidence, "tool_calls": tool_calls}

    def generate_response(self, query: str, chat_history: list) -> dict:
        """Generates a simulated LLM response, reasoning, and confidence.
        Incorporates a simplified form of context from chat_history.
        """
        context = " ".join([msg for _, msg in chat_history[-2:]]) if chat_history else ""
        return self._analyze_query(query, context)

# --- Agent Core Logic --- 

class IntelligentCustomerSupportAgent:
    """Orchestrates LLM (simulated) and external tools for customer support.
    Focuses on explainability, trustworthiness, and controlled abstention.
    """
    def __init__(self):
        self.llm_simulator = LLMSimulator()
        self.chat_history = []
        self.customer_id = None # To store identified customer ID
        self.feedback_log = []

    def process_query(self, user_query: str) -> dict:
        """Processes a user query through the agent, including LLM and tool calls.
        Returns a structured response with explanation and confidence.
        """
        self.chat_history.append(("user", user_query))
        print(f"\n[Agent] Processing user query: '{user_query}'")

        # Step 1: LLM (Simulated) understanding and initial response generation
        llm_output = self.llm_simulator.generate_response(user_query, self.chat_history)
        agent_response = llm_output["response"]
        agent_reasoning = llm_output["reasoning"]
        agent_confidence = llm_output["confidence"]
        tool_calls = llm_output["tool_calls"]
        
        executed_tools_info = []

        # Step 2: Tool Execution based on LLM's simulated intent
        for tool_call in tool_calls:
            tool_name = tool_call["tool"]
            
            if tool_name == "crm_lookup":
                # Try to extract customer ID from query or history if placeholder is used
                customer_id_for_lookup = self.customer_id 
                if "customer id is" in user_query.lower():
                    parts = user_query.lower().split("customer id is")
                    if len(parts) > 1:
                        customer_id_for_lookup = parts[1].strip().split()[0] # Take first word after 