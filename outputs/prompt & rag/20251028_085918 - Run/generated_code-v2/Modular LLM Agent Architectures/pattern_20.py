import os
from collections import deque

# Mock external dependencies for demonstration
class MockKnowledgeBase:
    def __init__(self, docs):
        self.documents = docs

    def search(self, query):
        # Simple keyword-based search for demonstration
        results = [doc for doc in self.documents if query.lower() in doc.lower()]
        return results if results else ["No exact match found in knowledge base."]

class MockCRM:
    def __init__(self, customer_data):
        self.customer_data = customer_data

    def get_customer_info(self, customer_id):
        return self.customer_data.get(customer_id, "Customer not found.")

class MockLLM:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.model_name = model_name

    def invoke(self, prompt):
        # Simulate LLM response based on prompt content
        if "summarize the key information" in prompt.lower():
            return {
                "response": "Mock summary: Customer wants to track order. Order number ORD123. KB suggests 'Order Status' page. CRM shows recent purchase. Sentiment is neutral."
            }
        elif "plan a helpful and professional customer support response" in prompt.lower():
            return {
                "response": "\nPlan: 1. Acknowledge inquiry. 2. Provide link to order tracking. 3. Offer further assistance.\nResponse: Hello! I understand you're looking for an update on your order. Please visit our shipping status page at [Link] and enter your order number. If you have any further questions, feel free to ask. Thank you!"
            }
        return {"response": "Mock LLM response for: " + prompt}

# --- Core Agent Components ---

def simple_sentiment_analysis(text):
    """A very basic sentiment analysis for demonstration."""
    text_lower = text.lower()
    if "issue" in text_lower or "problem" in text_lower or "unhappy" in text_lower:
        return "negative"
    elif "thank" in text_lower or "great" in text_lower or "happy" in text_lower:
        return "positive"
    else:
        return "neutral"

class CognitiveLoadAgent:
    def __init__(self, knowledge_base, crm_system, llm):
        self.knowledge_base = knowledge_base
        self.crm_system = crm_system
        self.llm = llm # Using a single LLM instance for simplicity

    def stage_1_information_gathering(self, customer_query, customer_id=None):
        """
        Stage 1: Gathers and analyzes information.
        """
        print(f"\n--- Stage 1: Information Gathering for query: '{customer_query}' ---")

        # 1. Querying a knowledge base
        kb_results = self.knowledge_base.search(customer_query)
        print(f"KB Results: {kb_results}")

        # 2. Accessing CRM data
        crm_info = "No customer ID provided."
        if customer_id:
            crm_info = self.crm_system.get_customer_info(customer_id)
            print(f"CRM Info for {customer_id}: {crm_info}")

        # 3. Sentiment analysis
        sentiment = simple_sentiment_analysis(customer_query)
        print(f"Sentiment: {sentiment}")

        # 4. Identifying keywords and intent (simple for demo)
        keywords = [word for word in customer_query.lower().split() if len(word) > 3 and word not in ["the", "a", "is", "of", "to", "for"]]
        print(f"Keywords: {keywords}")

        # Synthesize information for the LLM to summarize
        information_summary_prompt = (
            f"Customer Query: {customer_query}\n"
            f"Customer ID: {customer_id if customer_id else 'N/A'}\n"
            f"Knowledge Base Findings: {', '.join(kb_results)}\n"
            f"CRM Data: {crm_info}\n"
            f"Sentiment: {sentiment}\n"
            f"Extracted Keywords: {', '.join(keywords)}\n\n"
            "Based on the above, summarize the key information and the likely customer intent. Focus on what is necessary to formulate a response."
        )

        # Use LLM to summarize and structure the gathered info
        llm_response = self.llm.invoke(information_summary_prompt)
        print(f"LLM's Information Summary:\n{llm_response['response']}")

        return {
            "customer_query": customer_query,
            "customer_id": customer_id,
            "kb_results": kb_results,
            "crm_info": crm_info,
            "sentiment": sentiment,
            "keywords": keywords,
            "llm_summary": llm_response['response']
        }

    def stage_2_response_planning_and_generation(self, gathered_info):
        """
        Stage 2: Plans and generates a response based on gathered information.
        """
        print("\n--- Stage 2: Response Planning and Generation ---")

        planning_prompt = (
            f"Based on the following gathered information, plan a helpful and professional customer support response.\n"
            f"**Information Summary:**\n{gathered_info['llm_summary']}\n\n"
            f"Customer's Original Query: {gathered_info['customer_query']}\n"
            f"Consider brand guidelines: Be polite, concise, and offer clear next steps.\n\n"
            "First, outline a plan (e.g., Acknowledge, Provide Solution, Offer Further Help). "
            "Then, generate the actual response based on the plan."
        )

        # Use LLM for planning and response generation
        llm_response = self.llm.invoke(planning_prompt)
        print(f"LLM's Plan and Response:\n{llm_response['response']}")

        return {
            "final_response": llm_response['response'] # Simulating combined plan and response
        }

    def handle_inquiry(self, customer_query, customer_id=None):
        """
        Orchestrates the two-stage process.
        """
        print(f"\n--- Handling new inquiry for Customer ID: {customer_id if customer_id else 'N/A'} ---")
        print(f"Customer says: '{customer_query}'")

        # Execute Stage 1
        gathered_info = self.stage_1_information_gathering(customer_query, customer_id)

        # Execute Stage 2
        final_output = self.stage_2_response_planning_and_generation(gathered_info)

        print("\n--- Inquiry Handling Complete ---")
        print(f"Agent's Proposed Response:\n{final_output['final_response']}")
        return final_output['final_response']

# --- Main Execution ---
if __name__ == "__main__":
    # Initialize Mock Components
    mock_kb_docs = [
        "How to track your order: Visit our 'Order Status' page and enter your order number.",
        "Returns policy: Items can be returned within 30 days with original receipt.",
        "Contact support: Reach us via live chat or phone during business hours.",
        "Shipping delays: Due to high volume, some orders may experience slight delays."
    ]
    mock_knowledge_base = MockKnowledgeBase(mock_kb_docs)

    mock_crm_data = {
        "CUST123": {"name": "Alice Smith", "recent_order": "ORD123", "status": "Gold"},
        "CUST456": {"name": "Bob Johnson", "recent_order": "ORD456", "status": "Silver"}
    }
    mock_crm_system = MockCRM(mock_crm_data)

    mock_llm = MockLLM() # Using our custom mock LLM

    # Initialize the Cognitive Load Agent
    agent = CognitiveLoadAgent(mock_knowledge_base, mock_crm_system, mock_llm)

    # Test inquiries
    print("\n--- Test Case 1: Order Tracking ---")
    agent.handle_inquiry("Where is my order? My order number is ORD123.", customer_id="CUST123")

    print("\n\n--- Test Case 2: Return Policy ---")
    agent.handle_inquiry("What is your return policy?", customer_id="CUST456")

    print("\n\n--- Test Case 3: General Question (No specific customer ID) ---")
    agent.handle_inquiry("How do I contact customer support?")