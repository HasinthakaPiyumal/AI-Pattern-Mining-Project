import os
from dotenv import load_dotenv
from typing import Dict, Any

from langchain.agents import AgentExecutor, initialize_agent, AgentType
from langchain_core.tools import BaseTool, tool
from langchain_openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Mock Data for demonstration
INTERNAL_KB = {
    "shipping_policy": "Our standard shipping takes 3-5 business days. Expedited shipping is available for an extra fee.",
    "return_policy": "Items can be returned within 30 days of purchase with a valid receipt. Some exclusions apply.",
    "product_warranty": "All electronics come with a 1-year manufacturer's warranty.",
    "account_setup": "To set up a new account, please visit our website and click 'Sign Up'."
}

CRM_RECORDS = {
    "customer_123": {
        "name": "Alice Wonderland",
        "email": "alice@example.com",
        "order_history": ["ORD1001", "ORD1005"],
        "status": "active"
    },
    "customer_456": {
        "name": "Bob The Builder",
        "email": "bob@example.com",
        "order_history": ["ORD1002"],
        "status": "inactive"
    }
}

# --- Tool Abstraction Layer --- 

@tool
def internal_kb_search(query: str) -> str:
    """Searches the internal company knowledge base for relevant information. Use this tool for queries about company policies, product information, or common FAQs."""
    print(f"\n[DEBUG] InternalKBSearchTool called with query: {query}")
    # Simple string matching for demonstration
    for key, value in INTERNAL_KB.items():
        if query.lower() in key or query.lower() in value.lower():
            return value
    return "No relevant information found in the internal knowledge base."

@tool
def external_web_search(query: str) -> str:
    """Performs a general web search for information not available in internal sources. Use this tool for broad queries or external news/updates."""
    print(f"\n[DEBUG] ExternalWebSearchTool called with query: {query}")
    # Simulate external web search
    if "latest tech news" in query.lower():
        return "The latest tech news includes advancements in AI models and quantum computing."
    if "weather" in query.lower():
        return "The current weather in London is cloudy with a chance of rain."
    return f"Simulated web search result for '{query}': Information about {query} can be found on various online platforms."

@tool
def crm_retrieve_customer_info(customer_id: str) -> Dict[str, Any]:
    """Retrieves customer information from the CRM system using a customer ID. Use this to get details like name, email, order history, or status."""
    print(f"\n[DEBUG] CRMRetrievalTool called for customer_id: {customer_id}")
    info = CRM_RECORDS.get(customer_id)
    if info:
        return info
    return {"error": "Customer not found."}

@tool
def crm_update_customer_status(customer_id: str, new_status: str) -> str:
    """Updates the status of a customer in the CRM system. Requires customer ID and the new status."""
    print(f"\n[DEBUG] CRMUpdateTool called for customer_id: {customer_id}, new_status: {new_status}")
    if customer_id in CRM_RECORDS:
        old_status = CRM_RECORDS[customer_id]["status"]
        CRM_RECORDS[customer_id]["status"] = new_status
        return f"Customer {customer_id} status updated from '{old_status}' to '{new_status}'."
    return "Customer not found. Status update failed."


# --- Meta-Learning Agent (Orchestrator) --- 

def create_adaptive_bot():
    llm = OpenAI(temperature=0.0, openai_api_key=os.getenv("OPENAI_API_KEY"))

    tools = [
        internal_kb_search,
        external_web_search,
        crm_retrieve_customer_info,
        crm_update_customer_status
    ]

    # The agent is initialized with a robust prompt that encourages meta-learning
    # and strategic tool use. It learns *when* to use a type of tool.
    agent_chain = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        agent_kwargs={
            "suffix": """Begin!\nQuestion: {input}\n{agent_scratchpad}\nYour response should be helpful and concise, using the most appropriate tool strategy.""",
            "prefix": """You are an adaptive customer support AI. Your goal is to assist users by strategically using the provided tools. Think about the *type* of information needed or action required to determine the best tool to use, rather than memorizing specific tool names for specific keywords. If a query requires searching for internal information, use a knowledge base search tool. If it's about customer data, use a CRM tool. If it's general external information, use a web search tool. If a tool fails, try to deduce why and explain. Always prioritize using the most specific tool for the task.

Answer the following questions as best you can."""
        }
    )
    return agent_chain

if __name__ == "__main__":
    # Ensure OPENAI_API_KEY is set in your .env file
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set. Please create a .env file with your key.")
    else:
        bot = create_adaptive_bot()
        print("\nAdaptive Customer Support Bot Initialized. Type 'exit' to quit.")
        while True:
            user_query = input("\nUser: ")
            if user_query.lower() == 'exit':
                break
            try:
                response = bot.invoke({"input": user_query})
                print(f"\nBot: {response['output']}")
            except Exception as e:
                print(f"\nBot: An error occurred: {e}")
                print("Please try rephrasing your query or contact support.")
