import os
from typing import Dict, Any
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain.tools import tool

# Load environment variables (e.g., OPENAI_API_KEY)
load_dotenv()

# --- Simulated External Tools (External Tool Interface Module) ---

def knowledge_base_lookup(query: str) -> str:
    """Looks up information in a simulated knowledge base based on the query."""
    print(f"[TOOL] Knowledge Base Lookup: {query}")
    # Simulate a knowledge base lookup
    if "password reset" in query.lower():
        return "To reset your password, visit our website and click 'Forgot Password'."
    elif "shipping policy" in query.lower():
        return "Our standard shipping takes 3-5 business days. Expedited options are available."
    elif "return policy" in query.lower():
        return "Items can be returned within 30 days of purchase with a valid receipt for a full refund."
    elif "product x troubleshooting" in query.lower():
        return "For Product X troubleshooting, ensure all cables are connected and try restarting the device."
    else:
        return "I couldn't find specific information for that query in the knowledge base. Please rephrase or provide more details."

@tool
def lookup_knowledge_base(query: str) -> str:
    """Useful for looking up information in the company's knowledge base. Input should be a specific question or topic related to support."""
    return knowledge_base_lookup(query)


def crm_lookup(customer_id: str) -> Dict[str, Any]:
    """Fetches customer details from a simulated CRM system based on customer ID."""
    print(f"[TOOL] CRM Lookup: {customer_id}")
    # Simulate CRM data
    if customer_id == "CUST123":
        return {"name": "Alice Smith", "email": "alice.s@example.com", "status": "Premium", "last_purchase": "Product X", "issue_history": ["slow internet", "billing query"]}
    elif customer_id == "CUST456":
        return {"name": "Bob Johnson", "email": "bob.j@example.com", "status": "Standard", "last_purchase": "Product Y", "issue_history": ["delivery delay"]}
    else:
        return {"error": f"Customer ID {customer_id} not found in CRM."}

@tool
def lookup_crm_details(customer_id: str) -> Dict[str, Any]:
    """Useful for fetching customer details from the CRM system using a customer ID."""
    return crm_lookup(customer_id)


def order_tracking(order_id: str) -> str:
    """Tracks an order status using a simulated order tracking system."""
    print(f"[TOOL] Order Tracking: {order_id}")
    # Simulate order tracking data
    if order_id == "ORD789":
        return "Order ORD789 is currently out for delivery and expected today."
    elif order_id == "ORD101":
        return "Order ORD101 was delivered on 2023-10-26."
    elif order_id == "ORD555":
        return "Order ORD555 is awaiting shipment from warehouse."
    else:
        return f"Order ID {order_id} not found or invalid."

@tool
def track_customer_order(order_id: str) -> str:
    """Useful for tracking the status of a customer order using an order ID."""
    return order_tracking(order_id)

# List of all available tools for the agent
tools = [lookup_knowledge_base, lookup_crm_details, track_customer_order]

# --- Core LLM Integration ---
# Initialize the LLM. Ensure OPENAI_API_KEY is set in your environment variables.
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# --- Memory Module ---
# Initialize conversation memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True, # Return messages as a list of message objects
    output_key="output" # The key in the agent's output that contains the final response
)

# --- Planning Module & Context Management (Orchestrated by the Agent) ---
# Define the prompt template for the ReAct agent
# This prompt guides the LLM to think, observe, and act.
agent_prompt_template = PromptTemplate.from_template(
    """You are a helpful and efficient Smart Customer Support Agent. Your goal is to assist customers with their queries by using the available tools, remembering past interactions, and providing clear, concise, and accurate information.

Your capabilities include:
- Answering questions based on a knowledge base (`lookup_knowledge_base`).
- Looking up customer details (`lookup_crm_details`).
- Tracking order statuses (`track_customer_order`).

If you need to ask for more information from the user (e.g., order ID, customer ID), do so clearly.
Always aim to resolve the customer's issue or provide the best possible next steps.

Begin! Remember to use the chat history to maintain context.

Previous conversation:
{chat_history}

New Human input: {input}
{agent_scratchpad}"""
)

# Create the ReAct agent
agent = create_react_agent(llm, tools, agent_prompt_template)

# Create the AgentExecutor which runs the agent with memory and tools
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True, # Set to True to see the agent's thought process
    handle_parsing_errors=True # Robustness for tool call errors
)

def run_customer_support_agent(user_query: str) -> str:
    """Handles a customer query using the Smart Customer Support Agent."""
    print(f"\n--- User: {user_query} ---")
    # Invoke the agent executor with the current user query
    # The memory component automatically appends the current input and the agent's response to chat_history
    response = agent_executor.invoke({"input": user_query})
    agent_response = response["output"]
    print(f"--- Agent: {agent_response} ---")
    return agent_response

if __name__ == "__main__":
    print("Welcome to the Smart Customer Support Agent! Type 'exit' to end the conversation.\n")
    print("You can ask me about password resets, shipping, returns, customer details (CUST123, CUST456), or order status (ORD789, ORD101, ORD555).\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Agent: Goodbye!")
            break
        try:
            run_customer_support_agent(user_input)
        except Exception as e:
            print(f"Agent encountered an error: {e}")
        print("\n" + "="*50 + "\n")
