from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
import os

from langchain.agents import initialize_agent, AgentType, AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool

# --- 1. Pydantic Models for API Interface ---
class ChatRequest(BaseModel):
    user_query: str

class ChatResponse(BaseModel):
    agent_response: str
    conversation_history: List[Dict[str, str]]

# --- 2. Mock Specialized Tools ---
# In a real application, these would interact with actual external systems (databases, CRMs, APIs)

def retrieve_product_info(query: str) -> str:
    """Simulates retrieving product information from a knowledge base."""
    if "laptop" in query.lower():
        return "The XYZ-1000 laptop features an Intel i7 processor, 16GB RAM, and a 512GB SSD. It costs $1200."
    elif "return policy" in query.lower():
        return "Our return policy allows returns within 30 days of purchase, provided the item is in its original condition."
    else:
        return "Could not find specific information for your query in the knowledge base. Please try rephrasing."

def get_customer_order_details(customer_id: str) -> str:
    """Simulates retrieving customer order details from a CRM system."""
    if customer_id == "CUST123":
        return "Customer CUST123 has an order #ORD987 for an XYZ-1000 laptop, placed on 2023-10-26, status: Shipped."
    elif customer_id == "CUST456":
        return "Customer CUST456 has no recent orders."
    else:
        return f"No customer found with ID {customer_id}."

def perform_action(action_details: str) -> str:
    """Simulates performing an action like initiating a refund or updating an order."""
    if "refund" in action_details.lower():
        order_id = action_details.split(" ")[-1]
        return f"Initiated refund process for order {order_id}. Please allow 3-5 business days."
    elif "escalate" in action_details.lower():
        return "Issue escalated to a human agent. A specialist will contact you shortly."
    else:
        return f"Action '{action_details}' performed successfully (mock)."

# --- 3. Langchain Tools --- 
# Wrap the mock functions as Langchain Tools
knowledge_base_tool = Tool(
    name="KnowledgeBaseSearch",
    func=retrieve_product_info,
    description="Useful for answering questions about products, policies, and general information. Input should be a specific query."
)

crm_tool = Tool(
    name="CRMCustomerLookup",
    func=get_customer_order_details,
    description="Useful for retrieving specific customer order details or history. Input should be a customer ID (e.g., 'CUST123')."
)

action_tool = Tool(
    name="CustomerServiceAction",
    func=perform_action,
    description="Useful for performing customer service actions like initiating refunds, updating orders, or escalating issues. Input should be a description of the action and relevant IDs (e.g., 'refund order ORD987')."
)

# List of all tools available to the agent
tools = [knowledge_base_tool, crm_tool, action_tool]

# --- 4. LLM and Memory Module ---
# Ensure OPENAI_API_KEY is set in your environment variables
# Example: export OPENAI_API_KEY='your_openai_api_key'
llm = ChatOpenAI(temperature=0, model="gpt-4") 

# Conversation memory to maintain state across turns
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# --- 5. Agent Initialization ---
# Initialize the agent with the LLM, tools, and memory
# Using AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION for conversational abilities and tool use
agent_chain: AgentExecutor = initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True, # Set to True to see the agent's thought process
    memory=memory,
    handle_parsing_errors=True # To gracefully handle LLM parsing errors
)

# --- 6. FastAPI Application ---
app = FastAPI(
    title="Smart Customer Support Agent API",
    description="An API for an AI-powered customer support agent leveraging a modular LLM framework."
)

@app.post("/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    """Endpoint to interact with the Smart Customer Support Agent."""
    try:
        # Invoke the agent with the user's query
        response = agent_chain.run(input=request.user_query)
        
        # Retrieve current conversation history for the response
        current_history = []
        for msg in memory.buffer_as_messages:
            if hasattr(msg, 'content') and hasattr(msg, 'type'):
                current_history.append({"type": msg.type, "content": msg.content})

        return ChatResponse(agent_response=response, conversation_history=current_history)
    except Exception as e:
        # Basic error handling
        print(f"An error occurred: {e}")
        return ChatResponse(agent_response="I apologize, but I encountered an error. Please try again.", conversation_history=[])

# To run this application:
# 1. Save the code as main.py
# 2. Set your OpenAI API key as an environment variable: export OPENAI_API_KEY='your_key_here'
# 3. Install necessary libraries: pip install fastapi uvicorn langchain openai pydantic
# 4. Run from your terminal: uvicorn main:app --reload
# 5. Access the API at http://127.0.0.1:8000/docs for the Swagger UI.