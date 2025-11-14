import os
import requests
from dotenv import load_dotenv
from loguru import logger
from typing import List, Dict, Any

# Langchain imports
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Streamlit for UI
import streamlit as st

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found in environment variables. Please set it in a .env file.")
    st.stop()

# Configure logger
logger.add("file.log", rotation="500 MB", level="INFO")

### 1. Tool Integration Layer - Simulated APIs ###

# Pydantic models for tool input schemas
class CustomerInfoInput(BaseModel):
    customer_id: str = Field(description="ID of the customer to retrieve information for.")

class KnowledgeBaseSearchInput(BaseModel):
    query: str = Field(description="Search query for the knowledge base.")

class OrderStatusInput(BaseModel):
    order_id: str = Field(description="ID of the order to check status for.")
    customer_id: str = Field(description="ID of the customer associated with the order.")

class ModifyOrderInput(BaseModel):
    order_id: str = Field(description="ID of the order to modify.")
    action: str = Field(description="Action to perform on the order (e.g., 'cancel', 'update_address').")
    details: Dict[str, Any] = Field(description="Details for the modification, e.g., {'new_address': '123 Main St'}.")

@tool("CRM_API_get_customer_info", args_schema=CustomerInfoInput)
def get_customer_info(customer_id: str) -> Dict[str, Any]:
    """Fetches detailed customer information from the CRM system."""
    logger.info(f"CRM_API: Fetching info for customer_id: {customer_id}")
    # Simulate an API call
    if customer_id == "CUST001":
        return {
            "customer_id": "CUST001",
            "name": "Alice Wonderland",
            "email": "alice@example.com",
            "phone": "555-1234",
            "address": "123 Rabbit Hole, Fancyland",
            "loyalty_status": "Gold",
            "last_order_id": "ORD001"
        }
    elif customer_id == "CUST002":
        return {
            "customer_id": "CUST002",
            "name": "Bob The Builder",
            "email": "bob@example.com",
            "phone": "555-5678",
            "address": "456 Construction Site, Buildsville",
            "loyalty_status": "Silver",
            "last_order_id": "ORD003"
        }
    else:
        logger.warning(f"CRM_API: Customer {customer_id} not found.")
        return {"error": "Customer not found", "customer_id": customer_id}

@tool("KnowledgeBase_API_search", args_schema=KnowledgeBaseSearchInput)
def search_knowledge_base(query: str) -> Dict[str, Any]:
    """Searches the internal knowledge base for relevant articles or FAQs."""
    logger.info(f"KnowledgeBase_API: Searching for query: {query}")
    # Simulate searching a knowledge base
    query = query.lower()
    if "return policy" in query:
        return {"article_title": "Return Policy", "content": "Our return policy allows returns within 30 days of purchase with a valid receipt. Items must be in original condition."}
    elif "shipping times" in query:
        return {"article_title": "Shipping Information", "content": "Standard shipping takes 5-7 business days. Express shipping takes 2-3 business days."}
    elif "reset password" in query:
        return {"article_title": "Password Reset Guide", "content": "To reset your password, visit our website and click 'Forgot Password' on the login page. Follow the instructions sent to your email."}
    else:
        logger.warning(f"KnowledgeBase_API: No articles found for query: {query}")
        return {"error": "No relevant articles found", "query": query}

@tool("OrderManagement_API_get_order_status", args_schema=OrderStatusInput)
def get_order_status(order_id: str, customer_id: str) -> Dict[str, Any]:
    """Retrieves the current status and details of a customer's order."""
    logger.info(f"OrderManagement_API: Checking status for order {order_id} for customer {customer_id}")
    # Simulate order status retrieval
    if order_id == "ORD001" and customer_id == "CUST001":
        return {"order_id": "ORD001", "status": "Shipped", "item": "Product A", "tracking_number": "TRK12345", "delivery_date": "2023-10-26"}
    elif order_id == "ORD002" and customer_id == "CUST001":
        return {"order_id": "ORD002", "status": "Processing", "item": "Product B"}
    elif order_id == "ORD003" and customer_id == "CUST002":
        return {"order_id": "ORD003", "status": "Delivered", "item": "Product C"}
    else:
        logger.warning(f"OrderManagement_API: Order {order_id} for customer {customer_id} not found.")
        return {"error": "Order not found", "order_id": order_id, "customer_id": customer_id}

@tool("OrderManagement_API_modify_order", args_schema=ModifyOrderInput)
def modify_order(order_id: str, action: str, details: Dict[str, Any]) -> Dict[str, Any]:
    """Modifies an existing order (e.g., cancel, update address)."""
    logger.info(f"OrderManagement_API: Modifying order {order_id} with action '{action}' and details {details}")
    # Simulate order modification
    if order_id in ["ORD001", "ORD002", "ORD003"]:
        if action == "cancel":
            return {"order_id": order_id, "status": "Cancelled", "message": "Order successfully cancelled."}
        elif action == "update_address" and "new_address" in details:
            return {"order_id": order_id, "status": "Address Updated", "new_address": details["new_address"], "message": "Order address updated."}
        else:
            logger.warning(f"OrderManagement_API: Invalid action '{action}' or missing details for order {order_id}.")
            return {"error": f"Invalid action '{action}' or missing details", "order_id": order_id}
    else:
        logger.warning(f"OrderManagement_API: Order {order_id} not found for modification.")
        return {"error": "Order not found", "order_id": order_id}

# List of all tools available to the agent
tools = [
    get_customer_info,
    search_knowledge_base,
    get_order_status,
    modify_order
]

### 2. LLM Core & 3. Orchestration Layer ###

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=OPENAI_API_KEY)

# Agent Prompt
# The prompt defines the agent's persona and how it should use tools and respond.
# The `MessagesPlaceholder` for `agent_scratchpad` is crucial for Langchain's ReAct agent.
prompt_template = ChatPromptTemplate.from_messages(
    [
        ( "system",
            "You are an intelligent and helpful customer support agent. "
            "Your goal is to assist customers by accurately answering their questions, "
            "providing information, and resolving issues using the tools available to you. "
            "Always try to be polite and provide clear, concise answers. "
            "If a piece of information is missing (e.g., customer ID for an order), ask the user for it."
            "After using a tool, if you get an unexpected or insufficient result, reflect on it and try a different approach or ask clarifying questions."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# Create the Langchain ReAct agent
# ReAct (Reasoning and Acting) is a pattern for agents to interleave reasoning (planning) and acting (tool use).
agent = create_react_agent(llm, tools, prompt_template)

# Create the AgentExecutor
# This is the runtime for the agent, executing the agent's thoughts and actions.
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

### 4. Feedback and Self-Correction Module (Simplified) ###

# In a full Langgraph implementation, this would be a dedicated node or state transition.
# For this simplified agent, we'll embed a basic reflection logic in the Streamlit app's chat loop.

def reflect_and_correct(query: str, last_response: str, chat_history: List[Any]) -> str:
    """A simplified reflection mechanism to simulate self-correction based on a basic heuristic."""
    logger.info("Agent reflecting on last response...")
    # Simple heuristic: If the last response was an error or didn't directly answer, try again or rephrase.
    if "error" in last_response.lower() or "not found" in last_response.lower():
        reflection_prompt = (
            f"The previous attempt to answer the user's query '{query}' resulted in an error or 'not found'. "
            f"The last response was: '{last_response}'. "
            f"Given the chat history: {chat_history}, "
            "how can I rephrase the query or use a different tool to better assist the customer? "
            "Or should I ask for more information? Provide a corrected or alternative response/action."
        )
        logger.warning(f"Reflection triggered. Attempting to rephrase or ask for clarification.\nPrompt: {reflection_prompt}")
        # Use the LLM to generate a corrective action or a better response
        reflection_output = llm.invoke(HumanMessage(content=reflection_prompt)).content
        logger.info(f"Reflection output: {reflection_output}")
        return reflection_output
    return last_response # No correction needed based on this simple heuristic

### 5. User Interface (Streamlit) ###

st.set_page_config(page_title="Intelligent Customer Support Agent", layout="centered")
st.title("🛒 Intelligent Customer Support Agent")
st.markdown("Hello! I am an AI-powered customer support agent. How can I assist you today?")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.chat_history_lc = [] # For Langchain's chat_history

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask me a question..."):
    # Add user message to chat history and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.chat_history_lc.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Invoke the agent executor with current chat history and input
                response = agent_executor.invoke(
                    {"input": prompt, "chat_history": st.session_state.chat_history_lc}
                )
                agent_output = response["output"]

                # Apply simplified self-correction
                corrected_output = reflect_and_correct(prompt, agent_output, st.session_state.chat_history_lc)

                st.markdown(corrected_output)
                st.session_state.messages.append({"role": "assistant", "content": corrected_output})
                st.session_state.chat_history_lc.append(AIMessage(content=corrected_output))

            except Exception as e:
                logger.error(f"Agent execution failed: {e}")
                st.error("I apologize, but I encountered an error while processing your request. Please try again or rephrase.")
                st.session_state.messages.append({"role": "assistant", "content": "I apologize, but I encountered an error while processing your request."})
                st.session_state.chat_history_lc.append(AIMessage(content="Error in processing."))

# Optional: Clear chat history button
if st.button("Clear Chat History"):
    st.session_state.messages = []
    st.session_state.chat_history_lc = []
    st.rerun()

logger.info("Customer Support Agent Streamlit app started.")