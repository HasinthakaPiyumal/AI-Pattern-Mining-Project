"""
This module implements a modular agentic augmentation framework for an e-commerce customer support agent.
It integrates a core LLM with specialized modules for memory, planning, and external tool interaction.
"""

import os
from typing import List, Dict, Any, Optional
import requests
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document

load_dotenv() # Load environment variables from .env file

# --- 1. Configuration & Setup ---

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ECOM_API_BASE_URL = os.getenv("ECOM_API_BASE_URL", "http://localhost:8000/api")
VECTOR_DB_DIR = "./chroma_db"

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)

# Initialize Embeddings for Vector Store
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# --- 2. Memory Module ---

# Short-term conversational memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    return_messages=True,
    k=5 # Keep last 5 turns of conversation
)

# Long-term customer data memory (ChromaDB for semantic search)
def get_customer_vector_db():
    if not os.path.exists(VECTOR_DB_DIR):
        print(f"Initializing ChromaDB at {VECTOR_DB_DIR}. No existing data found.")
        # In a real application, you would load customer data from a database
        # For demonstration, we'll create some dummy documents.
        dummy_customer_data = [
            Document(page_content="Customer ID: 101, Name: Alice Smith, Email: alice@example.com, Prefers electronics, Last order: Laptop, Order ID: ORD-2023-001"),
            Document(page_content="Customer ID: 102, Name: Bob Johnson, Email: bob@example.com, Prefers books and apparel, Last order: T-shirt, Order ID: ORD-2023-002"),
            Document(page_content="Customer ID: 103, Name: Charlie Brown, Email: charlie@example.com, Interested in home goods, Last order: Coffee Maker, Order ID: ORD-2023-003")
        ]
        db = Chroma.from_documents(documents=dummy_customer_data, embedding=embeddings, persist_directory=VECTOR_DB_DIR)
        db.persist()
        return db
    else:
        print(f"Loading ChromaDB from {VECTOR_DB_DIR}")
        return Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings)

customer_vector_db = get_customer_vector_db()

# --- 3. External Tool Interfaces (Simulated E-commerce API) ---

# In a real scenario, these would make actual API calls.
# For this example, we'll simulate API responses.

def make_api_request(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Simulates an API request to the e-commerce backend."""
    print(f"Simulating API call to {ECOM_API_BASE_URL}/{endpoint} with params: {params}")
    if endpoint == "orders/lookup":
        order_id = params.get("order_id")
        if order_id == "ORD-2023-001":
            return {"status": "success", "order": {"id": order_id, "customer_id": "101", "items": [{"name": "Laptop", "qty": 1, "price": 1200}], "status": "shipped", "shipping_address": "123 Main St"}}
        elif order_id == "ORD-2023-002":
            return {"status": "success", "order": {"id": order_id, "customer_id": "102", "items": [{"name": "T-shirt", "qty": 2, "price": 25}], "status": "processing", "shipping_address": "456 Oak Ave"}}
        elif order_id == "ORD-2023-003":
            return {"status": "success", "order": {"id": order_id, "customer_id": "103", "items": [{"name": "Coffee Maker", "qty": 1, "price": 80}], "status": "delivered", "shipping_address": "789 Pine Ln"}}
        else:
            return {"status": "error", "message": f"Order {order_id} not found."}
    
    elif endpoint == "inventory/check":
        product_id = params.get("product_id")
        if product_id == "laptop123":
            return {"status": "success", "product_id": product_id, "stock": 50, "available": True}
        elif product_id == "tshirtabc":
            return {"status": "success", "product_id": product_id, "stock": 200, "available": True}
        elif product_id == "coffeemakerxyz":
            return {"status": "success", "product_id": product_id, "stock": 10, "available": True}
        elif product_id == "outofstockitem":
            return {"status": "success", "product_id": product_id, "stock": 0, "available": False}
        else:
            return {"status": "error", "message": f"Product {product_id} not found."}
            
    elif endpoint == "products/recommend":
        customer_id = params.get("customer_id")
        query = params.get("query")
        if customer_id == "101": # Alice prefers electronics
            return {"status": "success", "recommendations": [{"name": "Wireless Mouse", "id": "mouse456"}, {"name": "External Monitor", "id": "monitor789"}]}
        elif customer_id == "102": # Bob prefers books and apparel
            return {"status": "success", "recommendations": [{"name": "Fantasy Novel", "id": "book101"}, {"name": "Hoodie", "id": "hoodie202"}]}
        else:
            return {"status": "success", "recommendations": [{"name": "Popular Item A", "id": "pa1"}, {"name": "Popular Item B", "id": "pb2"}]}
            
    elif endpoint == "orders/refund":
        order_id = params.get("order_id")
        reason = params.get("reason")
        if order_id in ["ORD-2023-001", "ORD-2023-002", "ORD-2023-003"]:
            return {"status": "success", "message": f"Refund initiated for order {order_id} due to: {reason}. Processing time 3-5 business days."}
        else:
            return {"status": "error", "message": f"Cannot process refund for unknown order {order_id}."}

    elif endpoint == "shipping/update":
        order_id = params.get("order_id")
        new_address = params.get("new_address")
        if order_id in ["ORD-2023-001", "ORD-2023-002", "ORD-2023-003"]:
            return {"status": "success", "message": f"Shipping address for order {order_id} updated to {new_address}."}
        else:
            return {"status": "error", "message": f"Cannot update shipping address for unknown order {order_id}."}

    return {"status": "error", "message": "Unknown API endpoint."}


@tool
def order_lookup_tool(order_id: str) -> str:
    """Looks up details for a given order ID. Returns order information including items, status, and shipping address."""
    response = make_api_request("orders/lookup", {"order_id": order_id})
    if response["status"] == "success":
        order = response["order"]
        items_str = ", ".join([f"{item['qty']}x {item['name']}" for item in order["items"]])
        return f"Order {order['id']} (Customer ID: {order['customer_id']}): Status - {order['status']}, Items - {items_str}, Shipping Address - {order['shipping_address']}."
    else:
        return response["message"]

@tool
def inventory_check_tool(product_id: str) -> str:
    """Checks the current stock level and availability for a specific product ID."""
    response = make_api_request("inventory/check", {"product_id": product_id})
    if response["status"] == "success":
        return f"Product {response['product_id']}: Stock - {response['stock']}, Available - {response['available']}."
    else:
        return response["message"]

@tool
def product_recommendation_tool(customer_id: str, query: str) -> str:
    """Recommends products to a customer based on their ID and a specific query or interest. Useful for upselling or cross-selling."""
    response = make_api_request("products/recommend", {"customer_id": customer_id, "query": query})
    if response["status"] == "success" and response["recommendations"]:
        recs_str = ", ".join([f"{rec['name']} (ID: {rec['id']})" for rec in response["recommendations"]])
        return f"Here are some recommendations for customer {customer_id} based on '{query}': {recs_str}."
    elif response["status"] == "success" and not response["recommendations"]:
        return f"No specific recommendations found for customer {customer_id} based on '{query}'."
    else:
        return response["message"]

@tool
def process_refund_tool(order_id: str, reason: str) -> str:
    """Initiates a refund for a given order ID with a specified reason. Requires a valid order ID and a clear reason for the refund."""
    response = make_api_request("orders/refund", {"order_id": order_id, "reason": reason})
    return response["message"]

@tool
def update_shipping_address_tool(order_id: str, new_address: str) -> str:
    """Updates the shipping address for a given order ID. Requires a valid order ID and the complete new shipping address."""
    response = make_api_request("shipping/update", {"order_id": order_id, "new_address": new_address})
    return response["message"]


# List of all tools available to the agent
all_tools = [
    order_lookup_tool,
    inventory_check_tool,
    product_recommendation_tool,
    process_refund_tool,
    update_shipping_address_tool,
]

# --- 4. Planning Module (Agent Definition) ---

# The prompt template for the agent
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", (
            "You are an AI assistant for an e-commerce customer support. Your goal is to help customers with their queries "
            "efficiently and accurately. You have access to various tools to lookup orders, check inventory, "
            "recommend products, process refunds, and update shipping addresses. \n\n"
            "Use the following process:\n"
            "1. First, try to understand the user's intent.\n"
            "2. If the user is asking about a specific order, try to extract the order ID. If not provided, ask for it.\n"
            "3. If the user is asking for product recommendations, try to extract their customer ID or previous interests.\n"
            "4. Use the appropriate tools to gather information or perform actions.\n"
            "5. Provide clear, concise, and helpful responses. Always confirm actions taken.\n"
            "6. If you need customer-specific information not directly available via tools, try searching the customer long-term memory.\n"
            "7. Be polite and professional."
        )),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# Create the agent
agent = create_openai_tools_agent(llm, all_tools, prompt)

# Create the Agent Executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=all_tools,
    verbose=True, # Set to True to see the agent's thought process
    memory=conversational_memory,
    handle_parsing_errors=True # Handle cases where LLM output is not in expected format
)

# --- 5. Context Management & Interaction Loop ---

def get_customer_context_from_memory(query: str) -> str:
    """Retrieves relevant customer information from long-term vector memory."""
    print(f"Searching customer long-term memory for: '{query}'")
    docs = customer_vector_db.similarity_search(query, k=1) # Retrieve top 1 relevant document
    if docs:
        return docs[0].page_content
    return "No specific customer information found in long-term memory."


def run_agent(user_query: str) -> str:
    """
    Runs the customer support agent with the given user query, integrating
    conversational memory and long-term customer context.
    """
    print(f"\n--- User: {user_query} ---")
    
    # Optionally enrich the query with long-term customer data before passing to agent
    # This part demonstrates Context Management by injecting retrieved memory
    customer_specific_context = get_customer_context_from_memory(user_query)
    
    # The agent's prompt already includes chat_history from conversational_memory
    # We can inject additional context via the input if needed, or modify the prompt
    # to explicitly include a 'long_term_memory_context' variable.
    # For simplicity, we'll let the LLM implicitly use it if it's part of the user_query
    # or if the agent itself is designed to query it based on initial inputs.
    
    # A more robust approach would be to make `get_customer_context_from_memory`
    # another tool, or to pre-process the user query with this context before passing.
    
    # For this example, let's inject a simplified version into the initial input
    # if relevant context is found, to show how context can influence the agent.
    enriched_query = user_query
    if "No specific customer information" not in customer_specific_context:
        enriched_query = f"User Query: {user_query}\nRelevant Customer Data: {customer_specific_context}"
        print(f"Enriched Query: {enriched_query}")

    try:
        # Invoke the agent executor
        response = agent_executor.invoke({"input": enriched_query})
        agent_response = response["output"]
        print(f"\n--- Agent: {agent_response} ---")
        return agent_response
    except Exception as e:
        print(f"An error occurred: {e}")
        return "I apologize, but I encountered an error while processing your request. Please try again later."


if __name__ == "__main__":
    print("E-commerce Customer Support Agent initialized. Type 'exit' to quit.")

    while True:
        user_input = input("\nYou (Customer): ")
        if user_input.lower() == 'exit':
            print("Thank you for using the E-commerce Customer Support Agent. Goodbye!")
            break
        
        # Process the user input with the agent
        agent_response = run_agent(user_input)
