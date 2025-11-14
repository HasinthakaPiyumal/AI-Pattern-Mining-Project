
import os
from typing import List, Dict, Any
from dotenv import load_dotenv

# --- Simulate External Systems (Knowledge Base, CRM, External APIs) ---

# Mock Internal Knowledge Base (FAQs, Product Manuals)
# In a real-world scenario, this would be a vector database (e.g., Chroma, Pinecone) with embedded documents.
INTERNAL_KNOWLEDGE_BASE = [
    {
        "id": "kb_001",
        "title": "Troubleshooting Internet Connectivity",
        "content": "If you are experiencing internet connectivity issues, first check if your router is powered on. Try restarting your router and modem. If the issue persists, ensure all cables are securely connected. For further assistance, contact technical support."
    },
    {
        "id": "kb_002",
        "title": "Product Return Policy",
        "content": "Our return policy allows returns within 30 days of purchase for a full refund, provided the item is in its original condition with all packaging. For defective products, we offer an exchange or full refund within 90 days. Please visit our returns portal on the website to initiate a return."
    },
    {
        "id": "kb_003",
        "title": "How to Reset Your Password",
        "content": "To reset your password, navigate to the login page and click on the 'Forgot Password' link. Enter your registered email address, and we will send you a password reset link. Follow the instructions in the email to create a new password."
    },
    {
        "id": "kb_004",
        "title": "Shipping Times and Costs",
        "content": "Standard shipping usually takes 5-7 business days. Expedited shipping is available for an additional cost and typically delivers within 2-3 business days. Shipping costs are calculated at checkout based on your location and the weight of your order. International shipping times vary."
    }
]

# Mock CRM System (Customer History, Order Details)
CRM_DATA = {
    "user_123": {
        "name": "Alice Smith",
        "email": "alice.smith@example.com",
        "orders": [
            {"order_id": "ORD789", "product": "Laptop X", "status": "Shipped", "shipping_date": "2023-10-20", "delivery_date": "2023-10-25"},
            {"order_id": "ORD456", "product": "Mouse Y", "status": "Delivered", "delivery_date": "2023-09-10"}
        ],
        "support_tickets": [
            {"ticket_id": "TKT001", "subject": "Internet connectivity issue", "status": "Closed"}
        ]
    },
    "user_456": {
        "name": "Bob Johnson",
        "email": "bob.j@example.com",
        "orders": [
            {"order_id": "ORD101", "product": "Keyboard Z", "status": "Processing", "expected_ship_date": "2023-11-15"}
        ],
        "support_tickets": []
    }
}

# Mock External Real-time APIs (e.g., Shipping Updates, Stock Availability)
# In a real application, these would make actual API calls.
EXTERNAL_APIS = {
    "shipping_tracker": {
        "ORD789": {"status": "In Transit", "current_location": "Warehouse A", "estimated_delivery": "2023-10-28"},
        "ORD101": {"status": "Awaiting Pickup", "current_location": "Processing Facility", "estimated_delivery": "2023-11-18"}
    },
    "stock_checker": {
        "Laptop X": {"in_stock": True, "quantity": 150},
        "Mouse Y": {"in_stock": False, "quantity": 0},
        "Keyboard Z": {"in_stock": True, "quantity": 25}
    }
}

def get_knowledge_base_articles(query: str) -> List[Dict[str, str]]:
    """Simulates retrieving relevant articles from the internal knowledge base based on a query."""
    # A very basic keyword-based retrieval. In a real RAG system, this would involve embeddings and vector search.
    relevant_articles = []
    query_lower = query.lower()
    for article in INTERNAL_KNOWLEDGE_BASE:
        if query_lower in article["title"].lower() or query_lower in article["content"].lower():
            relevant_articles.append(article)
    return relevant_articles

def get_customer_orders(user_id: str) -> List[Dict[str, Any]]:
    """Retrieves a customer's order history from the CRM system."""
    customer_data = CRM_DATA.get(user_id)
    return customer_data.get("orders", []) if customer_data else []

def get_order_details(order_id: str) -> Dict[str, Any]:
    """Retrieves specific order details from the CRM or shipping API."""
    for user_data in CRM_DATA.values():
        for order in user_data.get("orders", []):
            if order["order_id"] == order_id:
                # Augment with real-time shipping info if available
                shipping_info = EXTERNAL_APIS["shipping_tracker"].get(order_id, {})
                return {**order, **shipping_info}
    return {}

def check_product_stock(product_name: str) -> Dict[str, Any]:
    """Checks the stock availability of a product via an external API."""
    return EXTERNAL_APIS["stock_checker"].get(product_name, {"in_stock": False, "quantity": 0})

# --- LLM Integration (using Langchain) ---

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

load_dotenv() # Load environment variables from .env file

# Define the LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Define Tools for the Agent
@tool
def search_knowledge_base(query: str) -> str:
    """Searches the internal knowledge base for relevant articles based on the user's query. Useful for FAQs, troubleshooting, and product information."""
    articles = get_knowledge_base_articles(query)
    if not articles:
        return "No relevant articles found in the knowledge base."
    return "\n---\n".join([f"Title: {a['title']}\nContent: {a['content']}" for a in articles])

@tool
def retrieve_customer_orders(user_id: str) -> str:
    """Retrieves the order history for a given customer ID from the CRM system. Requires a specific user_id."""
    orders = get_customer_orders(user_id)
    if not orders:
        return f"No orders found for customer ID {user_id}."
    return f"Customer {user_id} has the following orders: {str(orders)}"

@tool
def get_current_order_status(order_id: str) -> str:
    """Gets the real-time status and details for a specific order ID, including shipping information if available. Useful for tracking shipments."""
    order_details = get_order_details(order_id)
    if not order_details:
        return f"Order ID {order_id} not found or no details available."
    return str(order_details)

@tool
def check_stock_availability(product_name: str) -> str:
    """Checks the current stock availability and quantity for a specified product. Useful for answering questions about product availability."""
    stock_info = check_product_stock(product_name)
    if not stock_info["in_stock"]:
        return f"Product '{product_name}' is currently out of stock."
    return f"Product '{product_name}' is in stock. Quantity: {stock_info['quantity']}."

# Create a list of all tools
tools = [search_knowledge_base, retrieve_customer_orders, get_current_order_status, check_stock_availability]

# Define the prompt for the LLM agent
# The prompt guides the LLM on how to use the tools and respond to the user.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful customer support assistant. You have access to various tools to answer questions and assist customers. Always try to use the tools to get the most accurate and up-to-date information. If a user_id is required for a tool, ask the user for it if not provided."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# Create the Langchain Agent
agent = create_tool_calling_agent(llm, tools, prompt)

# Create the Agent Executor to run the agent
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- Chatbot Interaction Loop ---

def run_chatbot():
    print("Welcome to the Knowledge-Augmented Customer Support Chatbot!")
    print("Type 'exit' to end the conversation.")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            print("Thank you for contacting support. Goodbye!")
            break

        try:
            # Invoke the agent with the user's input
            response = agent_executor.invoke({"input": user_input})
            print(f"\nChatbot: {response['output']}")
        except Exception as e:
            print(f"\nChatbot: An error occurred: {e}")
            print("Chatbot: Please try again or rephrase your question.")

if __name__ == "__main__":
    # Ensure OPENAI_API_KEY is set in your .env file or environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set. The chatbot might not function correctly without it.")
        print("Please create a .env file with OPENAI_API_KEY='your_openai_api_key_here'")

    run_chatbot()
