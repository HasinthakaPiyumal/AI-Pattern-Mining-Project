import os
from langchain.agents import AgentExecutor, initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain.tools import Tool

# Set your OpenAI API key as an environment variable or replace 'os.environ["OPENAI_API_KEY"]' directly
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# 1. Simulated External Tools
def get_order_status(order_id: str) -> dict:
    """Simulates fetching order status from an Order Management System."""
    print(f"-> Calling Order Management System for Order ID: {order_id}")
    if order_id == "ORD123":
        return {"order_id": "ORD123", "status": "Shipped", "estimated_delivery": "2023-10-27"}
    elif order_id == "ORD456":
        return {"order_id": "ORD456", "status": "Processing", "estimated_delivery": "2023-11-05"}
    else:
        return {"order_id": order_id, "status": "Not Found", "message": "Please check the order ID."}

def get_product_details(product_id: str) -> dict:
    """Simulates fetching product details from a Product Catalog System."""
    print(f"-> Calling Product Catalog System for Product ID: {product_id}")
    if product_id == "PROD001":
        return {"product_id": "PROD001", "name": "Wireless Headphones", "price": 99.99, "in_stock": True, "description": "Noise-cancelling, Bluetooth 5.0, 20-hour battery life."}
    elif product_id == "PROD002":
        return {"product_id": "PROD002", "name": "Smartwatch", "price": 249.99, "in_stock": False, "description": "Heart rate monitor, GPS, Water-resistant."}
    else:
        return {"product_id": product_id, "name": "Unknown Product", "message": "Product not found in catalog."}

def get_billing_info(customer_id: str) -> dict:
    """Simulates fetching billing information from a Billing System."""
    print(f"-> Calling Billing System for Customer ID: {customer_id}")
    if customer_id == "CUST789":
        return {"customer_id": "CUST789", "balance_due": 50.00, "last_payment_date": "2023-09-30", "next_bill_date": "2023-11-01"}
    else:
        return {"customer_id": customer_id, "balance_due": 0.00, "message": "No outstanding balance or customer not found."}

def perform_general_search(query: str) -> str:
    """Simulates performing a general web search for factual questions."""
    print(f"-> Performing General Search for query: '{query}'")
    if "capital of france" in query.lower():
        return "The capital of France is Paris."
    elif "current weather in london" in query.lower():
        return "The current weather in London is partly cloudy with a temperature of 15°C."
    else:
        return f"Search results for '{query}': Information relevant to your query found online..."

# 2. Tool Definitions
tools = [
    Tool(
        name="Order Status Tool",
        func=get_order_status,
        description="Useful for getting the current status of a customer's order. Input should be an order ID (e.g., ORD123)."
    ),
    Tool(
        name="Product Details Tool",
        func=get_product_details,
        description="Useful for retrieving detailed information about a product from the catalog. Input should be a product ID (e.g., PROD001)."
    ),
    Tool(
        name="Billing Information Tool",
        func=get_billing_info,
        description="Useful for checking a customer's billing balance or history. Input should be a customer ID (e.g., CUST789)."
    ),
    Tool(
        name="General Search Tool",
        func=perform_general_search,
        description="Useful for answering general knowledge questions or fetching real-world information not covered by other tools. Input should be a natural language query."
    ),
]

# 3. LLM Router
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

# 4. Agent Executor
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS, # or AgentType.ZERO_SHOT_REACT_DESCRIPTION
    verbose=True,
    handle_parsing_errors=True,
)

# 5. Input/Output Interface
def run_customer_support_agent():
    print("\n--- Smart Customer Support Agent ---\n")
    print("Ask me anything about orders, products, billing, or general questions.")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("You: ")
        if query.lower() == 'exit':
            break
        try:
            response = agent.run(query)
            print(f"Agent: {response}\n")
        except Exception as e:
            print(f"Agent Error: {e}\n")

if __name__ == "__main__":
    run_customer_support_agent()