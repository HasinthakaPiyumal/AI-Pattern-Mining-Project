import os
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory

# Load environment variables from .env file
load_dotenv()

# --- Tool-Use Module ---

def get_product_details(product_id: str) -> dict:
    """Simulates fetching details for a product from an e-commerce database."""
    print(f"[TOOL CALL] get_product_details called for product_id: {product_id}")
    products = {
        "P1001": {"name": "Wireless Headphones", "price": 99.99, "description": "High-quality wireless headphones with noise cancellation.", "category": "Electronics"},
        "P1002": {"name": "Ergonomic Office Chair", "price": 249.99, "description": "Comfortable office chair for long working hours.", "category": "Furniture"},
        "P1003": {"name": "Smartwatch Series 5", "price": 199.99, "description": "Fitness tracker and notification hub on your wrist.", "category": "Electronics"},
    }
    return products.get(product_id, {"error": "Product not found."})

def process_return(order_id: str, product_id: str, reason: str) -> str:
    """Simulates initiating a return process for an order and product."""
    print(f"[TOOL CALL] process_return called for order_id: {order_id}, product_id: {product_id}, reason: {reason}")
    if order_id.startswith("ORD") and product_id.startswith("P"):
        return f"Return for Order {order_id}, Product {product_id} with reason '{reason}' has been successfully initiated. Awaiting approval."
    return "Failed to initiate return. Please check order and product details."

def get_order_status(order_id: str) -> str:
    """Simulates checking the status of an order."""
    print(f"[TOOL CALL] get_order_status called for order_id: {order_id}")
    orders = {
        "ORD789": "Shipped on 2023-10-26, estimated delivery 2023-10-30.",
        "ORD123": "Processing. Will be shipped within 2 business days.",
        "ORD456": "Delivered on 2023-10-20."
    }
    return orders.get(order_id, "Order not found or invalid order ID.")

def recommend_products(category: str) -> list:
    """Simulates fetching product recommendations based on a category."""
    print(f"[TOOL CALL] recommend_products called for category: {category}")
    recommendations = {
        "Electronics": ["Wireless Mouse", "Portable Speaker", "USB-C Hub"],
        "Furniture": ["Standing Desk", "Bookshelf", "Table Lamp"],
        "Apparel": ["Casual T-shirt", "Denim Jeans", "Winter Jacket"],
    }
    return recommendations.get(category, ["No specific recommendations for this category yet."])

# Wrap the functions as Langchain tools
tools = [
    Tool(
        name="get_product_details",
        func=get_product_details,
        description="Useful for getting detailed information about a product by its ID."
    ),
    Tool(
        name="process_return",
        func=process_return,
        description="Useful for initiating a return for a product within an order. Requires order ID, product ID, and reason for return."
    ),
    Tool(
        name="get_order_status",
        func=get_order_status,
        description="Useful for checking the current shipping or processing status of an order by its ID."
    ),
    Tool(
        name="recommend_products",
        func=recommend_products,
        description="Useful for recommending products based on a given category."
    ),
]

# --- Memory Module ---
memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

# --- LLM Agent Core ---

# Initialize the LLM
# Ensure you have OPENAI_API_KEY set in your environment variables or .env file
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-1106") # You can use other models like gpt-4 if available

# Define the system prompt for the agent
system_message = SystemMessage(
    content=(
        "You are an adaptive and helpful e-commerce customer support agent. "
        "Your goal is to assist customers with their inquiries, process returns, "
        "provide product recommendations, and check order statuses. "
        "You have access to several tools to perform these tasks. "
        "Always be polite, clear, and concise. "
        "If you need more information from the user, ask for it clearly. "
        "If a task is complex or requires human intervention (e.g., complex refund scenarios, technical issues), "
        "state that you are escalating the issue to a human agent and provide a brief summary of the problem."
    )
)

# Create the prompt template for the agent
prompt = ChatPromptTemplate.from_messages(
    [
        system_message,
        MessagesPlaceholder(variable_name="chat_history"),  # For memory
        HumanMessage(content="{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"), # For agent's thoughts and tool outputs
    ]
)

# Create the agent executor
agent_executor = create_openai_functions_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# Integrate memory into the agent executor
agent = AgentExecutor(
    agent=agent_executor,
    tools=tools,
    verbose=True, # Set to True to see the agent's thought process
    memory=memory,
    handle_parsing_errors=True # To gracefully handle LLM output parsing errors
)

# --- Main Application ---

def run_customer_support_agent():
    print("\n--- E-commerce Customer Support Agent ---")
    print("Hello! How can I assist you today? (Type 'exit' to quit)\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Agent: Goodbye! Have a great day.")
            break

        try:
            response = agent.invoke({"input": user_input})
            print(f"Agent: {response['output']}")
        except Exception as e:
            print(f"Agent: An error occurred: {e}. Please try again or rephrase your request.")

if __name__ == "__main__":
    run_customer_support_agent()