import os
from typing import List, Dict, Any

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langchain.memory import ConversationBufferWindowMemory
from pydantic import BaseModel, Field

# Set your OpenAI API key (replace with environment variable in production)
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# --- 1. Define Simulated External Systems/Tools ---

# In a real application, these would interact with actual databases, APIs, etc.

# Product Database
product_db = {
    "laptop": {"price": 1200, "stock": 10, "description": "Powerful laptop for work and gaming."},
    "keyboard": {"price": 75, "stock": 50, "description": "Mechanical keyboard with RGB lighting."},
    "mouse": {"price": 30, "stock": 100, "description": "Ergonomic wireless mouse."},
}

# Order Management System
order_system = {
    "ORD12345": {"status": "Shipped", "items": [{"name": "laptop", "qty": 1}], "delivery_date": "2023-11-15"},
    "ORD67890": {"status": "Processing", "items": [{"name": "keyboard", "qty": 1}, {"name": "mouse", "qty": 1}], "delivery_date": "N/A"},
}

# Knowledge Base (FAQs)
knowledge_base = {
    "shipping": "Standard shipping takes 3-5 business days. Express shipping takes 1-2 business days.",
    "returns": "You can return most items within 30 days of purchase. Items must be in their original condition.",
    "warranty": "All electronics come with a 1-year manufacturer\'s warranty.",
    "payment methods": "We accept Visa, Mastercard, American Express, and PayPal."
}


# --- 2. Define Pydantic Models for Tool Inputs ---

class ProductDetailsInput(BaseModel):
    product_name: str = Field(description="The name of the product to get details for.")

class OrderStatusInput(BaseModel):
    order_id: str = Field(description="The unique identifier for the order.")

class InitiateReturnInput(BaseModel):
    order_id: str = Field(description="The unique identifier for the order from which to return an item.")
    product_name: str = Field(description="The name of the product to be returned.")
    reason: str = Field(description="The reason for the return (e.g., defective, wrong item, changed mind).")

class SearchKnowledgeBaseInput(BaseModel):
    query: str = Field(description="The search query for the knowledge base.")


# --- 3. Define Tools for the Agent ---

@tool(
    args_schema=ProductDetailsInput,
    description="Gets detailed information about a product, including price, stock, and description."
)
def get_product_details(product_name: str) -> Dict[str, Any]:
    """Gets detailed information about a product, including price, stock, and description."""
    product_name = product_name.lower()
    if product_name in product_db:
        return product_db[product_name]
    return {"error": "Product not found.", "product_name": product_name}

@tool(
    args_schema=OrderStatusInput,
    description="Retrieves the current status and delivery information for a given order ID."
)
def get_order_status(order_id: str) -> Dict[str, Any]:
    """Retrieves the current status and delivery information for a given order ID."""
    if order_id in order_system:
        return order_system[order_id]
    return {"error": "Order not found.", "order_id": order_id}

@tool(
    args_schema=InitiateReturnInput,
    description="Initiates a return process for a specific product within an order. Requires order ID, product name, and reason."
)
def initiate_return(order_id: str, product_name: str, reason: str) -> Dict[str, str]:
    """Initiates a return process for a specific product within an order."""
    if order_id not in order_system:
        return {"status": "failed", "message": f"Order {order_id} not found."}
    
    # Simulate checking if the product is in the order
    order_items = [item["name"] for item in order_system[order_id]["items"]]
    if product_name.lower() not in order_items:
        return {"status": "failed", "message": f"Product {product_name} not found in order {order_id}."}

    # In a real system, this would trigger a return workflow
    return {"status": "success", "message": f"Return initiated for {product_name} from order {order_id}. Reason: {reason}."}

@tool(
    args_schema=SearchKnowledgeBaseInput,
    description="Searches the customer support knowledge base for answers to common questions."
)
def search_knowledge_base(query: str) -> Dict[str, str]:
    """Searches the customer support knowledge base for answers to common questions."""
    query_lower = query.lower()
    for keyword, answer in knowledge_base.items():
        if keyword in query_lower:
            return {"query": query, "answer": answer}
    return {"query": query, "answer": "No relevant information found in the knowledge base for your query. Please try rephrasing or contact a human agent for further assistance.", "source": "knowledge_base"}

# List of all tools available to the agent
tools = [
    get_product_details,
    get_order_status,
    initiate_return,
    search_knowledge_base
]


# --- 4. Initialize LLM and Memory ---

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Memory for conversational context
memory = ConversationBufferWindowMemory(
    memory_key="chat_history", 
    return_messages=True, 
    input_key="input",
    k=5 # Keep track of the last 5 turns of conversation
)


# --- 5. Create Agent Prompt ---

# The system message guides the agent's persona and responsibilities
system_message = (
    "You are a helpful and adaptive AI customer support agent for an e-commerce store. "
    "Your goal is to assist customers with product inquiries, order statuses, returns, and general questions. "
    "You have access to various tools to retrieve information and perform actions. "
    "Always try to use the available tools to answer questions before resorting to general knowledge. "
    "If a user asks for something outside your capabilities or that requires human intervention, politely state so."
)

# The prompt template includes the system message, chat history, and placeholders for tools and input
prompt = ChatPromptTemplate.from_messages([
    ("system", system_message),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])


# --- 6. Create Agent and Executor ---

agent = create_openai_tools_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True # Set to True to see the agent's thought process
)


# --- 7. Main Application Loop ---

def run_customer_support_agent():
    print("Welcome to our E-commerce Customer Support! How can I assist you today?")
    print("Type 'exit' to end the conversation.")

    while True:
        user_input = input("\nCustomer: ")
        if user_input.lower() == 'exit':
            print("Thank you for contacting support. Goodbye!")
            break

        try:
            # The agent_executor handles the full cycle: planning, tool use, response generation
            response = agent_executor.invoke({"input": user_input})
            print(f"Agent: {response['output']}")
        except Exception as e:
            print(f"Agent: An error occurred: {e}. Please try again or rephrase your request.")
            # Optionally, log the error for debugging
            # import traceback
            # traceback.print_exc()

if __name__ == "__main__":
    run_customer_support_agent()