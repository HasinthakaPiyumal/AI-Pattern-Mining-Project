import random
from typing import TypedDict, Annotated, List, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_community.chat_models import ChatOllama
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
import operator
import json

# --- 1. Mock E-commerce Platform Data and Tools ---

PRODUCTS_DB = [
    {"id": "p101", "name": "Gaming Laptop Pro X", "category": "laptop", "brand": "BrandX", "price": 1800.00, "screen_size": 17, "stock": 5, "description": "Powerful gaming laptop with high-end graphics."},
    {"id": "p102", "name": "Budget Gaming Laptop", "category": "laptop", "brand": "BrandY", "price": 1200.00, "screen_size": 15, "stock": 10, "description": "Affordable gaming laptop for casual gamers."},
    {"id": "p103", "name": "Ultraportable Laptop Z", "category": "laptop", "brand": "BrandZ", "price": 1100.00, "screen_size": 13, "stock": 8, "description": "Lightweight and sleek laptop for professionals."},
    {"id": "p104", "name": "High-End Workstation Laptop", "category": "laptop", "brand": "BrandX", "price": 2500.00, "screen_size": 16, "stock": 3, "description": "Designed for demanding tasks and creative professionals."},
    {"id": "p105", "name": "Gaming Desktop Beast", "category": "desktop", "brand": "BrandA", "price": 3000.00, "stock": 2, "description": "Extreme performance gaming desktop."},
    {"id": "p106", "name": "Ergonomic Office Chair", "category": "furniture", "brand": "BrandC", "price": 350.00, "stock": 20, "description": "Comfortable chair for long working hours."},
    {"id": "p107", "name": "Wireless Gaming Mouse", "category": "accessory", "brand": "BrandY", "price": 75.00, "stock": 50, "description": "Precision gaming mouse with customizable buttons."},
    {"id": "p108", "name": "Affordable 15-inch Laptop", "category": "laptop", "brand": "BrandZ", "price": 950.00, "screen_size": 15, "stock": 12, "description": "Good value laptop for everyday tasks."},
    {"id": "p109", "name": "Premium 16-inch Laptop", "category": "laptop", "brand": "BrandX", "price": 1600.00, "screen_size": 16, "stock": 7, "description": "High-performance laptop with a stunning display."},
]

USER_CARTS = {} # key: user_id, value: {product_id: {"product": product_dict, "quantity": int}}
USER_ORDER_HISTORIES = {} # key: user_id, value: list of past orders

# Mock functions simulating e-commerce API calls
def _search_products_impl(query: str = None, min_price: float = 0, max_price: float = float('inf'), brand: str = None, screen_size: int = None, category: str = None) -> List[dict]:
    results = []
    query_lower = query.lower() if query else ""
    for product in PRODUCTS_DB:
        match = True
        if query and not (query_lower in product["name"].lower() or query_lower in product["category"].lower() or query_lower in product["description"].lower()):
            match = False
        if product["price"] < min_price or product["price"] > max_price:
            match = False
        if brand and product["brand"].lower() != brand.lower():
            match = False
        if screen_size and "screen_size" in product and product["screen_size"] != screen_size:
            match = False
        if category and product["category"].lower() != category.lower():
            match = False
        if match:
            results.append(product)
    return results

def _get_product_details_impl(product_id: str) -> dict:
    for product in PRODUCTS_DB:
        if product["id"] == product_id:
            return product
    return {"error": "Product not found"}

def _add_to_cart_impl(product_id: str, quantity: int = 1, user_id: str = "default_user") -> str:
    product = _get_product_details_impl(product_id)
    if "error" in product:
        return f"Error: {product['error']}"
    if product["stock"] < quantity:
        return f"Error: Only {product['stock']} units of {product['name']} are in stock."
    
    if user_id not in USER_CARTS:
        USER_CARTS[user_id] = {}

    if product_id in USER_CARTS[user_id]:
        USER_CARTS[user_id][product_id]["quantity"] += quantity
    else:
        USER_CARTS[user_id][product_id] = {"product": product, "quantity": quantity}
    
    current_cart_summary = ", ".join([f"{item['quantity']}x {item['product']['name']}" for item in USER_CARTS[user_id].values()])
    return f"Added {quantity} x {product['name']} to your cart. Your current cart: {current_cart_summary}"

def _check_stock_impl(product_id: str) -> dict:
    product = _get_product_details_impl(product_id)
    if "error" in product:
        return {"error": product["error"]}
    return {"product_id": product_id, "name": product["name"], "stock": product["stock"]}

def _get_user_order_history_impl(user_id: str = "default_user") -> List[dict]:
    return USER_ORDER_HISTORIES.get(user_id, [])

# Langchain Tool Definitions
@tool
def search_products_tool(query: str = None, min_price: float = 0, max_price: float = float('inf'), brand: str = None, screen_size: int = None, category: str = None) -> List[dict]:
    """Searches the product catalog based on various filters.
    Args:
        query (str): General search query (e.g., "gaming laptop").
        min_price (float): Minimum price for the product.
        max_price (float): Maximum price for the product.
        brand (str): Brand of the product.
        screen_size (int): Screen size in inches (for laptops/monitors).
        category (str): Category of the product (e.g., "laptop", "accessory").
    Returns:
        list: A list of dictionaries, where each dictionary represents a product.
    """
    return _search_products_impl(query, min_price, max_price, brand, screen_size, category)

@tool
def get_product_details_tool(product_id: str) -> dict:
    """Fetches detailed information about a specific product.
    Args:
        product_id (str): The unique identifier of the product.
    Returns:
        dict: A dictionary containing product details or an error message if not found.
    """
    return _get_product_details_impl(product_id)

@tool
def add_to_cart_tool(product_id: str, quantity: int = 1, user_id: str = "default_user") -> str:
    """Adds a specified quantity of a product to the user's cart.
    Args:
        product_id (str): The unique identifier of the product.
        quantity (int): The number of units to add to the cart. Defaults to 1.
        user_id (str): The unique identifier of the user. Defaults to "default_user".
    Returns:
        str: A confirmation message or an error if the product is out of stock/not found.
    """
    return _add_to_cart_impl(product_id, quantity, user_id)

@tool
def check_stock_tool(product_id: str) -> dict:
    """Checks the current stock level for a given product.
    Args:
        product_id (str): The unique identifier of the product.
    Returns:
        dict: A dictionary with product ID, name, and stock level, or an error if not found.
    """
    return _check_stock_impl(product_id)

@tool
def get_user_order_history_tool(user_id: str = "default_user") -> List[dict]:
    """Retrieves the order history for a given user.
    Args:
        user_id (str): The unique identifier of the user. Defaults to "default_user".
    Returns:
        list: A list of past orders.
    """
    return _get_user_order_history_impl(user_id)

ECOMMERCE_TOOLS = [search_products_tool, get_product_details_tool, add_to_cart_tool, check_stock_tool, get_user_order_history_tool]

# --- 2. LLM Agent Orchestrator (Iterative Task Solver) ---

class AgentState(TypedDict):
    chat_history: Annotated[List[BaseMessage], operator.add]
    intermediate_steps: Annotated[List[tuple[AgentAction, str]], operator.add]
    user_query: str
    product_filters: dict # To store current filters based on user feedback
    user_id: str # To manage user-specific carts/history

# Initialize LLM (using ChatOllama as an example - requires Ollama to be running)
# Replace with ChatOpenAI, Anthropic, or other LLMs as per your setup
try:
    llm = ChatOllama(model="llama2") # Ensure 'llama2' model is available in Ollama
except Exception as e:
    print(f"Warning: Could not initialize ChatOllama. Ensure Ollama is running and model 'llama2' is pulled. Error: {e}")
    print("Falling back to a dummy LLM for demonstration. Tool calls will not be executed by a real LLM.")
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import ToolMessage
    class DummyChatLLM(BaseChatModel):
        def invoke(self, messages: List[BaseMessage], **kwargs):
            last_user_message = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
            if "laptop" in last_user_message.lower() and "gaming" in last_user_message.lower():
                # Simulate a tool call to search_products_tool
                return AIMessage(tool_calls=[{"name": "search_products_tool", "args": {"query": "gaming laptop"}}])
            elif "expensive" in last_user_message.lower() and "$1500" in last_user_message.lower():
                 return AIMessage(tool_calls=[{"name": "search_products_tool", "args": {"query": "gaming laptop", "max_price": 1500}}])
            elif "brandx" in last_user_message.lower() and "15-inch" in last_user_message.lower():
                return AIMessage(tool_calls=[{"name": "search_products_tool", "args": {"query": "laptop", "brand": "BrandX", "screen_size": 15}}])
            elif "add to cart p102" in last_user_message.lower():
                return AIMessage(tool_calls=[{"name": "add_to_cart_tool", "args": {"product_id": "p102", "quantity": 1}}])
            elif "details p101" in last_user_message.lower():
                return AIMessage(tool_calls=[{"name": "get_product_details_tool", "args": {"product_id": "p101"}}])
            else:
                return AIMessage(content=f"I received your message: '{last_user_message}'. I can't process this with dummy LLM logic right now, but I can simulate specific tool calls. Try 'I need a gaming laptop' or 'add to cart p102'.")
        async def ainvoke(self, messages: List[BaseMessage], **kwargs):
            return self.invoke(messages, **kwargs)
        @property
        def _llm_type(self) -> str:
            return "dummy-chat"

    llm = DummyChatLLM()


llm_with_tools = llm.bind_tools(ECOMMERCE_TOOLS)

def run_agent(state: AgentState) -> dict:
    user_query = state["user_query"]
    chat_history = state["chat_history"]
    product_filters = state["product_filters"]

    system_message_content = (
        "You are an AI-powered personalized product recommendation and purchase assistant for an e-commerce store. "
        "You help users find products, provide details, add to cart, and check stock. "
        "Your goal is to iteratively refine recommendations based on user feedback and available tools. "
        "If the user provides new constraints (e.g., price, brand, screen size), update the product_filters and use the search_products_tool with the refined filters. "
        "Always try to use the most relevant tool. If you can't find a product, inform the user and ask for clarification. "
        "When calling search_products_tool, combine all known filters from `product_filters` with the current query. "
        "When presenting results from `search_products_tool`, list relevant product names, brands, prices, and IDs." 
        f"Current product search parameters (do not directly modify this dictionary, use tool args): {json.dumps(product_filters, indent=2)}\n"
        "Your responses should be helpful and guide the user through their shopping journey."
    )

    messages = [AIMessage(content=system_message_content)] + chat_history
    messages.append(HumanMessage(content=user_query))

    response = llm_with_tools.invoke(messages)
    return {"chat_history": [response]}

def call_tool(state: AgentState) -> dict:
    last_message = state["chat_history"][-1]
    tool_outputs = []
    new_filters = state["product_filters"].copy()

    for action in last_message.tool_calls:
        tool_output = None
        current_tool_name = action["name"]
        current_tool_args = action["args"]

        # Update product_filters for search_products_tool
        if current_tool_name == "search_products_tool":
            for k, v in current_tool_args.items():
                if v is not None:
                    # Special handling for numerical ranges or specific values that overwrite
                    if k in ["min_price", "max_price", "screen_size"]:
                        new_filters[k] = v
                    elif k in ["query", "brand", "category"]: # For string filters, prefer new if different, otherwise extend/overwrite
                        new_filters[k] = v
                
            # If the tool is `search_products_tool`, ensure we pass all accumulated filters
            # as its arguments to keep the LLM aware of the full context.
            # This is a simplification; a real system might have more sophisticated filter merging.
            merged_args = {**new_filters, **current_tool_args} # New args take precedence for conflicts
            print(f"DEBUG: Calling search_products_tool with merged args: {merged_args}")
            current_tool_args = merged_args
        
        # Add user_id to tools that need it
        if current_tool_name in ["add_to_cart_tool", "get_user_order_history_tool"]:
            current_tool_args["user_id"] = state["user_id"]


        for t in ECOMMERCE_TOOLS:
            if t.name == current_tool_name:
                try:
                    tool_output = t.invoke(current_tool_args)
                except Exception as e:
                    tool_output = f"Error executing tool '{current_tool_name}': {e}"
                break
        
        if tool_output is None:
            tool_output = f"Error: Tool '{current_tool_name}' not found or execution failed."
        
        tool_outputs.append((action, str(tool_output)))

    # Update state with potentially new product filters
    return {"intermediate_steps": tool_outputs, "product_filters": new_filters}


# Define the graph
workflow = StateGraph(AgentState)

workflow.add_node("agent", run_agent)
workflow.add_node("call_tool", call_tool)

workflow.set_entry_point("agent")

def should_continue(state: AgentState) -> str:
    last_message = state["chat_history"][-1]
    if not last_message.tool_calls:
        return "end"
    return "call_tool"

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "call_tool": "call_tool",
        "end": END
    }
)
workflow.add_edge('call_tool', 'agent')

app = workflow.compile()

# --- 3. User Interface (Conceptual/Streamlit-like) ---
# This part simulates a Streamlit/Gradio-like interaction loop for demonstration.

def get_assistant_response(user_input: str, current_state: dict) -> dict:
    # Initialize state for a new conversation or update existing
    if not current_state:
        current_state = {
            "chat_history": [],
            "intermediate_steps": [],
            "product_filters": {},
            "user_id": "user123" # A static user ID for demonstration
        }
    current_state["user_query"] = user_input
    
    # Run the graph
    inputs = {
        "user_query": user_input,
        "chat_history": current_state["chat_history"],
        "intermediate_steps": current_state["intermediate_steps"],
        "product_filters": current_state["product_filters"],
        "user_id": current_state["user_id"]
    }
    
    # Iterate through the graph until END node is reached
    for s in app.stream(inputs):
        for key, value in s.items():
            if key == "agent" and "chat_history" in value:
                current_state["chat_history"].extend(value["chat_history"])
            elif key == "call_tool" and "intermediate_steps" in value:
                # Add the tool execution result as a ToolMessage to chat_history
                for action, output in value["intermediate_steps"]:
                    current_state["chat_history"].append(ToolMessage(content=output, tool_call_id=action["id"] if "id" in action else ""))
                if "product_filters" in value:
                    current_state["product_filters"] = value["product_filters"] # Update filters
            elif key == "__end__":
                pass # End of graph execution
            
            # For demonstration, print immediate thoughts/actions
            # print(f"DEBUG: Current State Update: {key} -> {value}")
            
    final_output = current_state["chat_history"][-1].content if current_state["chat_history"] else "No response."
    
    return {"response": final_output, "new_state": current_state}


if __name__ == "__main__":
    print("Welcome to the AI E-commerce Assistant! Type 'exit' to quit.")
    
    current_conversation_state = {}
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        
        response_data = get_assistant_response(user_input, current_conversation_state)
        assistant_response = response_data["response"]
        current_conversation_state = response_data["new_state"]
        
        print(f"Assistant: {assistant_response}")
        # print(f"DEBUG: Current Filters after interaction: {current_conversation_state['product_filters']}")
        # print(f"DEBUG: Current Cart for user123: {USER_CARTS.get('user123', {})}")
