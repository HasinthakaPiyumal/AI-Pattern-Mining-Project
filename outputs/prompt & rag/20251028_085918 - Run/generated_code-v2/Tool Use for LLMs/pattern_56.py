
import os
from typing import List, Union

from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentExecutor, AgentType
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate

# --- 1. Mock Tool Definitions ---

def get_order_status(order_id: str) -> str:
    """Fetches the current status of a customer's order."""
    if order_id == "ORDER123":
        return "Order ORDER123 is currently 'Shipped' and expected on 2024-07-20."
    elif order_id == "ORDER456":
        return "Order ORDER456 is currently 'Processing'."
    else:
        return f"Order {order_id} not found."

def modify_order(order_id: str, new_item: str) -> str:
    """Modifies an existing order by adding a new item."""
    if order_id in ["ORDER123", "ORDER456"]:
        return f"Order {order_id} successfully updated to include {new_item}. Please allow 24 hours for system reflection."
    else:
        return f"Cannot modify order {order_id}: Order not found or not modifiable."

def search_faq(query: str) -> str:
    """Searches the knowledge base for answers to frequently asked questions or troubleshooting guides."""
    if "return policy" in query.lower():
        return "Our return policy allows returns within 30 days of purchase with original receipt. See full policy at example.com/returns."
    elif "troubleshoot internet" in query.lower():
        return "To troubleshoot internet connectivity, please restart your router and modem. If the issue persists, contact technical support."
    else:
        return f"No direct FAQ found for '{query}'. Please try rephrasing or contact support."

def get_customer_history(customer_id: str) -> str:
    """Retrieves the interaction history and profile details for a given customer."""
    if customer_id == "CUST001":
        return "Customer CUST001 (John Doe): Has 3 previous orders, last contact regarding a shipping delay on 2024-06-01."
    else:
        return f"Customer {customer_id} not found in CRM."

def track_shipping_with_carrier(tracking_number: str, carrier: str) -> str:
    """Uses an external API to track a package with a specific shipping carrier."""
    if carrier.lower() == "fedex" and tracking_number == "FEDEX789":
        return "FedEx tracking FEDEX789: Package out for delivery, expected today."
    elif carrier.lower() == "ups" and tracking_number == "UPS456":
        return "UPS tracking UPS456: Package arrived at local distribution center."
    else:
        return f"Could not track {tracking_number} with {carrier}. Invalid details or carrier not supported."

# --- 2. Wrap Tools for Langchain ---

llm_tools = [
    Tool(
        name="GetOrderStatus",
        func=get_order_status,
        description="Useful for when you need to get the current status of a customer's order. Input should be an order ID (string)."
    ),
    Tool(
        name="ModifyOrder",
        func=modify_order,
        description="Useful for when you need to modify an existing order, typically by adding a new item. Input should be order ID (string) and the new item (string)."
    ),
    Tool(
        name="SearchFAQ",
        func=search_faq,
        description="Useful for when you need to find answers to common questions or troubleshooting steps from the knowledge base. Input should be a search query (string)."
    ),
    Tool(
        name="GetCustomerHistory",
        func=get_customer_history,
        description="Useful for when you need to retrieve a customer's past interactions and profile information. Input should be a customer ID (string)."
    ),
    Tool(
        name="TrackShippingWithCarrier",
        func=track_shipping_with_carrier,
        description="Useful for when you need to track a package using a tracking number and a specific carrier (e.g., FedEx, UPS). Input should be the tracking number (string) and the carrier name (string)."
    ),
]

# --- 3. Initialize LLM ---
# Ensure you have your OpenAI API key set as an environment variable (OPENAI_API_KEY)
# For local development or testing without an actual OpenAI API key, you might need
# to mock the LLM or use a local model if configured.

try:
    llm = ChatOpenAI(temperature=0.7, model_name="gpt-4o") # or gpt-3.5-turbo, gpt-4, etc.
except Exception as e:
    print(f"Warning: Could not initialize ChatOpenAI. Ensure OPENAI_API_KEY is set. Error: {e}")
    print("Falling back to a mock LLM for demonstration. Functionality will be limited.")
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import BaseMessage, AIMessage
    
    class MockChatOpenAI(BaseChatModel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
        
        def _generate(self, messages: List[BaseMessage], stop: Union[List[str], None] = None, **kwargs) -> dict:
            # Simple mock response logic for demonstration without actual API key
            last_user_message = messages[-1].content if messages else ""
            if "order status for ORDER123" in last_user_message:
                return {"generations": [[AIMessage(content="Tool call: GetOrderStatus(order_id='ORDER123')")]], "llm_output": {}}
            elif "track my fedex package FEDEX789" in last_user_message:
                return {"generations": [[AIMessage(content="Tool call: TrackShippingWithCarrier(tracking_number='FEDEX789', carrier='fedex')")]], "llm_output": {}}
            elif "I need to know about your return policy" in last_user_message:
                return {"generations": [[AIMessage(content="Tool call: SearchFAQ(query='return policy')")]], "llm_output": {}}
            else:
                return {"generations": [[AIMessage(content="I'm sorry, I cannot fulfill that request with my current tools or understanding. Could you please rephrase?")]], "llm_output": {}}
        
        @property
        def _llm_type(self) -> str:
            return "mock-chat-openai"

    llm = MockChatOpenAI()

# --- 4. Agent Executor Configuration ---

system_message = (
    "You are an advanced AI customer support agent. Your goal is to resolve customer inquiries by dynamically "
    "composing and chaining the available tools. You are designed to 'innovate' and find novel ways to combine "
    "tools even if the exact solution path hasn't been explicitly shown to you. "
    "Think step-by-step and leverage your reasoning capabilities to solve complex, multi-faceted problems. "
    "Prioritize accurate information and efficient problem resolution. If you need more information from the user, ask for it clearly. "
    "Once you have fully resolved the issue or answered the question, provide a concise summary of the resolution."
)

agent_executor = initialize_agent(
    llm_tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS, # This agent type is excellent for tool composition
    verbose=True,
    agent_kwargs={
        "system_message": system_message,
    },
    handle_parsing_errors=True # Robustness for potential parsing issues
)

# --- 5. Main Interaction Loop ---

def run_customer_support_agent():
    print("\nWelcome to the AI Customer Support! How can I assist you today? (Type 'exit' to quit)")
    while True:
        user_query = input("\nCustomer: ")
        if user_query.lower() == 'exit':
            print("Thank you for contacting support. Goodbye!")
            break
        
        try:
            print("\nAgent Thinking...")
            response = agent_executor.invoke({"input": user_query})
            print(f"\nAgent: {response['output']}")
        except Exception as e:
            print(f"\nAgent Error: An unexpected error occurred: {e}")
            print("Please try again or contact a human agent if the issue persists.")

if __name__ == "__main__":
    # Set your OpenAI API key as an environment variable
    # os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
    
    run_customer_support_agent()
