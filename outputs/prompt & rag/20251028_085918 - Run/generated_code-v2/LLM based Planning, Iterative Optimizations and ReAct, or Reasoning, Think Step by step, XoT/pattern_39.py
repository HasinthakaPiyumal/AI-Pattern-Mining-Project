import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from pydantic import BaseModel, Field

load_dotenv()

# 1. Define Tools

class KnowledgeBaseSearchInput(BaseModel):
    query: str = Field(description="the query to search the knowledge base for")

@tool("knowledge_base_search", args_schema=KnowledgeBaseSearchInput)
def knowledge_base_search(query: str) -> str:
    if "refund policy" in query.lower():
        return "Our refund policy states that items can be returned within 30 days of purchase for a full refund, provided they are in original condition. Digital goods are non-refundable."
    elif "shipping times" in query.lower() or "delivery" in query.lower():
        return "Standard shipping within the US takes 5-7 business days. Expedited shipping is available for an additional fee and takes 2-3 business days. International shipping varies."
    elif "account login" in query.lower():
        return "If you are having trouble logging into your account, please ensure you are using the correct email and password. You can reset your password using the 'Forgot Password' link on the login page."
    return f"No direct answer found in the knowledge base for '{query}'. Please try a different query or escalate."

class CRMSearchInput(BaseModel):
    customer_id: str = Field(description="the unique identifier for the customer")

@tool("crm_system_search", args_schema=CRMSearchInput)
def crm_system_search(customer_id: str) -> str:
    if customer_id == "CUST123":
        return "Customer CUST123: John Doe, Email: john.doe@example.com, Phone: 555-1234, Recent Orders: ORD987 (completed), ORD988 (pending). VIP Status: Gold."
    elif customer_id == "CUST456":
        return "Customer CUST456: Jane Smith, Email: jane.smith@example.com, Phone: 555-5678, Recent Orders: ORD989 (completed). VIP Status: None."
    return f"No customer found with ID '{customer_id}'."

class OrderManagementInput(BaseModel):
    order_id: str = Field(description="the unique identifier for the order")
    action: str = Field(description="the action to perform, e.g., 'check_status', 'modify_item', 'initiate_refund'")
    details: str = Field(description="additional details for the action, e.g., item to modify, reason for refund")

@tool("order_management_api", args_schema=OrderManagementInput)
def order_management_api(order_id: str, action: str, details: str = "") -> str:
    if order_id == "ORD987":
        if action == "check_status":
            return "Order ORD987 Status: Completed, Shipped on 2023-10-20. Tracking: TRK789."
        elif action == "initiate_refund":
            return "Refund initiated for Order ORD987. Processing in 3-5 business days. Reason: " + details
    elif order_id == "ORD988":
        if action == "check_status":
            return "Order ORD988 Status: Pending, Expected ship date: 2023-11-05. Items: Laptop, Mouse."
        elif action == "modify_item" and "mouse" in details.lower():
            return "Item 'Mouse' in Order ORD988 has been updated to 'Gaming Mouse'."
    return f"Could not perform '{action}' for Order '{order_id}'. Details: {details}"

class ShippingCarrierInput(BaseModel):
    tracking_number: str = Field(description="the tracking number for the shipment")

@tool("shipping_carrier_api", args_schema=ShippingCarrierInput)
def shipping_carrier_api(tracking_number: str) -> str:
    if tracking_number == "TRK789":
        return "Tracking TRK789: Package delivered on 2023-10-22 at 2:30 PM. Signed by customer."
    elif tracking_number == "TRK001":
        return "Tracking TRK001: Package currently in transit, expected delivery 2023-11-07. Last scanned in Chicago."
    return f"No tracking information found for '{tracking_number}'."

tools = [
    knowledge_base_search,
    crm_system_search,
    order_management_api,
    shipping_carrier_api
]

# 2. Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# 3. Create ReAct Agent
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful customer support agent. Use the provided tools to answer customer queries."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

agent = create_react_agent(llm, tools, prompt)

# 4. Create Agent Executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# 5. Simulate Interaction
if __name__ == "__main__":
    print("\n--- Customer Query 1: What is your refund policy? ---")
    response = agent_executor.invoke({"input": "What is your refund policy?"})
    print(f"Agent Response: {response['output']}")

    print("\n--- Customer Query 2: I need to know the status of order ORD987. My customer ID is CUST123. ---")
    response = agent_executor.invoke({"input": "I need to know the status of order ORD987. My customer ID is CUST123."})
    print(f"Agent Response: {response['output']}")

    print("\n--- Customer Query 3: Can I get a refund for order ORD987? I bought it accidentally. ---")
    response = agent_executor.invoke({"input": "Can I get a refund for order ORD987? I bought it accidentally."})
    print(f"Agent Response: {response['output']}")

    print("\n--- Customer Query 4: Where is my package with tracking number TRK001? ---")
    response = agent_executor.invoke({"input": "Where is my package with tracking number TRK001?"})
    print(f"Agent Response: {response['output']}")

    print("\n--- Customer Query 5: Update the mouse in my pending order ORD988 to a gaming mouse. ---")
    response = agent_executor.invoke({"input": "Update the mouse in my pending order ORD988 to a gaming mouse."})
    print(f"Agent Response: {response['output']}")

    print("\n--- Customer Query 6: I need help with an issue that the tools can't resolve. ---")
    response = agent_executor.invoke({"input": "I need help with an issue that the tools can't resolve."})
    print(f"Agent Response: {response['output']}")