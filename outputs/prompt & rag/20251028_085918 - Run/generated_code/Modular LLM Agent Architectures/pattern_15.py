import os
from dotenv import load_dotenv
from typing import Any, List, Optional, Type

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.agents import AgentFinish
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

# --- 0. Environment Setup ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- 1. External Tool Interfaces (tools/) ---

class OrderManagementTool(BaseTool):
    name: str = "order_management_tool"
    description: str = "Useful for retrieving order status and details."

    def _run(self, query: str) -> str:
        # In a real application, this would interact with an order management system
        if "status" in query:
            order_id = self._extract_order_id(query)
            if order_id:
                return f"Order {order_id} status: Shipped. Estimated delivery: 2-3 business days."
            return "Please provide an order ID to check its status."
        elif "details" in query:
            order_id = self._extract_order_id(query)
            if order_id:
                return f"Order {order_id} details: Items: Laptop, Mouse. Total: $1200. Shipping Address: 123 Main St."
            return "Please provide an order ID to get its details."
        return "Unsupported order management query."

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("Async not implemented for OrderManagementTool")

    def _extract_order_id(self, query: str) -> Optional[str]:
        # Simple regex or keyword extraction for order ID
        import re
        match = re.search(r'order\s*ID\s*(\w+)|order\s*number\s*(\w+)', query, re.IGNORECASE)
        if match:
            return match.group(1) or match.group(2)
        return None


class InventoryTool(BaseTool):
    name: str = "inventory_tool"
    description: str = "Useful for checking product stock availability."

    def _run(self, product_name: str) -> str:
        # In a real application, this would interact with an inventory system
        if "laptop" in product_name.lower():
            return f"Stock for {product_name}: 50 units available."
        elif "mouse" in product_name.lower():
            return f"Stock for {product_name}: 200 units available."
        else:
            return f"Stock for {product_name}: Currently out of stock."

    async def _arun(self, product_name: str) -> str:
        raise NotImplementedError("Async not implemented for InventoryTool")


class KnowledgeBaseTool(BaseTool):
    name: str = "knowledge_base_tool"
    description: str = "Useful for searching the knowledge base for FAQs and product information."

    def _run(self, query: str) -> str:
        # In a real application, this would query a knowledge base (e.g., vector database)
        if "return policy" in query.lower():
            return "Our return policy allows returns within 30 days of purchase with the original receipt."
        elif "shipping costs" in query.lower():
            return "Standard shipping within the US is $5.99. Free shipping for orders over $50."
        elif "warranty" in query.lower():
            return "All electronics come with a 1-year manufacturer's warranty."
        return f"No direct answer found in the knowledge base for '{query}'. Please rephrase or contact support."

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("Async not implemented for KnowledgeBaseTool")


class CRMTool(BaseTool):
    name: str = "crm_tool"
    description: str = "Useful for retrieving customer contact and preference details."

    def _run(self, customer_id: str) -> str:
        # In a real application, this would interact with a CRM system
        if customer_id == "CUST123":
            return f"Customer ID {customer_id}: Name: Jane Doe, Email: jane.doe@example.com, Preferred Contact: Email. VIP Status: Gold."
        return f"Customer ID {customer_id} not found in CRM."

    async def _arun(self, customer_id: str) -> str:
        raise NotImplementedError("Async not implemented for CRMTool")


# --- 2. Memory Module (modules/memory_module.py logic integrated) ---

class CustomerHistoryManager:
    """Simulates retrieval of customer-specific data from a mock database."""
    def get_past_orders(self, customer_id: str) -> List[str]:
        if customer_id == "CUST123":
            return ["Order_XYZ_2023-01-15: Laptop, Mouse", "Order_ABC_2022-11-01: Keyboard"]
        return []

    def get_customer_preferences(self, customer_id: str) -> dict:
        if customer_id == "CUST123":
            return {"newsletter_subscribed": True, "product_categories_of_interest": ["electronics", "accessories"]}
        return {}

# --- 3. Core LLM & Orchestration (agent.py logic integrated) ---

class CustomerSupportAgent:
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key, model_name="gpt-4o")
        self.memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=5)
        self.customer_history_manager = CustomerHistoryManager()

        self.tools = [
            OrderManagementTool(),
            InventoryTool(),
            KnowledgeBaseTool(),
            CRMTool(),
            # More tools can be added here
        ]

        # Define the agent prompt
        self.prompt_template = PromptTemplate.from_template(
            """You are an intelligent customer support agent for an e-commerce company.
            You have access to various tools to assist customers with their inquiries.
            Your goal is to provide accurate, personalized, and efficient support.
            Always try to use the available tools to find the most relevant information.
            If a query requires information about a customer, try to infer their ID if not explicitly provided.
            If you use a tool, explain what you found.
            
            Chat History: {chat_history}
            
            Current Customer Query: {input}
            
            {agent_scratchpad}
            """
        )

        self.agent = create_react_agent(self.llm, self.tools, self.prompt_template)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True, memory=self.memory)

    def run(self, query: str) -> Any:
        print(f"\nCustomer: {query}")
        response = self.agent_executor.invoke({"input": query})
        agent_response = response["output"]
        print(f"Agent: {agent_response}")
        return agent_response

# --- Main execution block (main.py logic integrated) ---
if __name__ == "__main__":
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY not found in .env file.")
        exit()

    agent = CustomerSupportAgent(openai_api_key=OPENAI_API_KEY)

    print("Intelligent Customer Support Agent initialized. Type 'exit' to quit.")

    while True:
        user_query = input("\nUser (or type 'exit'): ")
        if user_query.lower() == 'exit':
            break
        agent.run(user_query)

    print("Exiting customer support agent.")
