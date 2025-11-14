import re
from typing import Any, Type, Dict, List, Optional
from pydantic import BaseModel, Field

# Mock Databases
PRODUCT_CATALOG = {
    "SKU123": {"name": "Wireless Headphones", "price": 99.99, "availability": "In Stock", "description": "High-quality wireless headphones with noise cancellation."},
    "SKU456": {"name": "Smartwatch X", "price": 199.99, "availability": "Low Stock", "description": "Advanced smartwatch with health tracking and notifications."},
    "SKU789": {"name": "Ergonomic Office Chair", "price": 249.00, "availability": "Out of Stock", "description": "Comfortable office chair designed for long working hours."},
    "keyboard": {"name": "Mechanical Keyboard", "price": 75.00, "availability": "In Stock", "description": "RGB Mechanical Keyboard with clicky switches."}
}

ORDER_DATABASE = {
    "ORD001": {"customer": "Alice Smith", "items": ["SKU123"], "status": "Shipped", "tracking": "TRK987654321"},
    "ORD002": {"customer": "Bob Johnson", "items": ["SKU456", "SKU123"], "status": "Processing", "tracking": "N/A"},
    "ORD003": {"customer": "Charlie Brown", "items": ["SKU789"], "status": "Delivered", "tracking": "TRK123456789"}
}

KNOWLEDGE_BASE = {
    "return policy": "Our return policy allows returns within 30 days of purchase for a full refund. Items must be in their original condition.",
    "shipping costs": "Standard shipping within the country is $5.99. Express shipping is $15.99. International shipping varies by destination.",
    "payment methods": "We accept Visa, MasterCard, American Express, PayPal, and Apple Pay.",
    "reset headphones": "To reset your wireless headphones, hold the power button and volume down button for 10 seconds until the indicator light blinks.",
    "smartwatch battery life": "The Smartwatch X typically has a battery life of up to 48 hours, depending on usage."
}

# 1. Core LLM Component (Simulated Langchain LLM)
# This uses a basic FakeListChatModel for demonstration purposes.
# In a real application, this would be replaced with an actual LLM integration (e.g., OpenAI, HuggingFace).
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.callbacks import CallbackManagerForLLMRun

class FakeListChatModel(BaseChatModel):
    """Fake ChatModel for testing purposes, provides responses from a predefined list."""
    responses: List[str]
    i: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake-list-chat-model"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if not self.responses:
            raise ValueError("No responses provided for FakeListChatModel.")
        response = self.responses[self.i % len(self.responses)] # Cycle through responses
        self.i += 1
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=response))])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages, stop, run_manager, **kwargs) # Simplified async


# 2. Plug-and-Play LLM Modules (Tools)
from langchain.tools import BaseTool

class ProductSearchInput(BaseModel):
    query: str = Field(description="The product name, keyword, or SKU to search for.")

class ProductCatalogTool(BaseTool):
    name: str = "product_catalog_tool"
    description: str = "Searches the product catalog for product details like name, price, availability, and description."
    args_schema: Type[BaseModel] = ProductSearchInput

    def _run(self, query: str) -> str:
        query_lower = query.lower()
        if query_lower in PRODUCT_CATALOG:
            product = PRODUCT_CATALOG[query_lower]
            return (f"Product Name: {product['name']}, Price: ${product['price']:.2f}, "
                    f"Availability: {product['availability']}, Description: {product['description']}")
        for sku, product in PRODUCT_CATALOG.items():
            if query_lower in product['name'].lower() or query_lower in sku.lower():
                return (f"Product Name: {product['name']}, Price: ${product['price']:.2f}, "
                        f"Availability: {product['availability']}, Description: {product['description']}")
        return f"Could not find product details for '{query}'."

    async def _arun(self, query: str) -> str:
        return self._run(query)

class OrderSearchInput(BaseModel):
    order_id: str = Field(description="The order ID to search for.")

class OrderManagementTool(BaseTool):
    name: str = "order_management_tool"
    description: str = "Retrieves order status and tracking information based on the order ID."
    args_schema: Type[BaseModel] = OrderSearchInput

    def _run(self, order_id: str) -> str:
        order = ORDER_DATABASE.get(order_id.upper()) # Ensure case-insensitivity for common IDs
        if order:
            return (f"Order ID: {order_id}, Customer: {order['customer']}, "
                    f"Items: {', '.join(order['items'])}, Status: {order['status']}, "
                    f"Tracking: {order['tracking']}")
        return f"Order with ID '{order_id}' not found."

    async def _arun(self, order_id: str) -> str:
        return self._run(order_id)

class TroubleshootingInput(BaseModel):
    issue: str = Field(description="The issue or question to find a solution for.")

class TroubleshootingTool(BaseTool):
    name: str = "troubleshooting_tool"
    description: str = "Accesses a knowledge base to provide solutions to common issues or FAQs."
    args_schema: Type[BaseModel] = TroubleshootingInput

    def _run(self, issue: str) -> str:
        issue_lower = issue.lower()
        for key, solution in KNOWLEDGE_BASE.items():
            if issue_lower in key or issue_lower in solution.lower():
                return solution
        return f"No specific solution found for '{issue}'. Please rephrase your question or contact live support."

    async def _arun(self, issue: str) -> str:
        return self._run(issue)

# 3. Agentic Architecture Components
from langchain.memory import ConversationBufferWindowMemory

class PlanningModule:
    """
    Simulates the Planning Module for Cognitive Load Management.
    For simplicity, it uses keyword matching to suggest a plan.
    In a real scenario, this would involve a more sophisticated LLM call
    to break down complex tasks into executable steps.
    """
    def generate_plan(self, query: str) -> Optional[List[str]]:
        query_lower = query.lower()
        if ("order status" in query_lower or "track my order" in query_lower) and ("product" in query_lower or "item" in query_lower):
            return ["First, identify the product details if mentioned.", "Then, check the order status using the order ID."]
        elif "product" in query_lower and ("price" in query_lower or "availability" in query_lower or "details" in query_lower):
            return ["Search product catalog for details."]
        elif "order" in query_lower and ("track" in query_lower or "status" in query_lower):
            return ["Check order management system for status and tracking."]
        elif "troubleshoot" in query_lower or "issue with" in query_lower or "problem with" in query_lower or "help with" in query_lower:
            return ["Consult the troubleshooting knowledge base."]
        return None # No specific multi-step plan needed, can be handled by direct tool use or LLM.

# 4. MRKL System (`MRKLRouter`)
class MRKLRouter:
    """
    Intelligent routing mechanism to select the most appropriate tool or
    direct to the LLM. For this mock, it uses keyword-based routing.
    """
    def __init__(self, tools: List[BaseTool]):
        self.tools = {tool.name: tool for tool in tools}

    def route_query(self, query: str) -> Dict[str, Any]:
        query_lower = query.lower()

        # Product Catalog Tool routing
        product_keywords = ["product", "item", "price", "available", "availability", "description", "details of", "about"]
        if any(keyword in query_lower for keyword in product_keywords):
            # Attempt to extract a specific product query
            match = re.search(r"(?:product|item|sku)\s*(\w+)|details (?:of|for)\s*([\w\s]+)|about\s*([\w\s]+)", query_lower)
            extracted_query = next((m for m in match.groups() if m is not None), query) if match else query
            return {"action": "tool_use", "tool_name": "product_catalog_tool", "tool_input": {"query": extracted_query.strip()}}

        # Order Management Tool routing
        order_keywords = ["order", "status", "track", "shipment", "my order"]
        if any(keyword in query_lower for keyword in order_keywords):
            match = re.search(r"order (?:id )?(\w+)", query_lower)
            if match:
                order_id = match.group(1).upper()
                return {"action": "tool_use", "tool_name": "order_management_tool", "tool_input": {"order_id": order_id}}
            return {"action": "tool_use", "tool_name": "order_management_tool", "tool_input": {"order_id": "unknown"}} # Requires an ID

        # Troubleshooting Tool routing
        troubleshooting_keywords = ["troubleshoot", "issue", "problem", "fix", "help with", "faq"]
        if any(keyword in query_lower for keyword in troubleshooting_keywords):
            return {"action": "tool_use", "tool_name": "troubleshooting_tool", "tool_input": {"issue": query}}

        return {"action": "llm_direct"} # If no specific tool, direct to LLM


# 5. ShopAssistProAgent (Main Orchestrator)
class ShopAssistProAgent:
    """
    The central orchestrator for ShopAssist Pro, integrating LLM, tools, memory,
    planning, and routing to handle customer inquiries.
    """
    def __init__(self):
        # 1. Core LLM Component (Simulated)
        self.llm = FakeListChatModel(responses=["Hello! How can I assist you today?"])

        # 2. Plug-and-Play LLM Modules (Tools)
        self.product_tool = ProductCatalogTool()
        self.order_tool = OrderManagementTool()
        self.troubleshooting_tool = TroubleshootingTool()
        self.all_tools = [self.product_tool, self.order_tool, self.troubleshooting_tool]

        # 3. Agentic Architecture Components
        self.memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=5)
        self.planning_module = PlanningModule()

        # 4. MRKL System
        self.router = MRKLRouter(self.all_tools)

        print("ShopAssist Pro Agent initialized.")

    def _synthesize_response_with_llm(self, user_query: str, tool_output: Optional[str] = None, plan_output: Optional[List[str]] = None) -> str:
        """
        Synthesizes a coherent response. In this mock, it prioritizes tool_output.
        For generic LLM responses, it uses predefined responses from the FakeListChatModel.
        """
        if tool_output:
            return tool_output # Directly return tool output for simplicity in mock
        
        # Simulate LLM's general conversational responses
        query_lower = user_query.lower()
        if "hello" in query_lower or "hi" in query_lower:
            self.llm.responses = ["Hello! How can I assist you today?"]
        elif "thanks" in query_lower or "thank you" in query_lower:
            self.llm.responses = ["You're welcome! Is there anything else I can help you with?"]
        else:
            self.llm.responses = ["I understand. How else may I assist you?", "I'm here to help. What can I do for you?"]
        
        # In a real scenario, the LLM would process the full context to generate a response.
        # Here, we just pick one of the simulated responses.
        return self.llm.invoke(user_query).content


    def run(self, user_query: str) -> str:
        """
        Processes a user query through the ShopAssist Pro agent flow.
        """
        print(f"\n--- User Query: {user_query} ---")
        # 1. Add query to AgenticWorkingMemory
        self.memory.chat_memory.add_user_message(user_query)

        agent_response = ""
        tool_output = None

        # 2. Pass query to PlanningModule to check for multi-step tasks
        plan = self.planning_module.generate_plan(user_query)
        if plan:
            print(f"[Agent Planning] Detected a potential multi-step task. Plan: {'; '.join(plan)}")
            # In a full agent, this plan would guide iterative tool execution.
            # For this mock, the router proceeds to act on the immediate query based on keywords.

        # 3. Use MRKLRouter to select and invoke the relevant module/tool
        routing_decision = self.router.route_query(user_query)
        action_type = routing_decision["action"]

        if action_type == "tool_use":
            tool_name = routing_decision["tool_name"]
            tool_input = routing_decision["tool_input"]
            print(f"[Agent Routing] Decided to use tool: '{tool_name}' with input: {tool_input}")
            try:
                # Execute the chosen tool
                if tool_name == self.product_tool.name:
                    tool_output = self.product_tool.run(tool_input["query"])
                elif tool_name == self.order_tool.name:
                    tool_output = self.order_tool.run(tool_input["order_id"])
                elif tool_name == self.troubleshooting_tool.name:
                    tool_output = self.troubleshooting_tool.run(tool_input["issue"])
                
                print(f"[Tool Output] {tool_output}")

            except Exception as e:
                tool_output = f"Error executing tool '{tool_name}': {e}"
                print(f"[Tool Error] {tool_output}")
            
            # 6. Synthesize response using the LLM, incorporating memory and tool results
            agent_response = self._synthesize_response_with_llm(user_query, tool_output, plan)

        elif action_type == "llm_direct":
            print("[Agent Routing] No specific tool identified. Responding directly via LLM.")
            agent_response = self._synthesize_response_with_llm(user_query, plan_output=plan)

        # 7. Store agent's response in AgenticWorkingMemory
        self.memory.chat_memory.add_ai_message(agent_response)

        # 8. Return response to the user
        return agent_response

# Example Usage:
if __name__ == "__main__":
    agent = ShopAssistProAgent()

    # --- Interaction 1: Product Search ---
    user_input = "Tell me about the Wireless Headphones."
    agent_response = agent.run(user_input)
    print(f"ShopAssist Pro: {agent_response}")

    # --- Interaction 2: Order Status ---
    user_input = "What is the status of my order ORD001?"
    agent_response = agent.run(user_input)
    print(f"ShopAssist Pro: {agent_response}")

    # --- Interaction 3: Troubleshooting ---
    user_input = "I need help with my smartwatch battery life."
    agent_response = agent.run(user_input)
    print(f"ShopAssist Pro: {agent_response}")

    # --- Interaction 4: Generic Query (LLM Direct) ---
    user_input = "Hello, how are you?"
    agent_response = agent.run(user_input)
    print(f"ShopAssist Pro: {agent_response}")

    # --- Interaction 5: Another Product Search (using SKU) ---
    user_input = "Can you give me details for SKU456?"
    agent_response = agent.run(user_input)
    print(f"ShopAssist Pro: {agent_response}")

    # --- Interaction 6: Complex Query (Planning - simplified) ---
    user_input = "I want to know the price of the Ergonomic Office Chair and then check if it's shipped."
    agent_response = agent.run(user_input)
    print(f"ShopAssist Pro: {agent_response}") 
    # Due to the mock router's current keyword-based nature, it primarily focuses on the product search.
    # A real 'Cognitive Load Management' planning system would sequence these steps.

    # --- Interaction 7: Unknown Product ---
    user_input = "What about a Quantum Flanger?"
    agent_response = agent.run(user_input)
    print(f"ShopAssist Pro: {agent_response}")

    # --- Interaction 8: Unknown Order ---
    user_input = "What about my order ABC123?"
    agent_response = agent.run(user_input)
    print(f"ShopAssist Pro: {agent_response}")

    # --- Interaction 9: Thanks ---
    user_input = "Thank you!"
    agent_response = agent.run(user_input)
    print(f"ShopAssist Pro: {agent_response}")