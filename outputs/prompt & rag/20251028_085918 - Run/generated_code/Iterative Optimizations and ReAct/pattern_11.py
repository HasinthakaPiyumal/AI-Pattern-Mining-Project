import os
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# --- 1. Define Tools ---

class OrderDetailsInput(BaseModel):
    order_id: str = Field(..., description="The ID of the customer's order.")

def get_order_details(input: OrderDetailsInput) -> Dict[str, Any]:
    """Retrieves details for a specific order."""
    if input.order_id == "ORD12345":
        return {"status": "success", "order_id": input.order_id, "items": ["Laptop", "Mouse"], "total": 1200.00, "shipping_status": "Shipped", "tracking_number": "TRACK67890"}
    return {"status": "error", "message": "Order not found."}

class ProductInfoInput(BaseModel):
    product_name: str = Field(..., description="The name of the product to search for.")

def get_product_info(input: ProductInfoInput) -> Dict[str, Any]:
    """Retrieves information about a specific product."""
    if "laptop" in input.product_name.lower():
        return {"status": "success", "product_name": "ProBook Laptop", "price": 1100.00, "description": "High-performance laptop.", "availability": "In Stock"}
    elif "mouse" in input.product_name.lower():
        return {"status": "success", "product_name": "Ergonomic Mouse", "price": 50.00, "description": "Wireless ergonomic mouse.", "availability": "In Stock"}
    return {"status": "error", "message": "Product not found."}

class ShippingStatusInput(BaseModel):
    tracking_number: str = Field(..., description="The tracking number for the shipment.")

def get_shipping_status(input: ShippingStatusInput) -> Dict[str, Any]:
    """Retrieves the current shipping status of a package."""
    if input.tracking_number == "TRACK67890":
        return {"status": "success", "tracking_number": input.tracking_number, "current_status": "Out for Delivery", "estimated_delivery": "2023-11-20"}
    return {"status": "error", "message": "Tracking number not found."}

class CRMUpdateInput(BaseModel):
    customer_id: str = Field(..., description="The ID of the customer.")
    note: str = Field(..., description="A note to add to the customer's CRM record.")

def update_crm(input: CRMUpdateInput) -> Dict[str, Any]:
    """Adds a note to the customer's CRM record."""
    return {"status": "success", "customer_id": input.customer_id, "message": "CRM updated successfully."}

# --- 2. Tool Registry ---

class Tool:
    def __init__(self, name: str, description: str, func, input_model: type[BaseModel]):
        self.name = name
        self.description = description
        self.func = func
        self.input_model = input_model

    def run(self, **kwargs) -> Dict[str, Any]:
        try:
            input_obj = self.input_model(**kwargs)
            return self.func(input_obj)
        except Exception as e:
            return {"status": "error", "message": f"Tool '{self.name}' failed with error: {str(e)}"}

class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register_tool(self, tool: Tool):
        self.tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[Tool]:
        return self.tools.get(name)

    def get_tool_descriptions(self) -> str:
        descriptions = []
        for tool_name, tool in self.tools.items():
            param_schema = {k: {"type": str(v.annotation.__name__), "description": v.field_info.description} for k, v in tool.input_model.model_fields.items()}
            descriptions.append(
                f"Tool Name: {tool.name}\n"
                f"Description: {tool.description}\n"
                f"Parameters: {json.dumps(param_schema)}\n"
            )
        return "\n---\n".join(descriptions)

tool_registry = ToolRegistry()
tool_registry.register_tool(Tool("get_order_details", "Retrieves comprehensive details about a customer's order.", get_order_details, OrderDetailsInput))
tool_registry.register_tool(Tool("get_product_info", "Fetches information about a specific product in the catalog.", get_product_info, ProductInfoInput))
tool_registry.register_tool(Tool("get_shipping_status", "Checks the delivery status of a package using a tracking number.", get_shipping_status, ShippingStatusInput))
tool_registry.register_tool(Tool("update_crm", "Adds a note to a customer's CRM record to log interactions or outcomes.", update_crm, CRMUpdateInput))

# --- 3. Mock LLM Class ---
class MockLLM:
    """A mock LLM that simulates generating responses and tool calls."""
    def __init__(self):
        self.call_count = 0

    def invoke(self, prompt: str) -> str:
        self.call_count += 1
        print(f"\n--- LLM Call {self.call_count} ---")
        print(f"Prompt sent to LLM (first 500 chars):\n{prompt[:500]}...\n")

        # Simulate LLM thinking and tool calling
        if "order status" in prompt.lower() and "ORD12345" in prompt and "NONEXISTENT123" not in prompt:
            return '{"tool_name": "get_order_details", "tool_args": {"order_id": "ORD12345"}}'
        elif "product price" in prompt.lower() and "laptop" in prompt:
            return '{"tool_name": "get_product_info", "tool_args": {"product_name": "Laptop"}}'
        elif "shipping status" in prompt.lower() and "TRACK67890" in prompt:
            return '{"tool_name": "get_shipping_status", "tool_args": {"tracking_number": "TRACK67890"}}'
        elif "customer CUST001 issue resolved" in prompt and "logging to CRM" in prompt:
            return '{"tool_name": "update_crm", "tool_args": {"customer_id": "CUST001", "note": "Customer issue resolved, provided order details."}}'
        elif "refund" in prompt.lower() and "policy" in prompt.lower():
            return "The refund policy states that items can be returned within 30 days of purchase for a full refund, provided they are in their original condition. For specific issues, please provide your order ID."
        elif "order status" in prompt.lower() and "NONEXISTENT123" in prompt and "error" in prompt.lower() and "order not found" in prompt.lower():
            # This simulates the LLM self-correcting after a tool error for a non-existent order.
            return "I couldn't find details for order NONEXISTENT123. Could you please double-check the order ID or provide more information?"
        else:
            return "I am a customer support agent. How can I assist you further? If you need to know about an order, please provide the order ID. If about a product, provide the product name."

# --- 4. Intelligent Customer Support Agent ---

class IntelligentCustomerSupportAgent:
    def __init__(self, llm: MockLLM, tool_registry: ToolRegistry):
        self.llm = llm
        self.tool_registry = tool_registry
        self.conversation_history: List[Dict[str, str]] = []
        self.max_iterations = 5

    def _generate_prompt(self, user_query: str, tool_output: Optional[Dict[str, Any]] = None, feedback: Optional[str] = None) -> str:
        tool_descriptions = self.tool_registry.get_tool_descriptions()
        history_str = "\n".join([f"{entry['role']}: {entry['content']}" for entry in self.conversation_history])

        prompt_parts = [
            "You are an Intelligent Customer Support Agent for an e-commerce platform. Your goal is to resolve customer queries efficiently by using available tools and adapting your responses.",
            "You have access to the following tools:",
            tool_descriptions,
            "When you need to use a tool, respond ONLY with a JSON object in the format: `{\"tool_name\": \"<tool_name>\", \"tool_args\": {<arg1>: <value1>, ...}}`.",
            "If you have a final answer or need to ask a clarifying question, respond ONLY with a natural language answer.",
            "If a tool call fails or you receive negative feedback, try to self-correct by re-evaluating the problem, trying a different tool, or asking for more information.",
            "Current Conversation History:",
            history_str,
            f"\nCustomer Query: {user_query}"
        ]

        if tool_output:
            prompt_parts.append(f"\nPrevious Tool Output: {json.dumps(tool_output)}")
        if feedback:
            prompt_parts.append(f"\nFeedback: {feedback}\nBased on this feedback, re-evaluate your approach.")

        return "\n".join(prompt_parts)

    def _process_llm_response(self, llm_response: str) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Parses LLM response to determine if it's a tool call or a natural language response."""
        try:
            response_json = json.loads(llm_response)
            if "tool_name" in response_json and "tool_args" in response_json:
                return response_json["tool_name"], response_json["tool_args"]
        except json.JSONDecodeError:
            pass # Not a tool call, treat as natural language

        return llm_response, None # Natural language response

    def handle_query(self, user_query: str) -> str:
        self.conversation_history.append({"role": "user", "content": user_query})
        current_tool_output: Optional[Dict[str, Any]] = None
        current_feedback: Optional[str] = None

        for i in range(self.max_iterations):
            prompt = self._generate_prompt(user_query, current_tool_output, current_feedback)
            llm_response = self.llm.invoke(prompt)

            parsed_response, tool_args = self._process_llm_response(llm_response)

            if isinstance(parsed_response, str) and tool_args is None:
                # LLM provided a natural language response (potentially a final answer or a clarifying question)
                self.conversation_history.append({"role": "agent", "content": parsed_response})
                print(f"\nAgent Final Response: {parsed_response}")
                return parsed_response
            elif isinstance(parsed_response, str) and tool_args is not None:
                # LLM wants to call a tool
                tool_name = parsed_response
                tool = self.tool_registry.get_tool(tool_name)
                if tool:
                    print(f"\nAgent is calling tool: {tool_name} with args: {tool_args}")
                    tool_result = tool.run(**tool_args)
                    self.conversation_history.append({"role": "tool_call", "content": f"Called {tool_name} with {tool_args}"})
                    self.conversation_history.append({"role": "tool_output", "content": json.dumps(tool_result)})
                    print(f"Tool Output: {tool_result}")

                    if tool_result.get("status") == "error":
                        current_feedback = f"Tool '{tool_name}' returned an error: {tool_result.get('message', 'Unknown error')}. Please try a different approach or ask for more details."
                        print(f"Applying self-correction feedback: {current_feedback}")
                    else:
                        current_tool_output = tool_result
                        current_feedback = None
                else:
                    error_message = f"Agent tried to call unknown tool: {tool_name}. Re-evaluating."
                    self.conversation_history.append({"role": "error", "content": error_message})
                    print(f"\nError: {error_message}")
                    current_feedback = error_message
            else:
                error_message = f"Agent produced an unparseable response. Re-evaluating: {llm_response}"
                self.conversation_history.append({"role": "error", "content": error_message})
                print(f"\nError: {error_message}")
                current_feedback = error_message

        final_response = "I'm sorry, I've reached the maximum number of iterations and cannot fully resolve your request. Please try rephrasing or provide more details."
        self.conversation_history.append({"role": "agent", "content": final_response})
        print(f"\nAgent Final Response (Max Iterations): {final_response}")
        return final_response


# --- Main Execution ---
if __name__ == "__main__":
    llm = MockLLM()
    agent = IntelligentCustomerSupportAgent(llm=llm, tool_registry=tool_registry)

    print("--- Starting Customer Support Agent Simulation ---")

    # Scenario 1: Order Status Query
    print("\n--- Scenario 1: Customer asks about a known order ---")
    agent.conversation_history = []
    response = agent.handle_query("What is the status of my order ORD12345?")
    print(f"\nFinal Agent Output to Customer: {response}")
    print("\nConversation History:")
    for entry in agent.conversation_history:
        print(f"  {entry['role']}: {entry['content']}")

    # Scenario 2: Product Information Query
    print("\n--- Scenario 2: Customer asks about a product ---")
    agent.conversation_history = []
    response = agent.handle_query("How much does the laptop cost?")
    print(f"\nFinal Agent Output to Customer: {response}")
    print("\nConversation History:")
    for entry in agent.conversation_history:
        print(f"  {entry['role']}: {entry['content']}")

    # Scenario 3: General Question (no tool needed immediately)
    print("\n--- Scenario 3: Customer asks a general question ---")
    agent.conversation_history = []
    response = agent.handle_query("What is your refund policy?")
    print(f"\nFinal Agent Output to Customer: {response}")
    print("\nConversation History:")
    for entry in agent.conversation_history:
        print(f"  {entry['role']}: {entry['content']}")

    # Scenario 4: Query that fails initially and self-corrects (simulated)
    print("\n--- Scenario 4: Customer asks about a non-existent order (simulated self-correction) ---")
    agent.conversation_history = []
    response = agent.handle_query("What is the status of my order NONEXISTENT123?")
    print(f"\nFinal Agent Output to Customer: {response}")
    print("\nConversation History:")
    for entry in agent.conversation_history:
        print(f"  {entry['role']}: {entry['content']}")

    # Scenario 5: Query leading to CRM update (simulated success path after some interaction)
    print("\n--- Scenario 5: Customer has an issue, leading to CRM update ---")
    agent.conversation_history = []
    print("\nAgent first gets order details for customer CUST001...")
    response_step1 = agent.handle_query("What is the status of my order ORD12345? This is for customer CUST001.")
    print(f"\nAgent's initial response to customer: {response_step1}")

    print("\nSimulating customer acknowledging resolution and agent deciding to log to CRM...")
    response_step2 = agent.handle_query("Thank you for the order status. Please log that this issue is resolved for CUST001.")
    print(f"\nFinal Agent Output to Customer: {response_step2}")
    print("\nConversation History:")
    for entry in agent.conversation_history:
        print(f"  {entry['role']}: {entry['content']}")