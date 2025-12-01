
import re

# 1. Tool Registry: Simulated external tools
def ProductCatalogSearch(product_name: str) -> str:
    if "laptop" in product_name.lower():
        return "Found 'XYZ Laptop Pro' (SKU: LAP-XYZ-PRO) - High performance, 16GB RAM, 512GB SSD. Price: $1200."
    elif "mouse" in product_name.lower():
        return "Found 'Ergonomic Wireless Mouse' (SKU: MOUSE-ERGO-WL) - USB-C rechargeable. Price: $35."
    return f"No product found matching '{product_name}'."

def OrderTrackingSystem(order_id: str) -> str:
    if order_id == "ORD-12345":
        return "Order ORD-12345: Shipped on 2023-10-26, ETA 2023-10-30. Tracking: TRK7890123."
    elif order_id == "ORD-67890":
        return "Order ORD-67890: Processing. Estimated ship date: 2023-11-05."
    return f"Order '{order_id}' not found in the system."

def KnowledgeBaseQuery(topic: str) -> str:
    if "return policy" in topic.lower():
        return "Our return policy allows returns within 30 days of purchase for a full refund. Items must be in original condition."
    elif "warranty" in topic.lower():
        return "All electronics come with a 1-year manufacturer's warranty. Extended warranties are available for purchase."
    return f"No relevant knowledge base article found for '{topic}'."

def TroubleshootingFlows(issue_type: str) -> str:
    if "internet connection" in issue_type.lower():
        return "Initiating internet connection troubleshooting. Please check router lights, restart your modem, and re-test connection. For further assistance, contact technical support."
    elif "software installation" in issue_type.lower():
        return "Initiating software installation troubleshooting. Please ensure system requirements are met, disable antivirus temporarily, and run installer as administrator. Review installation logs if issue persists."
    return f"No specific troubleshooting flow for '{issue_type}'. Please describe your issue in more detail."

tool_registry = {
    "ProductCatalogSearch": ProductCatalogSearch,
    "OrderTrackingSystem": OrderTrackingSystem,
    "KnowledgeBaseQuery": KnowledgeBaseQuery,
    "TroubleshootingFlows": TroubleshootingFlows,
}

# 2. Tool Definitions and Demonstrations for Prompt Engineering
tool_definitions = """
Available tools:
ProductCatalogSearch(product_name: str) -> str: Searches the product database for details.
OrderTrackingSystem(order_id: str) -> str: Retrieves order details and shipping status.
KnowledgeBaseQuery(topic: str) -> str: Queries an internal knowledge base for FAQs or troubleshooting steps.
TroubleshootingFlows(issue_type: str) -> str: Initiates a guided troubleshooting flow for common issues.
"""

tool_demonstrations = """
Example 1:
Customer: Where is my order ORD-12345?
CALL_TOOL: OrderTrackingSystem(order_id='ORD-12345')
OBSERVATION: OrderTrackingSystem_Output: Order ORD-12345: Shipped on 2023-10-26, ETA 2023-10-30. Tracking: TRK7890123.
Agent: Your order ORD-12345 was shipped on October 26th and is expected to arrive by October 30th. The tracking number is TRK7890123.

Example 2:
Customer: What is your return policy?
CALL_TOOL: KnowledgeBaseQuery(topic='return policy')
OBSERVATION: KnowledgeBaseQuery_Output: Our return policy allows returns within 30 days of purchase for a full refund. Items must be in original condition.
Agent: Our return policy states that you can return items within 30 days of purchase for a full refund, provided they are in their original condition.

Example 3:
Customer: I'm looking for a powerful laptop.
CALL_TOOL: ProductCatalogSearch(product_name='powerful laptop')
OBSERVATION: ProductCatalogSearch_Output: Found 'XYZ Laptop Pro' (SKU: LAP-XYZ-PRO) - High performance, 16GB RAM, 512GB SSD. Price: $1200.
Agent: We have the 'XYZ Laptop Pro', a high-performance model with 16GB RAM and a 512GB SSD, priced at $1200. Is this something you'd be interested in?

Example 4:
Customer: My internet connection is not working.
CALL_TOOL: TroubleshootingFlows(issue_type='internet connection')
OBSERVATION: TroubleshootingFlows_Output: Initiating internet connection troubleshooting. Please check router lights, restart your modem, and re-test connection. For further assistance, contact technical support.
Agent: I can help you troubleshoot your internet connection. Please try checking your router lights, restarting your modem, and then re-testing your connection. If the issue persists, you may need to contact technical support.
"""

# 3. Prompt Engineering Module
def construct_prompt(
    system_instructions: str,
    tool_definitions: str,
    tool_demonstrations: str,
    customer_query: str,
    observation: str = ""
) -> str:
    prompt_parts = [
        system_instructions,
        tool_definitions,
        tool_demonstrations,
        f"Customer: {customer_query}"
    ]
    if observation:
        prompt_parts.append(f"OBSERVATION: {observation}")
    prompt_parts.append("Agent:")
    return "\n".join(prompt_parts)

# 4. Simulated LLM Core (very basic simulation)
def simulate_llm_response(prompt: str) -> str:
    # This is a highly simplified simulation. In a real application, this would be an API call to an actual LLM.
    # We are using simple keyword matching to simulate the LLM's decision-making based on the prompt content.

    if "order" in prompt.lower() and "where is" in prompt.lower() and "ord-" in prompt.lower():
        order_id_match = re.search(r"ord-(\d+)", prompt.lower())
        if order_id_match:
            order_id = "ORD-" + order_id_match.group(1).upper()
            return f"CALL_TOOL: OrderTrackingSystem(order_id='{order_id}')"
        return "Please provide a valid order ID."
    elif "return policy" in prompt.lower():
        return "CALL_TOOL: KnowledgeBaseQuery(topic='return policy')"
    elif "warranty" in prompt.lower():
        return "CALL_TOOL: KnowledgeBaseQuery(topic='warranty')"
    elif "looking for a laptop" in prompt.lower() or "powerful laptop" in prompt.lower():
        return "CALL_TOOL: ProductCatalogSearch(product_name='powerful laptop')"
    elif "troubleshoot internet" in prompt.lower() or "internet connection not working" in prompt.lower():
        return "CALL_TOOL: TroubleshootingFlows(issue_type='internet connection')"
    elif "troubleshoot software" in prompt.lower() or "software installation issue" in prompt.lower():
        return "CALL_TOOL: TroubleshootingFlows(issue_type='software installation')"
    elif "hello" in prompt.lower() or "hi there" in prompt.lower():
        return "Hello! How can I assist you today?"
    elif "OBSERVATION: OrderTrackingSystem_Output" in prompt:
        match = re.search(r"OBSERVATION: OrderTrackingSystem_Output: (.*)", prompt)
        if match:
            details = match.group(1).strip()
            return f"Your order details: {details}"
        return "I received the order details. How else can I help?"
    elif "OBSERVATION: KnowledgeBaseQuery_Output" in prompt:
        match = re.search(r"OBSERVATION: KnowledgeBaseQuery_Output: (.*)", prompt)
        if match:
            details = match.group(1).strip()
            return f"Regarding your query: {details}"
        return "I found some information. What specifically would you like to know?"
    elif "OBSERVATION: ProductCatalogSearch_Output" in prompt:
        match = re.search(r"OBSERVATION: ProductCatalogSearch_Output: (.*)", prompt)
        if match:
            details = match.group(1).strip()
            return f"Here's what I found: {details}"
        return "I searched the product catalog. What would you like to know about it?"
    elif "OBSERVATION: TroubleshootingFlows_Output" in prompt:
        match = re.search(r"OBSERVATION: TroubleshootingFlows_Output: (.*)", prompt)
        if match:
            details = match.group(1).strip()
            return f"Here are the troubleshooting steps: {details}"
        return "I've initiated a troubleshooting flow. Did that help?"
    else:
        return "I'm sorry, I don't understand your request. Can you please rephrase it or ask about our products, orders, or policies?"

# 5. Tool Invocation and Parsing Logic
def process_llm_output(llm_output: str, tool_registry: dict) -> tuple[str, str | None]:
    tool_call_pattern = re.compile(r"CALL_TOOL: (\w+)\((.*?)\)")
    match = tool_call_pattern.match(llm_output)

    if match:
        tool_name = match.group(1)
        args_str = match.group(2)
        args = {}
        # Basic argument parsing (can be made more robust)
        for arg_pair in re.findall(r"(\w+)=['\"](.*?)['\"]", args_str):
            args[arg_pair[0]] = arg_pair[1]

        if tool_name in tool_registry:
            try:
                tool_function = tool_registry[tool_name]
                tool_result = tool_function(**args)
                return f"{tool_name}_Output: {tool_result}", "tool_called"
            except TypeError as e:
                return f"Error calling tool {tool_name} with args {args}: {e}", "error"
        else:
            return f"Error: Tool '{tool_name}' not found in registry.", "error"
    else:
        return llm_output, "final_answer"

# 6. Main Agent Loop
def customer_support_agent(initial_customer_query: str) -> str:
    system_instructions = "You are a helpful customer support agent. Your goal is to assist customers using the available tools. Respond clearly and concisely."
    current_query = initial_customer_query
    conversation_history = []
    observation = ""

    print(f"--- Customer Query: {initial_customer_query} ---")

    # First turn or subsequent turn with observation
    prompt = construct_prompt(system_instructions, tool_definitions, tool_demonstrations, current_query, observation)
    conversation_history.append(f"Prompt:\n{prompt}")

    llm_response = simulate_llm_response(prompt)
    conversation_history.append(f"LLM Raw Response: {llm_response}")

    processed_output, response_type = process_llm_output(llm_response, tool_registry)

    if response_type == "tool_called":
        print(f"Agent invoked tool. Tool output: {processed_output}")
        observation = processed_output
        # Second turn: LLM processes observation and provides final answer
        prompt_with_observation = construct_prompt(system_instructions, tool_definitions, tool_demonstrations, current_query, observation)
        conversation_history.append(f"Prompt (with observation):\n{prompt_with_observation}")
        final_llm_response = simulate_llm_response(prompt_with_observation)
        conversation_history.append(f"LLM Final Response: {final_llm_response}")
        print(f"Agent Final Answer: {final_llm_response}")
        return final_llm_response
    elif response_type == "final_answer":
        print(f"Agent Final Answer: {processed_output}")
        return processed_output
    else: # error case
        print(f"Agent Error: {processed_output}")
        return f"An error occurred: {processed_output}. Please try again."


if __name__ == "__main__":
    print("\n===== Test Case 1: Order Tracking =====")
    customer_support_agent("Where is my order ORD-12345?")

    print("\n===== Test Case 2: Knowledge Base Query =====")
    customer_support_agent("What is your return policy?")

    print("\n===== Test Case 3: Product Search =====")
    customer_support_agent("I'm looking for a powerful laptop.")

    print("\n===== Test Case 4: Troubleshooting =====")
    customer_support_agent("My internet connection is not working.")

    print("\n===== Test Case 5: Direct Answer (no tool needed) =====")
    customer_support_agent("Hello, how are you?")

    print("\n===== Test Case 6: Invalid Order ID (Simulated LLM handling) =====")
    customer_support_agent("Where is my order 999?")

    print("\n===== Test Case 7: Unknown Query =====")
    customer_support_agent("Tell me a joke.")
