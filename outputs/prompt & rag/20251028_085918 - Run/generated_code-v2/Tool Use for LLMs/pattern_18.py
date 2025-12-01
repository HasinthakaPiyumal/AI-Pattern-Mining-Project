import json
import re

# 1. Tool Definitions (Simulated Backend Services)
def check_order_status(order_id: str) -> str:
    if order_id == "ORD123":
        return "Order ORD123 is currently in 'Shipped' status. Estimated delivery: 2023-10-27."
    elif order_id == "ORD456":
        return "Order ORD456 is currently in 'Processing' status."
    else:
        return f"Order {order_id} not found."

def process_refund(order_id: str, amount: float) -> str:
    if order_id == "ORD123" and amount <= 100.0:
        return f"Refund of ${amount} for order {order_id} processed successfully."
    elif order_id == "ORD789":
        return "Refund failed: Order ORD789 is not eligible for a refund."
    else:
        return f"Refund for order {order_id} could not be processed with amount ${amount}."

def update_shipping_address(order_id: str, new_address: str) -> str:
    if order_id == "ORD123":
        return f"Shipping address for order {order_id} updated to '{new_address}'."
    elif order_id == "ORD456":
        return "Shipping address for order ORD456 cannot be updated at this stage."
    else:
        return f"Order {order_id} not found, cannot update address."

# Map tool names to their functions
TOOL_FUNCTIONS = {
    "check_order_status": check_order_status,
    "process_refund": process_refund,
    "update_shipping_address": update_shipping_address,
}

# 2. Prompt Engineering Module

TOOL_DESCRIPTIONS = """
You have access to the following tools:

1. check_order_status(order_id: str): Checks the current status of an e-commerce order. Returns the order status and estimated delivery if available.
2. process_refund(order_id: str, amount: float): Processes a refund for a specified order and amount. Returns success or failure message.
3. update_shipping_address(order_id: str, new_address: str): Updates the shipping address for an order. Returns confirmation or error.

To use a tool, respond with a 'Tool Call:' followed by the tool function call and its arguments in Python syntax.
Example: Tool Call: check_order_status(order_id="ORD123")
After a tool call, you will receive an 'Observation:' with the tool's output.
"""

FEW_SHOT_EXAMPLES = [
    """User Query: What is the status of my order ORD123?
Thought: The user is asking for an order status, so I should use the check_order_status tool.
Tool Call: check_order_status(order_id="ORD123")
Observation: Order ORD123 is currently in 'Shipped' status. Estimated delivery: 2023-10-27.
Assistant: Your order ORD123 is shipped and is expected to be delivered by 2023-10-27.""",

    """User Query: I need to refund order ORD456 for $50.
Thought: The user wants to process a refund. I need to call the process_refund tool.
Tool Call: process_refund(order_id="ORD456", amount=50.0)
Observation: Refund for order ORD456 could not be processed with amount $50.0.
Assistant: I'm sorry, I was unable to process a refund for order ORD456 for $50.0.""",

    """User Query: Can you change the delivery address for order ORD123 to 123 Main St, Anytown?
Thought: The user wants to update the shipping address. I should use the update_shipping_address tool.
Tool Call: update_shipping_address(order_id="ORD123", new_address="123 Main St, Anytown")
Observation: Shipping address for order ORD123 updated to '123 Main St, Anytown'.
Assistant: The shipping address for your order ORD123 has been updated to 123 Main St, Anytown.""",
]

def DynamicPromptConstructor(user_query: str, chat_history: list = None) -> str:
    prompt_parts = [
        "You are an intelligent customer support assistant for an e-commerce platform.",
        TOOL_DESCRIPTIONS,
        "Here are some examples of how to use the tools:",
    ]
    prompt_parts.extend(FEW_SHOT_EXAMPLES)

    if chat_history:
        for turn in chat_history:
            prompt_parts.append(turn)

    prompt_parts.append(f"User Query: {user_query}")
    prompt_parts.append("Thought:")

    return "\n".join(prompt_parts)

# 3. Large Language Model (LLM) Integration (Simulated)

def parse_tool_call(tool_call_string: str) -> tuple[str, dict] | None:
    match = re.match(r"(\w+)\\(.*)\\)", tool_call_string.strip())
    if not match:
        return None

    tool_name = match.group(1)
    args_str = match.group(2)
    args = {}

    if args_str:
        try:
            arg_pairs = re.findall(r'(\w+)\\s*=\\s*(.+?)(?:,\\s*|\\Z)', args_str)
            for key, value_str in arg_pairs:
                try:
                    args[key] = json.loads(value_str)
                except json.JSONDecodeError:
                    if value_str.startswith('"') and value_str.endswith('"'):
                        args[key] = value_str[1:-1]
                    else:
                        args[key] = value_str
        except Exception as e:
            print(f"Warning: Could not fully parse tool arguments: {e}. Raw args string: {args_str}")
            return None
    return tool_name, args

def simulate_llm_response(prompt: str) -> str:
    if "status" in prompt.lower() and "order" in prompt.lower():
        order_id_match = re.search(r"ORD\\d{3}", prompt)
        order_id = order_id_match.group(0) if order_id_match else "UNKNOWN"
        tool_call = f"check_order_status(order_id=\"{order_id}\")"
        thought = f"The user is asking for an order status, so I should use the check_order_status tool for {order_id}."
    elif "refund" in prompt.lower() and "order" in prompt.lower():
        order_id_match = re.search(r"ORD\\d{3}", prompt)
        order_id = order_id_match.group(0) if order_id_match else "UNKNOWN"
        amount_match = re.search(r"\\$(\\d+(\\.\\d{1,2})?)", prompt)
        amount = float(amount_match.group(1)) if amount_match else 0.0
        tool_call = f"process_refund(order_id=\"{order_id}\", amount={amount})"
        thought = f"The user wants to process a refund. I need to call the process_refund tool for order {order_id} with amount ${amount}."
    elif "address" in prompt.lower() and "shipping" in prompt.lower() and "order" in prompt.lower():
        order_id_match = re.search(r"ORD\\d{3}", prompt)
        order_id = order_id_match.group(0) if order_id_match else "UNKNOWN"
        address_match = re.search(r"(to|at)\\s+([\\w\\s\\d,.-]+)(?:\\?|\\.$|\\Z)", prompt, re.IGNORECASE)
        new_address = address_match.group(2).strip() if address_match else "UNKNOWN ADDRESS"
        if new_address.startswith("the new address "):
            new_address = new_address[len("the new address "):]
        new_address = new_address.replace('"', "").strip()
        tool_call = f"update_shipping_address(order_id=\"{order_id}\", new_address=\"{new_address}\")"
        thought = f"The user wants to update the shipping address. I should use the update_shipping_address tool for order {order_id}."
    else:
        return prompt + "\nAssistant: I'm not sure how to help with that. Could you please provide more details?"

    simulated_output = f"{prompt}\n{thought}\nTool Call: {tool_call}"

    print(f"Simulated LLM raw output (before tool execution):\n{simulated_output}")

    tool_name, args = parse_tool_call(tool_call)
    observation = "Error: Failed to parse tool call."
    if tool_name and tool_name in TOOL_FUNCTIONS:
        try:
            observation = TOOL_FUNCTIONS[tool_name](**args)
        except TypeError as e:
            observation = f"Error calling tool {tool_name}: {e}. Arguments: {args}"
        except Exception as e:
            observation = f"An unexpected error occurred during tool execution: {e}"
    elif tool_name:
        observation = f"Error: Tool '{tool_name}' not found."

    final_llm_response = f"{simulated_output}\nObservation: {observation}\nAssistant: "

    if "check_order_status" in tool_call and "Shipped" in observation:
        final_llm_response += f"Your order {order_id} is shipped. {observation.split('status. ')[1]}"
    elif "check_order_status" in tool_call and "Processing" in observation:
        final_llm_response += f"Your order {order_id} is currently processing."
    elif "check_order_status" in tool_call and "not found" in observation:
        final_llm_response += f"I could not find an order with ID {order_id}. Please double check the ID."
    elif "process_refund" in tool_call and "processed successfully" in observation:
        final_llm_response += observation
    elif "process_refund" in tool_call and "could not be processed" in observation:
        final_llm_response += f"I'm sorry, I was unable to process the refund. {observation}"
    elif "update_shipping_address" in tool_call and "updated to" in observation:
        final_llm_response += observation
    elif "update_shipping_address" in tool_call and "cannot be updated" in observation:
        final_llm_response += observation
    elif "update_shipping_address" in tool_call and "not found" in observation:
        final_llm_response += f"I could not find an order with ID {order_id} to update the address."
    else:
        final_llm_response += f"I have processed your request. The result was: {observation}"

    return final_llm_response

def run_chatbot():
    print("Welcome to the E-commerce Customer Support Chatbot!")
    print("Type 'exit' to end the conversation.")

    chat_history = []
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() == 'exit':
            break

        prompt = DynamicPromptConstructor(user_input, chat_history)
        print(f"\n--- PROMPT SENT TO LLM ---\n{prompt}\n--------------------------\n")

        llm_full_response = simulate_llm_response(prompt)
        print(f"\n--- LLM FULL RESPONSE ---\n{llm_full_response}\n-------------------------\n")

        assistant_answer_match = re.search(r"Assistant: (.+)", llm_full_response, re.DOTALL)
        if assistant_answer_match:
            assistant_final_answer = assistant_answer_match.group(1).strip()
            print(f"Assistant: {assistant_final_answer}")
            chat_history.append(f"User Query: {user_input}")
            chat_history.append(f"Assistant: {assistant_final_answer}")
        else:
            print("Assistant: I encountered an issue processing your request.")
            chat_history.append(f"User Query: {user_input}")
            chat_history.append(f"Assistant: I encountered an issue processing your request.")

if __name__ == "__main__":
    run_chatbot()