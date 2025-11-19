import os
from typing import Literal

from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent

# --- 1. Simulate External Tools ---
# In a real-world scenario, these would be actual API calls to your e-commerce systems.

class OrderStatusInput(BaseModel):
    order_id: str = Field(..., description="The unique identifier for the customer's order.")

@tool(args_schema=OrderStatusInput)
def get_order_status(order_id: str) -> str:
    """Fetches the current status of an order given its order ID."""
    print(f"[TOOL CALL] Getting status for order: {order_id}")
    if order_id == "ORDER123":
        return "Your order ORDER123 has been shipped and is expected to arrive on 2024-07-25."
    elif order_id == "ORDER456":
        return "Your order ORDER456 is currently being processed and will be shipped soon."
    else:
        return f"Order {order_id} not found or invalid."

class ProductInfoInput(BaseModel):
    product_name: str = Field(..., description="The name or keyword of the product to search for.")

@tool(args_schema=ProductInfoInput)
def get_product_info(product_name: str) -> str:
    """Retrieves detailed information about a product, including description, price, and stock."""
    print(f"[TOOL CALL] Getting info for product: {product_name}")
    if "laptop" in product_name.lower():
        return "The 'ProBook X' laptop features an i7 processor, 16GB RAM, and a 512GB SSD. Price: $1200. In stock."
    elif "headphone" in product_name.lower():
        return "The 'SoundBlast Pro' headphones offer noise cancellation and 24-hour battery. Price: $199. Limited stock."
    else:
        return f"No detailed information found for product: {product_name}."

class ProcessReturnInput(BaseModel):
    order_id: str = Field(..., description="The unique identifier of the order to process a return for.")
    reason: str = Field(..., description="The reason for the return (e.g., 'damaged', 'wrong item', 'not satisfied').")

@tool(args_schema=ProcessReturnInput)
def process_return(order_id: str, reason: str) -> str:
    """Initiates a return process for a given order ID and reason."""
    print(f"[TOOL CALL] Processing return for order: {order_id}, reason: {reason}")
    if order_id == "ORDER123":
        return "Return request for ORDER123 with reason 'damaged' has been initiated. A return label will be sent to your email within 24 hours."
    else:
        return f"Return processing failed for order {order_id}. Please ensure the order is valid and eligible for return."

# --- 2. LLM and Agent Setup ---

# Set your OpenAI API key as an environment variable or pass it directly
# os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

llm = ChatOpenAI(model="gpt-4o", temperature=0)

tools = [get_order_status, get_product_info, process_return]

# Define the agent's prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an advanced AI Customer Support Agent for an e-commerce platform. Your goal is to assist customers efficiently, transparently, and trustworthily. Always try to use the provided tools to answer questions. After using a tool, explain your reasoning and state your confidence level in the answer. If you cannot find an answer using the tools, state that you don't have enough information."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# Create the LangChain agent
agent = create_openai_tools_agent(llm, tools, prompt)

# Create the AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- 3. Transparency & Trustworthiness (Conceptual Implementation) ---
# This is a simplified approach. In a real system, confidence scoring
# and progressive disclosure would involve more sophisticated logic,
# potentially using additional LLM calls or specific model outputs.

def format_agent_response(response: dict) -> dict:
    """Formats the agent's response to include conceptual reasoning and confidence."""
    output = response.get("output", "")
    intermediate_steps = response.get("intermediate_steps", [])

    reasoning = ""
    confidence_score = "N/A"

    if intermediate_steps:
        # Simple heuristic: If a tool was used, the confidence might be higher
        # and the reasoning comes from the tool's output and LLM's thought.
        reasoning = "Based on information retrieved from our internal systems."
        confidence_score = "High" # Conceptual
        for step in intermediate_steps:
            # LangChain's intermediate steps include agent's thought and tool output
            if hasattr(step, 'log') and 'tool_code' in step.log:
                reasoning += f"\n- Agent thought: {step.log.split('tool_code')[0].strip()}"
            reasoning += f"\n- Tool output: {step.observation}"
    else:
        # If no tool was used, it's direct LLM generation or an unknown query
        reasoning = "Based on general knowledge or direct LLM inference."
        confidence_score = "Medium" # Conceptual

    # Progressive disclosure concept (simplified)
    # We return a summary and then full details.
    summary = output.split('.')[0] + "."

    return {
        "summary": summary,
        "full_response": output,
        "reasoning_path": reasoning,
        "confidence_score": confidence_score,
        "trust_level_explanation": f"The system's confidence is {confidence_score} {'' if confidence_score == 'N/A' else 'because ' + reasoning.split('because')[-1].split('.')[0] + '.'}"
    }

# --- 4. Quality Control (Conceptual Discussion) ---
# A full quality control module would involve:
# - LLM-generated guidelines for response quality (e.g., clarity, accuracy, completeness).
# - Structured scoring techniques (e.g., using another LLM to score responses against guidelines).
# - A feedback loop for continuous improvement and prompt refinement.
# - Anomaly detection for hallucination or prompt injection attempts.

# For demonstration, we'll just acknowledge its importance here.

# --- Main Interaction Loop (Example Usage) ---
if __name__ == "__main__":
    print("\n--- Advanced AI Customer Support Agent ---\n")
    print("Ask me about order status, product info, or returns. Type 'exit' to quit.\n")

    while True:
        user_query = input("You: ")
        if user_query.lower() == 'exit':
            break

        try:
            raw_agent_response = agent_executor.invoke({"input": user_query})
            formatted_response = format_agent_response(raw_agent_response)

            print("\nAgent (Summary):", formatted_response["summary"])
            print("Agent (Full Response):", formatted_response["full_response"])
            print("Reasoning Path:", formatted_response["reasoning_path"])
            print("Confidence Score:", formatted_response["confidence_score"])
            print("Trust Level Explanation:", formatted_response["trust_level_explanation"])
            print("\n" + "-"*50 + "\n")

        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please try again or rephrase your query.")
            print("\n" + "-"*50 + "\n")

    print("Goodbye!")