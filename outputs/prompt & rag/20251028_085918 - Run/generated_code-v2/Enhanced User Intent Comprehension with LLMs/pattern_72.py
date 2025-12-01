import gradio as gr
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory

# Mock E-commerce Backend Simulator
class MockEcommerceBackend:
    def __init__(self):
        self.orders = {
            "12345": {"status": "Shipped", "item": "Laptop", "address": "123 Main St"},
            "67890": {"status": "Processing", "item": "Headphones", "address": "456 Oak Ave"},
        }
        self.users = {
            "user1": {"address": "789 Pine Rd", "email": "user1@example.com"}
        }

    def track_order(self, order_id: str) -> str:
        order_info = self.orders.get(order_id)
        if order_info:
            return f"Order {order_id} status: {order_info['status']}. Item: {order_info['item']}."
        return f"Order {order_id} not found. Please check your order ID."

    def request_refund(self, order_id: str) -> str:
        if order_id in self.orders:
            self.orders[order_id]["status"] = "Refund Requested"
            return f"Refund requested for order {order_id}. It will be processed shortly."
        return f"Order {order_id} not found. Unable to process refund."

    def update_address(self, user_id: str, new_address: str) -> str:
        if user_id in self.users:
            self.users[user_id]["address"] = new_address
            return f"Address for user {user_id} updated to {new_address}."
        return f"User {user_id} not found. Unable to update address."

# Initialize Mock Backend
ecommerce_backend = MockEcommerceBackend()

# Initialize LangChain LLM (using a placeholder, replace with actual LLM like OpenAI or a local model)
# For local development without an OpenAI key, you might use a local LLM or mock it.
# For this example, we'll use a dummy LLM and rely heavily on tool descriptions and prompts.
# In a real scenario, you would configure an actual LLM here.

# Due to the constraint of not using external API keys directly in the generated code for a basic example,
# and to demonstrate the architecture, we'll use a simplified approach.
# For a full functional example, replace with:
# from langchain.llms import OpenAI
# llm = OpenAI(temperature=0, openai_api_key="YOUR_API_KEY")

# --- Simplified NLU and LLM Simulation --- 
# We'll use a basic placeholder for LLM that primarily relies on tool descriptions and a simple prompt structure
# to simulate intent understanding and tool selection within LangChain's agent framework.

class SimpleMockLLM:
    def __init__(self):
        pass

    def __call__(self, prompt: str, stop=None) -> str:
        # This is a very basic simulation. In a real app, this would be an actual LLM call.
        # The LangChain agent handles the prompt engineering for tool selection.
        # We're relying on the agent's ability to interpret prompts and use tools.
        return "" # The agent's logic will use the tools based on its internal reasoning

llm = SimpleMockLLM()

# Define Tools for the LangChain Agent
tools = [
    Tool(
        name="TrackOrder",
        func=ecommerce_backend.track_order,
        description="Useful for tracking the status of an order. Input should be an order ID (e.g., '12345')."
    ),
    Tool(
        name="RequestRefund",
        func=ecommerce_backend.request_refund,
        description="Useful for initiating a refund for an order. Input should be an order ID (e.g., '67890')."
    ),
    Tool(
        name="UpdateAddress",
        func=ecommerce_backend.update_address,
        description="Useful for updating a user's shipping address. Input should be a comma-separated string of user ID and new address (e.g., 'user1,100 New St')."
    )
]

# Initialize LangChain Agent with Conversation Buffer Memory
memory = ConversationBufferMemory(memory_key="chat_history")

agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True # To gracefully handle cases where the LLM output is not perfectly parsable
)

def chatbot_response(message, history):
    try:
        response = agent_chain.run(input=message)
        return response
    except Exception as e:
        return f"An error occurred: {str(e)}. Please try again or rephrase your request."

# Gradio User Interface
if __name__ == "__main__":
    gr.ChatInterface(
        chatbot_response,
        title="E-commerce Support Chatbot",
        description="Ask me about your orders, refunds, or update your address."
    ).launch()
