import streamlit as st
import os
from dotenv import load_dotenv
from langchain.agents import initialize_agent, Tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentType

# Load environment variables from .env file
load_dotenv()

# --- Simulated External Tools ---

def get_product_info(product_name: str) -> str:
    """Fetches details for a specific product from the product catalog."""
    product_catalog = {
        "Laptop": {"price": "$1200", "description": "Powerful laptop with 16GB RAM and 512GB SSD.", "availability": "In Stock"},
        "Smartphone": {"price": "$800", "description": "Latest model smartphone with a high-resolution camera.", "availability": "Low Stock"},
        "Headphones": {"price": "$150", "description": "Noise-cancelling over-ear headphones.", "availability": "In Stock"},
        "Keyboard": {"price": "$75", "description": "Mechanical gaming keyboard with RGB lighting.", "availability": "Out of Stock"},
    }
    info = product_catalog.get(product_name)
    if info:
        return f"Product: {product_name}, Price: {info['price']}, Description: {info['description']}, Availability: {info['availability']}"
    return f"Sorry, product '{product_name}' not found in our catalog."

def get_order_status(order_id: str) -> str:
    """Retrieves the current status of a customer order."""
    order_database = {
        "ORD123": {"status": "Shipped", "eta": "2 days", "items": "Laptop"},
        "ORD456": {"status": "Processing", "eta": "5 days", "items": "Smartphone"},
        "ORD789": {"status": "Delivered", "eta": "N/A", "items": "Headphones"},
    }
    status = order_database.get(order_id)
    if status:
        return f"Order ID: {order_id}, Status: {status['status']}, Estimated Delivery: {status['eta']}, Items: {status['items']}"
    return f"Sorry, order '{order_id}' not found. Please double-check the ID."

def search_knowledge_base(query: str) -> str:
    """Searches the knowledge base for answers to common questions."""
    faqs = {
        "return policy": "Our return policy allows returns within 30 days of purchase with a valid receipt.",
        "shipping cost": "Standard shipping is $5.99. Free shipping for orders over $50.",
        "payment methods": "We accept Visa, Mastercard, American Express, and PayPal.",
        "warranty": "All electronics come with a 1-year manufacturer's warranty.",
    }
    for key, value in faqs.items():
        if key in query.lower():
            return value
    return f"I couldn't find an answer for '{query}' in our knowledge base. Please try rephrasing or contact live support."

# --- Langchain Setup ---

# Initialize the LLM
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=os.getenv("OPENAI_API_KEY"))

# Define the tools the LLM agent can use
tools = [
    Tool(
        name="Product Catalog",
        func=get_product_info,
        description="Useful for when you need to get information about a product, like its price, description, or availability. Input should be the product name."
    ),
    Tool(
        name="Order Management System",
        func=get_order_status,
        description="Useful for when you need to find the status of a customer's order. Input should be the order ID."
    ),
    Tool(
        name="Knowledge Base",
        func=search_knowledge_base,
        description="Useful for when you need to answer common questions or provide troubleshooting information. Input should be a relevant query."
    ),
]

# Initialize the agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True, # Set to True to see the agent's thought process
    handle_parsing_errors=True
)

# --- Streamlit UI ---

st.set_page_config(page_title="Smart Customer Support Assistant", layout="centered")
st.title("🛒 Smart Customer Support Assistant")
st.markdown("Hello! How can I help you today? Ask me about products, orders, or common questions.")

# Input for customer query
user_query = st.text_input("Your question:", "What is the price of the Laptop?")

if st.button("Get Assistance"):
    if user_query:
        with st.spinner("Thinking..."):
            try:
                response = agent.run(user_query)
                st.success("Response from Assistant:")
                st.info(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.error("Please ensure your OpenAI API key is correctly set in the .env file.")
    else:
        st.warning("Please enter a question.")

st.markdown("___")
st.markdown("**Examples:**")
st.markdown("- `What is the price of the Smartphone?`")
st.markdown("- `Where is my order ORD123?`")
st.markdown("- `What is your return policy?`")
st.markdown("- `Tell me about the Keyboard.`")
