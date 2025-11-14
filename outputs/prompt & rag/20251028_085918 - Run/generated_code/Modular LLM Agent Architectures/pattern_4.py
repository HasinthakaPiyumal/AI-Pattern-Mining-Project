import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- 1. Mock Data Sources ---
PRODUCT_CATALOG = {
    "SKU001": {"name": "Wireless Bluetooth Headphones", "price": 79.99, "description": "High-quality wireless headphones with noise cancellation.", "in_stock": True},
    "SKU002": {"name": "Smart Fitness Tracker", "price": 49.99, "description": "Monitor your heart rate, steps, and sleep with this sleek tracker.", "in_stock": False},
    "SKU003": {"name": "Portable Power Bank 10000mAh", "price": 29.99, "description": "Fast charging power bank for all your devices.", "in_stock": True},
}

FAQ_KNOWLEDGE_BASE = [
    "What is your return policy? Our return policy allows returns within 30 days of purchase for a full refund.",
    "How can I track my order? You can track your order using the tracking number provided in your shipping confirmation email on our website's 'Track Order' page.",
    "Do you offer international shipping? Yes, we offer international shipping to most countries. Shipping fees and delivery times vary by destination.",
]

CUSTOMER_ORDERS = {
    "ORD12345": {"customer_id": "CUST001", "status": "Shipped", "items": [{"sku": "SKU001", "qty": 1}], "tracking_number": "TRK987654321"},
    "ORD67890": {"customer_id": "CUST002", "status": "Processing", "items": [{"sku": "SKU003", "qty": 2}]},
}

# --- 2. Embedding Model and Vector Database ---
@st.cache_resource
def get_vector_store():
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Populate ChromaDB with FAQ and product descriptions
    docs = []
    for faq in FAQ_KNOWLEDGE_BASE:
        docs.append({"page_content": faq, "source": "FAQ"})
    for sku, product in PRODUCT_CATALOG.items():
        docs.append({"page_content": f"Product: {product['name']}. Description: {product['description']}. Price: ${product['price']:.2f}. SKU: {sku}. In stock: {product['in_stock']}.", "source": "Product Catalog"})
    
    # Using in-memory Chroma for simplicity in this example
    vectorstore = Chroma.from_texts(
        [doc["page_content"] for doc in docs],
        embedding=embeddings,
        metadatas=[{"source": doc["source"]} for doc in docs]
    )
    return vectorstore

vectorstore = get_vector_store()

# --- 3. Mock Tool Implementations (APIs) ---
def get_order_status(order_id: str) -> str:
    """Fetches the status of a customer order given an order ID."""
    order = CUSTOMER_ORDERS.get(order_id)
    if order:
        status = order["status"]
        tracking_info = f" Tracking number: {order['tracking_number']}." if "tracking_number" in order else ""
        return f"Order {order_id} is currently {status}.{tracking_info}"
    return f"Order {order_id} not found."

def get_product_details(sku: str) -> str:
    """Retrieves detailed information about a product using its SKU."""
    product = PRODUCT_CATALOG.get(sku.upper())
    if product:
        stock_status = "In Stock" if product["in_stock"] else "Out of Stock"
        return f"Product: {product['name']}. Price: ${product['price']:.2f}. Description: {product['description']}. Status: {stock_status}."
    return f"Product with SKU {sku} not found."

def update_customer_contact(customer_id: str, new_email: str = None, new_phone: str = None) -> str:
    """Simulates updating customer contact information. Returns a confirmation message."""
    # In a real CRM, this would update a database
    update_info = []
    if new_email: update_info.append(f"email to {new_email}")
    if new_phone: update_info.append(f"phone to {new_phone}")
    
    if update_info:
        return f"Customer {customer_id} contact information updated: {', '.join(update_info)}."
    return "No update information provided."

# --- 4. Langchain Tools ---
tools = [
    Tool(
        name="GetOrderStatus",
        func=get_order_status,
        description="Useful for finding out the current status of a customer's order. Input should be an order ID (e.g., ORD12345)."
    ),
    Tool(
        name="GetProductDetails",
        func=get_product_details,
        description="Useful for getting detailed information about a product, including its price, description, and stock status. Input should be a product SKU (e.g., SKU001)."
    ),
    Tool(
        name="UpdateCustomerContact",
        func=update_customer_contact,
        description="Useful for updating a customer's contact details, such as email or phone number. Input should be customer_id, and optionally new_email or new_phone."
    ),
    Tool(
        name="KnowledgeBaseSearch",
        func=lambda query: vectorstore.similarity_search(query, k=1)[0].page_content,
        description="Useful for answering general questions about policies, FAQs, or product information from the knowledge base. Input should be a natural language query."
    ),
]

# --- 5. Core LLM & Memory ---
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=OPENAI_API_KEY)
memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=5)

# --- 6. Structured Planning & Central Router (Langchain Agent) ---
# Custom prompt template for better agent guidance
agent_prompt_template = PromptTemplate.from_template(
    """You are a helpful and efficient customer support agent for an e-commerce platform.
    Your goal is to assist customers with their queries, provide information, and perform actions using the available tools.
    
    You have access to the following tools:
    {tools}
    
    Use the following format:
    
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    Begin!
    
    Chat History:
    {chat_history}
    
    Question: {input}
    Thought:{agent_scratchpad}"""
)

agent = create_react_agent(llm, tools, agent_prompt_template)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory, handle_parsing_errors=True)

# --- 7. Streamlit UI ---
st.set_page_config(page_title="Smart Customer Support Agent", page_icon=":robot:")
st.title("🛒 Smart Customer Support Agent")
st.markdown("Hello! I am your AI assistant. How can I help you today?")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # The agent_executor uses the memory automatically
                response = agent_executor.invoke({"input": prompt})
                st.markdown(response["output"])
                st.session_state.messages.append({"role": "assistant", "content": response["output"]})
            except Exception as e:
                error_message = f"An error occurred: {e}. Please try again or rephrase your question."
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})

