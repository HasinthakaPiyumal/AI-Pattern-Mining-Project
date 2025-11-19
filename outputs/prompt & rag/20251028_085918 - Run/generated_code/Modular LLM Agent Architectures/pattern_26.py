import os
import json
import pandas as pd
import gradio as gr
from dotenv import load_dotenv

from langchain.agents import initialize_agent, AgentType, Tool
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


load_dotenv()

# --- Simulated E-commerce Backend Data and Functions ---

# Product Data (simulated CSV)
product_data_csv = """ID,Name,Category,Price,Description,Stock
101,Laptop Pro,Electronics,1200.00,High-performance laptop with 16GB RAM,50
102,Wireless Mouse,Electronics,25.00,Ergonomic wireless mouse,200
103,Mechanical Keyboard,Electronics,75.00,RGB mechanical keyboard,150
104,Smartphone X,Electronics,800.00,Latest smartphone model,80
105,Smartwatch Lite,Electronics,150.00,Fitness tracker and smartwatch,120
201,T-Shirt Basic,Apparel,20.00,Comfortable cotton t-shirt,300
202,Jeans Slim Fit,Apparel,60.00,Stylish slim fit jeans,100
203,Running Shoes,Apparel,90.00,Lightweight running shoes,70
301,Coffee Maker,Home & Kitchen,50.00,Automatic drip coffee maker,90
302,Blender Pro,Home & Kitchen,100.00,High-speed blender,60
"""

# Load product data into a Pandas DataFrame
product_df = pd.read_csv(pd.io.common.StringIO(product_data_csv))

# Order Data (simulated JSON/dict)
order_data = {
    "ORD7890": {"user_id": "user1", "items": [{"product_id": 101, "qty": 1}, {"product_id": 102, "qty": 1}], "status": "Shipped", "total": 1225.00},
    "ORD1234": {"user_id": "user2", "items": [{"product_id": 201, "qty": 2}, {"product_id": 301, "qty": 1}], "status": "Processing", "total": 90.00},
    "ORD5678": {"user_id": "user1", "items": [{"product_id": 103, "qty": 1}], "status": "Delivered", "total": 75.00}
}

def get_product_details(query: str) -> str:
    try:
        if query.isdigit():
            product = product_df[product_df["ID"] == int(query)]
        else:
            product = product_df[product_df["Name"].str.contains(query, case=False, na=False)]
        
        if not product.empty:
            return product.iloc[0].to_json()
        return "Product not found."
    except Exception as e:
        return f"Error retrieving product details: {e}"

def check_order_status(order_id: str) -> str:
    order = order_data.get(order_id)
    if order:
        return json.dumps({"order_id": order_id, "status": order["status"], "total": order["total"]})
    return f"Order ID {order_id} not found."

def initiate_return(order_id: str, product_id: int) -> str:
    order = order_data.get(order_id)
    if order:
        for item in order["items"]:
            if item["product_id"] == product_id:
                return f"Return initiated for product {product_id} in order {order_id}. Please follow instructions sent to your email."
        return f"Product {product_id} not found in order {order_id}."
    return f"Order ID {order_id} not found."

def get_recommendations(user_id: str) -> str:
    if user_id == "user1":
        return "Based on your past purchases, we recommend: Gaming Headset (ID: 401), Smart Home Hub (ID: 402)."
    elif user_id == "user2":
        return "Based on your past purchases, we recommend: Yoga Mat (ID: 501), Healthy Snack Box (ID: 502)."
    return "Please provide a valid user ID for recommendations."

# --- Knowledge Base Module (RAG) ---

faqs_text = (
    "Q: What is your return policy? A: You can return most items within 30 days of purchase for a full refund. "
    "Some exclusions apply, such as digital goods and personalized items. "
    "Q: How do I track my order? A: Once your order is shipped, you will receive a tracking number via email. "
    "You can use this number on our \"Track Your Order\" page. "
    "Q: What payment methods do you accept? A: We accept major credit cards (Visa, MasterCard, American Express), PayPal, and Apple Pay. "
    "Q: Do you offer international shipping? A: Yes, we offer international shipping to most countries. Shipping fees and delivery times vary by destination. "
)

return_policy_text = (
    "Our return policy allows for returns of unused and unopened items within 30 days of delivery. "
    "Refunds are processed within 5-7 business days after the item is received and inspected. "
    "Original shipping fees are non-refundable. For defective items, please contact customer support immediately. "
)

knowledge_base_documents = [
    Document(page_content=faqs_text, metadata={"source": "FAQs"}),
    Document(page_content=return_policy_text, metadata={"source": "Return Policy"}),
]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(knowledge_base_documents)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(texts, embeddings, collection_name="ecommerce_kb")
retriever = vectorstore.as_retriever()

# --- LangChain Tools ---

product_catalog_tool = Tool(
    name="ProductCatalog",
    func=get_product_details,
    description="Useful for fetching details about products, like price, description, and stock. Input should be a product ID (number) or a keyword (string)."
)

order_management_tool = Tool(
    name="OrderManagement",
    func=check_order_status,
    description="Useful for checking the status of an existing order. Input should be an order ID (string, e.g., 'ORD1234')."
)

recommendation_tool = Tool(
    name="Recommendation",
    func=get_recommendations,
    description="Useful for providing personalized product recommendations to a user. Input should be a user ID (string, e.g., 'user1')."
)

knowledge_base_search_tool = Tool(
    name="KnowledgeBaseSearch",
    func=lambda query: retriever.invoke(query)[0].page_content,
    description="Useful for answering general questions about policies, FAQs, or any information present in the e-commerce knowledge base. Input should be a query string."
)

tools = [product_catalog_tool, order_management_tool, recommendation_tool, knowledge_base_search_tool]

# --- Core LLM Agent ---

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key, model_name="gpt-4-0125-preview")

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True
)

# --- Gradio User Interface ---

def chat_interface(message, history):
    try:
        response = agent_chain.invoke({"input": message})
        return response["output"]
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == "__main__":
    gr.ChatInterface(
        chat_interface,
        title="E-commerce Customer Support AI Agent",
        description="Ask me about product details, order status, returns, or general e-commerce policies!",
        examples=[
            "What are the details for Laptop Pro?",
            "What is the status of order ORD7890?",
            "How do I return an item?",
            "Recommend products for user1."
        ]
    ).launch()