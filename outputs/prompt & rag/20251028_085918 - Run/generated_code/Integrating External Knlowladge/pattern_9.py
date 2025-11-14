
import os
from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain.tools import Tool
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.utilities import SerperAPIWrapper
import json

# Load environment variables
load_dotenv()

# --- 1. Simulated Databases ---
PRODUCT_DB = {
    "SKU001": {"name": "Wireless Headphones", "price": 99.99, "stock": 150, "features": "Noise-cancelling, Bluetooth 5.0, 20-hour battery life"},
    "SKU002": {"name": "Smartwatch Pro", "price": 249.99, "stock": 75, "features": "Heart rate monitor, GPS, Water-resistant"},
    "SKU003": {"name": "Portable Charger 10000mAh", "price": 29.99, "stock": 300, "features": "Fast charging, USB-C, Compact design"},
}

ORDER_DB = {
    "ORD12345": {"customer_id": "CUST001", "product_sku": "SKU001", "quantity": 1, "status": "Shipped", "tracking_id": "TRK987654", "shipping_date": "2023-10-26"},
    "ORD67890": {"customer_id": "CUST002", "product_sku": "SKU002", "quantity": 1, "status": "Processing", "tracking_id": None, "shipping_date": None},
}

# --- 2. Custom Tool Functions ---

def get_product_info(sku: str) -> str:
    """Fetches product details given a product SKU."""
    info = PRODUCT_DB.get(sku.upper())
    if info:
        return json.dumps(info)
    return f"No product found with SKU: {sku}"

def get_order_status(order_id: str) -> str:
    """Retrieves the status and details of an order given an order ID."""
    info = ORDER_DB.get(order_id.upper())
    if info:
        return json.dumps(info)
    return f"No order found with ID: {order_id}"

def create_support_ticket(customer_id: str, issue_description: str, product_sku: str = None) -> str:
    """Creates a new support ticket in the CRM system. Returns a ticket ID."""
    # In a real system, this would interact with a CRM API
    ticket_id = f"TICKET-{os.urandom(4).hex().upper()}"
    print(f"[SIMULATED ACTION] Created ticket {ticket_id} for customer {customer_id}: {issue_description}")
    return f"Support ticket created successfully. Your ticket ID is {ticket_id}. We will get back to you shortly."

def get_policy_info(policy_name: str) -> str:
    """Looks up information about a specific company policy (e.g., 'refund', 'warranty')."""
    policies = {
        "refund": "Our refund policy allows returns within 30 days of purchase for a full refund, provided the item is in its original condition.",
        "warranty": "All electronic products come with a 1-year limited warranty covering manufacturing defects.",
        "shipping": "Standard shipping takes 3-5 business days. Expedited options are available at checkout."
    }
    info = policies.get(policy_name.lower())
    if info:
        return info
    return f"No information found for policy: {policy_name}"

def recommend_product(customer_id: str, current_product_sku: str = None, interest: str = None) -> str:
    """Recommends a product based on customer history or expressed interest."""
    # This is a simplified recommendation. A real system would use more sophisticated logic.
    if interest and "headphones" in interest.lower():
        return json.dumps({"recommendation": "SKU001 - Wireless Headphones", "reason": "Based on your interest in audio products."})
    if current_product_sku == "SKU001": # If user has headphones, recommend a charger
        return json.dumps({"recommendation": "SKU003 - Portable Charger 10000mAh", "reason": "Customers often pair headphones with portable chargers."})
    return "Based on your query, I recommend checking out our 'Smartwatch Pro' (SKU002) for fitness enthusiasts."

# --- 3. Knowledge Base Ingestion (RAG System Setup) ---

# Create a dummy knowledge_base directory and files for demonstration
if not os.path.exists("knowledge_base"):
    os.makedirs("knowledge_base")
    with open("knowledge_base/faq.txt", "w") as f:
        f.write("Q: How do I reset my password? A: Go to the login page and click 'Forgot Password'.\n")
        f.write("Q: What are your customer service hours? A: We are available Monday to Friday, 9 AM to 5 PM EST.\n")
    with open("knowledge_base/troubleshooting.txt", "w") as f:
        f.write("If your device is not turning on, ensure it is fully charged and try a hard reset.\n")

print("Loading knowledge base...")
loader = DirectoryLoader("knowledge_base", glob="**/*.txt", loader_cls=TextLoader)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")
retriever = vectorstore.as_retriever()
print("Knowledge base loaded and indexed.")

# --- 4. Initialize LLM and Tools ---

# Serper API for external search
search = SerperAPIWrapper(serper_api_key=os.getenv("SERPER_API_KEY"))

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=os.getenv("OPENAI_API_KEY"))

# Define the tools for the agent
tools = [
    Tool(
        name="Product Information",
        func=get_product_info,
        description="Useful for getting details about a product given its SKU."
    ),
    Tool(
        name="Order Status",
        func=get_order_status,
        description="Useful for checking the status and details of a customer order given an order ID."
    ),
    Tool(
        name="Create Support Ticket",
        func=create_support_ticket,
        description="Useful for creating a new support ticket when a customer has an unresolved issue. Requires customer_id and issue_description."
    ),
    Tool(
        name="Policy Lookup",
        func=get_policy_info,
        description="Useful for looking up company policies like refund, warranty, or shipping."
    ),
    Tool(
        name="Product Recommendation",
        func=recommend_product,
        description="Useful for recommending products to customers based on their interest or current products."
    ),
    Tool(
        name="External Web Search",
        func=search.run,
        description="Useful for general web searches when internal knowledge is insufficient. Input should be a search query."
    )
]

# Add RAG system as a tool as well, or integrate it directly into the agent chain
# For simplicity, we'll let the agent decide to use the retriever indirectly via context, or directly if needed.
# A more advanced setup might use a custom LangChain chain or a different agent type to explicitly use RAG.
# For this zero-shot-react-description agent, we'll rely on its ability to infer when to use a tool vs. its own knowledge/retriever.

# We'll use RetrievalQA chain as a tool to leverage the RAG system effectively for direct questions to KB
kba_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

tools.append(
    Tool(
        name="Knowledge Base Assistant",
        func=kba_chain.run,
        description="Useful for answering questions based on the internal knowledge base and FAQs."
    )
)

# Initialize the agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10
)

print("\n--- Customer Support Assistant Initialized ---")
print("Type 'exit' to end the conversation.")
print("\nHow can I help you today?")

# --- 5. Interaction Loop ---
while True:
    user_input = input("\nCustomer: ")
    if user_input.lower() == 'exit':
        print("Assistant: Goodbye!")
        break

    try:
        response = agent.run(user_input)
        print(f"Assistant: {response}")
    except Exception as e:
        print(f"Assistant: An error occurred: {e}")
        print("Assistant: I apologize, I encountered an issue. Please try rephrasing your question or contact human support if the problem persists.")

