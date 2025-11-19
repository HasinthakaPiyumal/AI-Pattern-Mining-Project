
import os
import requests
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langchain.memory import ConversationBufferWindowMemory

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from transformers import pipeline

# Load environment variables
load_dotenv()

# --- 0. Knowledge Base Data (simulated knowledge_base_data.py content) ---
KNOWLEDGE_BASE_DOCS = [
    "Our return policy allows returns within 30 days of purchase with a valid receipt.",
    "To track your order, please visit our website and enter your order ID in the tracking section.",
    "Our customer service hours are Monday to Friday, 9 AM to 5 PM EST.",
    "You can reset your password by clicking 'Forgot Password' on the login page and following the instructions.",
    "For technical support, please contact us at support@example.com or call 1-800-TECH-HELP.",
    "We offer free standard shipping on all orders over $50.",
    "Our product catalog includes electronics, home goods, and apparel.",
    "Payments can be made via credit card, PayPal, or bank transfer.",
    "Warranty information for electronics is typically one year from the purchase date, covering manufacturing defects.",
    "To update your shipping address for an existing order, please contact customer support immediately.",
]

# --- 1. Knowledge Base Module Setup ---
# Initialize embeddings
# Using a small, efficient model for demonstration purposes
embeddings_model_name = "all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

# Create and persist ChromaDB
# In a real application, you'd persist this to disk and load it
vectorstore = Chroma.from_documents(
    [Document(page_content=doc) for doc in KNOWLEDGE_BASE_DOCS],
    embeddings,
    collection_name="customer_support_kb",
)

kbase_retriever = vectorstore.as_retriever()

@tool
def query_knowledge_base(query: str) -> str:
    """Queries the customer support knowledge base for relevant information. Use this for general questions about policies, procedures, or product information."""
    docs = kbase_retriever.invoke(query)
    return "\n".join([doc.page_content for doc in docs])

# --- 2. Tool-use Module (Simulated External Systems) ---
@tool
def get_customer_info(customer_id: str) -> str:
    """Simulates fetching customer information from a CRM system. Requires a customer ID."""
    # In a real scenario, this would make an API call to a CRM.
    if customer_id == "CUST123":
        return "Customer ID: CUST123, Name: Alice Smith, Email: alice@example.com, Last Purchase: 2023-10-26"
    elif customer_id == "CUST456":
        return "Customer ID: CUST456, Name: Bob Johnson, Email: bob@example.com, Last Purchase: 2023-11-15"
    else:
        return f"Customer with ID {customer_id} not found."

@tool
def create_ticket(customer_id: str, issue: str) -> str:
    """Simulates creating a support ticket in a ticketing system. Requires a customer ID and a description of the issue."""
    # In a real scenario, this would make an API call to a ticketing system.
    ticket_id = f"TICKET-{os.urandom(4).hex().upper()}"
    return f"Successfully created support ticket {ticket_id} for customer {customer_id} regarding: {issue}."

@tool
def get_order_status(order_id: str) -> str:
    """Simulates fetching the status of an order from an external order tracking API. Requires an order ID."""
    # In a real scenario, this would make an actual HTTP request using `requests`.
    if order_id == "ORD789":
        return "Order ORD789 status: Shipped, Tracking: TRK987654321, Estimated Delivery: 2023-12-05"
    elif order_id == "ORD101":
        return "Order ORD101 status: Processing, Estimated Delivery: 2023-12-10"
    else:
        return f"Order with ID {order_id} not found."

# --- 3. Sentiment Analysis Module ---
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

@tool
def analyze_sentiment(text: str) -> str:
    """Analyzes the sentiment of a given text (e.g., a customer's message). Returns 'POSITIVE', 'NEGATIVE', or 'NEUTRAL'."""
    result = sentiment_pipeline(text)[0]
    # The model typically returns 'POSITIVE' or 'NEGATIVE'. We can add a threshold for 'NEUTRAL'
    # For simplicity, we'll just return the label directly.
    return result['label']


# --- 4. Core Agent & Orchestration ---
# Initialize LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)

# Define the tools available to the agent
tools = [
    query_knowledge_base,
    get_customer_info,
    create_ticket,
    get_order_status,
    analyze_sentiment,
]

# Define the agent prompt
# The system message guides the agent's behavior and priority
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an intelligent customer support agent. Your goal is to assist customers efficiently by answering questions, providing information, and performing actions using the available tools. Always be polite and helpful. If a customer expresses frustration, analyze their sentiment and consider escalating or apologizing."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Initialize memory for conversation history
memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=5)

# Create the agent
agent = create_openai_tools_agent(llm, tools, prompt)

# Create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True, handle_parsing_errors=True)

# --- 5. Application Flow: Interaction Loop ---
def run_customer_support_agent():
    print("Hello! I'm your AI Customer Support Agent. How can I help you today? (Type 'exit' to quit)")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            print("Thank you for contacting us. Goodbye!")
            break
        try:
            response = agent_executor.invoke({"input": user_input})
            print(f"Agent: {response['output']}")
        except Exception as e:
            print(f"Agent Error: An unexpected error occurred: {e}")
            print("Please try rephrasing your request or contact a human agent if the issue persists.")

if __name__ == "__main__":
    # Ensure OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set. Agent might not function correctly.")
        print("Please set it in your .env file or environment variables.")

    print("Initializing agent components...")
    run_customer_support_agent()
