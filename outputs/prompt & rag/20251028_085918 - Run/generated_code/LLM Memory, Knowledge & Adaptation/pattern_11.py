
import os
import sqlite3
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel

from fastapi import FastAPI, HTTPException
import uvicorn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- 1. Environment Variables ---
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not found in environment variables. Please set it.")
    # In a real application, you might exit or raise an error here.

# --- 2. Logger Configuration ---
logger.add(
    "file_{time}.log",
    rotation="500 MB",
    level="INFO",
    colorize=True,
    format="{time} {level} {message}"
)
logger.info("Logger configured.")

# --- 3. Pydantic Models ---
class ChatRequest(BaseModel):
    session_id: str
    query: str

class ChatResponse(BaseModel):
    session_id: str
    response: str
    query_type: str
    sources: List[str] = []

class Product(BaseModel):
    product_id: str
    name: str
    description: str
    price: float
    category: str
    stock: int

class Order(BaseModel):
    order_id: str
    user_id: str
    products: List[Dict[str, Any]]  # [{'product_id': 'P001', 'quantity': 1}]
    status: str
    order_date: str

# --- 4. SQLite Database Setup and Helper Functions ---
DATABASE_FILE = "ecommerce_data.db"

def init_db():
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()

    # Products Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS products (
            product_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            price REAL NOT NULL,
            category TEXT,
            stock INTEGER
        )
    """)

    # Orders Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            order_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            products TEXT NOT NULL, -- Stored as JSON string
            status TEXT NOT NULL,
            order_date TEXT NOT NULL
        )
    """)

    # Sessions Table (for chat history tracking)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_sessions (
            session_id TEXT PRIMARY KEY,
            user_id TEXT, -- Can be linked to actual users later
            created_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()
    logger.info("SQLite database initialized.")

def seed_db():
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()

    # Seed Products
    products_data = [
        ("P001", "Laptop Pro X", "Powerful laptop for professionals", 1200.00, "Electronics", 50),
        ("P002", "Mechanical Keyboard", "Clicky keys for an enjoyable typing experience", 80.00, "Accessories", 150),
        ("P003", "Wireless Mouse", "Ergonomic design with long battery life", 35.00, "Accessories", 200),
        ("P004", "4K Monitor", "Stunning visuals for work and play", 350.00, "Electronics", 75),
        ("P005", "Gaming Headset", "Immersive audio with noise cancellation", 110.00, "Accessories", 90),
        ("P006", "Smartphone Model A", "Latest flagship smartphone with advanced camera features", 899.00, "Electronics", 120),
        ("P007", "Smartwatch Lite", "Track your fitness and receive notifications", 150.00, "Wearables", 180),
        ("P008", "External SSD 1TB", "Fast and portable storage solution", 120.00, "Storage", 100),
        ("P009", "Bluetooth Speaker", "Portable speaker with rich sound", 60.00, "Audio", 250),
        ("P010", "E-book Reader", "Paper-like display for comfortable reading", 120.00, "Books & Media", 60),
    ]
    cursor.executemany(
        "INSERT OR IGNORE INTO products VALUES (?, ?, ?, ?, ?, ?)", products_data
    )

    # Seed Orders
    orders_data = [
        ("ORD001", "user123", json.dumps([{"product_id": "P001", "quantity": 1}, {"product_id": "P002", "quantity": 1}]), "delivered", "2023-01-15"),
        ("ORD002", "user456", json.dumps([{"product_id": "P003", "quantity": 2}]), "shipped", "2023-02-01"),
        ("ORD003", "user123", json.dumps([{"product_id": "P004", "quantity": 1}]), "processing", "2023-03-10"),
        ("ORD004", "user789", json.dumps([{"product_id": "P005", "quantity": 1}]), "pending", "2023-04-05"),
    ]
    cursor.executemany(
        "INSERT OR IGNORE INTO orders VALUES (?, ?, ?, ?, ?)", orders_data
    )

    conn.commit()
    conn.close()
    logger.info("SQLite database seeded with dummy data.")


def get_product_from_db(product_id: str) -> Optional[Product]:
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM products WHERE product_id = ?", (product_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return Product(
            product_id=row[0], name=row[1], description=row[2], price=row[3],
            category=row[4], stock=row[5]
        )
    return None

def get_order_from_db(order_id: str) -> Optional[Order]:
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return Order(
            order_id=row[0], user_id=row[1], products=json.loads(row[2]),
            status=row[3], order_date=row[4]
        )
    return None

def get_user_orders_from_db(user_id: str) -> List[Order]:
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM orders WHERE user_id = ?", (user_id,))
    rows = cursor.fetchall()
    conn.close()
    return [
        Order(
            order_id=row[0], user_id=row[1], products=json.loads(row[2]),
            status=row[3], order_date=row[4]
        ) for row in rows
    ]

def save_chat_session(session_id: str, user_id: Optional[str] = None):
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO chat_sessions (session_id, user_id, created_at) VALUES (?, ?, ?)",
            (session_id, user_id, datetime.now().isoformat())
        )
        conn.commit()
        logger.info(f"New chat session '{session_id}' created.")
    except sqlite3.IntegrityError:
        logger.warning(f"Session ID '{session_id}' already exists. Skipping insertion.")
    finally:
        conn.close()

# --- 5. Simulated E-commerce API Client ---
class EcommerceAPI:
    def get_product_details(self, product_id: str) -> str:
        logger.info(f"Simulating API call: get_product_details for {product_id}")
        product = get_product_from_db(product_id)
        if product:
            return f"Product Name: {product.name}, Description: {product.description}, Price: ${product.price:.2f}, Stock: {product.stock}, Category: {product.category}"
        return f"Product with ID {product_id} not found."

    def check_order_status(self, order_id: str) -> str:
        logger.info(f"Simulating API call: check_order_status for {order_id}")
        order = get_order_from_db(order_id)
        if order:
            product_details = ", ".join(
                [f"P:{item['product_id']}(Qty:{item['quantity']})" for item in order.products]
            )
            return f"Order ID: {order.order_id}, User ID: {order.user_id}, Products: [{product_details}], Status: {order.status}, Order Date: {order.order_date}"
        return f"Order with ID {order_id} not found."

    def get_user_purchase_history(self, user_id: str) -> str:
        logger.info(f"Simulating API call: get_user_purchase_history for {user_id}")
        orders = get_user_orders_from_db(user_id)
        if orders:
            history = []
            for order in orders:
                product_details = ", ".join(
                    [f"P:{item['product_id']}(Qty:{item['quantity']})" for item in order.products]
                )
                history.append(
                    f"Order ID: {order.order_id}, Products: [{product_details}], Status: {order.status}, Date: {order.order_date}"
                )
            return f"Purchase history for user {user_id}:\n" + "\n".join(history)
        return f"No purchase history found for user {user_id}."

ecommerce_api = EcommerceAPI()

# --- 6. Query Classifier Training and Prediction ---
class QueryClassifier:
    def __init__(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', LogisticRegression(random_state=42))
        ])
        self.labels = []
        logger.info("QueryClassifier initialized.")

    def train(self, queries: List[str], labels: List[str]):
        self.labels = sorted(list(set(labels)))
        X_train, X_test, y_train, y_test = train_test_split(
            queries, labels, test_size=0.2, random_state=42, stratify=labels
        ) # Added stratify to handle potential imbalance
        self.pipeline.fit(X_train, y_train)
        accuracy = self.pipeline.score(X_test, y_test)
        logger.info(f"Query Classifier trained with accuracy: {accuracy:.2f}")

    def predict(self, query: str) -> str:
        if not self.labels:
            logger.warning("Query Classifier not trained, returning 'general'.")
            return "general"
        prediction = self.pipeline.predict([query])[0]
        logger.info(f"Query '{query}' classified as: {prediction}")
        return prediction

# Dummy data for training the classifier
dummy_queries = [
    "What is the price of Laptop Pro X?", "Tell me about P001.", "Details of Laptop Pro X",
    "How much does the mechanical keyboard cost?", "Information on product P002",
    "What is my order status for ORD001?", "Where is my package ORD002?",
    "Check status of order ORD003", "Has order ORD004 shipped?",
    "Show my purchases for user123", "What have I bought before as user456?",
    "Can I get a refund?", "How to return an item?",
    "My account is locked", "I can't log in.",
    "General greeting", "Hello", "How are you?",
    "What products do you have?", "Tell me about your electronics selection.",
]
dummy_labels = [
    "product_lookup", "product_lookup", "product_lookup",
    "product_lookup", "product_lookup",
    "order_status", "order_status",
    "order_status", "order_status",
    "purchase_history", "purchase_history",
    "returns_refunds", "returns_refunds",
    "account_issues", "account_issues",
    "greeting", "greeting", "greeting",
    "product_browsing", "product_browsing",
]

query_classifier = QueryClassifier()
query_classifier.train(dummy_queries, dummy_labels)

# --- 7. ChromaDB and Embedding Setup ---
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Dummy FAQ and knowledge base for RAG
dummy_kb_docs = [
    "Our return policy allows for returns within 30 days of purchase with a valid receipt.",
    "To initiate a return, please visit our 'Returns & Refunds' page and follow the instructions.",
    "We offer free shipping on all orders over $50.",
    "Standard shipping usually takes 5-7 business days.",
    "You can track your order using the tracking number provided in your shipping confirmation email.",
    "Our customer support is available 24/7 via chat, email, and phone.",
    "For technical support, please contact our specialized tech team.",
    "We accept major credit cards, PayPal, and various other payment methods.",
    "Your account can be reset by clicking 'Forgot Password' on the login page.",
    "We take data privacy seriously; your information is protected with advanced encryption.",
    "Our loyalty program offers exclusive discounts and early access to sales for members."
]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
kb_docs = text_splitter.create_documents(dummy_kb_docs)

vectorstore = Chroma.from_documents(documents=kb_docs, embedding=embeddings, persist_directory="./chroma_db")
vectorstore.persist()
logger.info("ChromaDB initialized and populated with dummy knowledge base.")

# --- 8. LangChain Tool Definitions ---
langchain_tools = [
    Tool(
        name="GetProductDetails",
        func=ecommerce_api.get_product_details,
        description="Useful for when you need to get detailed information about a product using its product ID. Input should be a product ID like P001."
    ),
    Tool(
        name="CheckOrderStatus",
        func=ecommerce_api.check_order_status,
        description="Useful for when you need to check the status of a customer's order using the order ID. Input should be an order ID like ORD001."
    ),
    Tool(
        name="GetUserPurchaseHistory",
        func=ecommerce_api.get_user_purchase_history,
        description="Useful for when you need to retrieve a customer's full purchase history using their user ID. Input should be a user ID like user123."
    ),
]

# --- 9. LangChain Agent Initialization ---
llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

# Dictionary to hold memory for each session
session_memories: Dict[str, ConversationBufferWindowMemory] = {}

def get_agent_for_session(session_id: str):
    if session_id not in session_memories:
        session_memories[session_id] = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=5)
        save_chat_session(session_id) # Save new session to DB
        logger.info(f"Initialized new memory for session: {session_id}")

    # The agent is re-initialized for each request to ensure it uses the latest memory
    # and potentially dynamically selected tools/retrievers based on query classification.
    # For this pattern, the tools are static, but RAG retriever can be dynamic.
    agent = initialize_agent(
        langchain_tools,
        llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=session_memories[session_id],
        handle_parsing_errors=True
    )
    logger.info(f"LangChain agent initialized for session: {session_id}")
    return agent

# --- 10. FastAPI Application ---
app = FastAPI(
    title="Adaptive LLM Customer Support Agent",
    description="Backend API for an intelligent customer support agent leveraging LLM augmentation."
)

@app.on_event("startup")
def startup_event():
    init_db()
    seed_db()
    logger.info("FastAPI application startup: Database initialized and seeded.")

@app.post("/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    logger.info(f"Received chat request for session {request.session_id}: {request.query}")
    
    # Get or create memory for the session
    session_memory = session_memories.get(request.session_id)
    if not session_memory:
        # If memory not found (e.g., server restart or first message),
        # create new memory and save session.
        session_memories[request.session_id] = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=5)
        save_chat_session(request.session_id)
        logger.info(f"Created new memory for previously unknown session: {request.session_id}")

    # 1. Query Classification
    query_type = query_classifier.predict(request.query)
    logger.info(f"Query classified as: {query_type}")

    response_text = "I am sorry, I could not process your request at this time."
    sources_used = []

    # 2. Adaptive LLM Processing Strategy based on Query Type
    if query_type in ["product_lookup", "order_status", "purchase_history"]:
        # Use the LangChain agent with tools for transactional queries
        agent = get_agent_for_session(request.session_id)
        try:
            result = agent.run(request.query)
            response_text = result
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            response_text = f"An error occurred while trying to fulfill your request: {e}"
    elif query_type in ["returns_refunds", "account_issues", "general", "product_browsing"]:
        # Use RAG for informational queries or direct LLM for general ones
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        retrieved_docs = retriever.get_relevant_documents(request.query)
        sources_used = [doc.metadata.get("source", "Knowledge Base") for doc in retrieved_docs]

        context_docs = "\n".join([doc.page_content for doc in retrieved_docs])
        
        # Use a simple LLM chain for general queries with retrieved context
        full_query = f"User: {request.query}\n\nContext from Knowledge Base:\n{context_docs}\n\nAssistant:"

        # If memory is used, this part needs to be integrated carefully.
        # For simplicity, if RAG, we're doing a more direct LLM call augmented with RAG.
        # A more advanced RAG agent would integrate memory and RAG within the agent.
        # For now, if classified as RAG-suitable, we use RAG; otherwise, agent.
        try:
            # Simulate a conversational chain with RAG context if applicable
            # We'll re-use the conversation chain logic but manually inject context
            # For a more robust solution, a custom LangChain agent that uses RAG based on query_type is better.
            
            # Get existing chat history for the session
            chat_history_messages = session_memories[request.session_id].buffer_as_messages
            
            # Convert messages to string format for direct LLM if needed, or pass as messages
            formatted_history = "\n".join([
                f"Human: {msg.content}" if msg.type == "human" else f"AI: {msg.content}"
                for msg in chat_history_messages
            ])
            
            prompt_with_context_and_history = f"""
You are an e-commerce customer support assistant. Answer the user's question based on the provided context and conversation history.

Conversation History:
{formatted_history}

Context:
{context_docs}

User Question: {request.query}

Assistant:"""

            # Direct LLM call with context and (simulated) history
            raw_llm_response = llm(prompt_with_context_and_history)
            response_text = raw_llm_response.strip()
            
            # Update memory with the current turn after LLM response
            session_memories[request.session_id].save_context(
                {"input": request.query},
                {"output": response_text}
            )

        except Exception as e:
            logger.error(f"LLM (RAG) execution failed: {e}")
            response_text = f"An error occurred while processing your informational request: {e}"
    else:
        # Fallback for unclassified or unrecognized query types
        response_text = "I am still learning to understand all types of queries. Can you please rephrase or ask something else?"
        logger.warning(f"Query type '{query_type}' not handled by specific strategy.")

    logger.info(f"Response for session {request.session_id}: {response_text}")
    return ChatResponse(session_id=request.session_id, response=response_text, query_type=query_type, sources=sources_used)


# --- 11. Streamlit Frontend (to be run separately) ---
# To run the Streamlit app, save this code as `app.py` or similar,
# then run `streamlit run app.py` in your terminal.
# Ensure the FastAPI server is running on http://127.0.0.1:8000

streamlit_code = """
import streamlit as st
import requests
import uuid
import json
import os

# Ensure the FastAPI backend is running on this address
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Adaptive LLM Customer Support")
st.title("🛒 Adaptive E-commerce Support Agent")
st.markdown("Hello! I am your intelligent customer support agent. How can I assist you today?")

# Initialize session state for chat history and session ID
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.initial_message_sent = False
    st.session_state.feedback_messages = [] # For debugging/feedback

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            st.caption(f"_Sources: {', '.join(message['sources'])}_ ")
        if "query_type" in message and message["query_type"]:
            st.caption(f"_Query Type: {message['query_type']}_ ")

# React to user input
if prompt := st.chat_input("Ask me anything..."):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        # Send query to FastAPI backend
        response = requests.post(
            f"{FASTAPI_URL}/chat",
            json={"session_id": st.session_state.session_id, "query": prompt}
        )
        response.raise_for_status() # Raise an exception for HTTP errors
        agent_response = response.json()

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(agent_response["response"])
            if agent_response["sources"]:
                st.caption(f"_Sources: {', '.join(agent_response['sources'])}_ ")
            if agent_response["query_type"]:
                st.caption(f"_Query Type: {agent_response['query_type']}_ ")

        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": agent_response["response"],
            "sources": agent_response["sources"],
            "query_type": agent_response["query_type"]
        })

    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the backend API. Please ensure the FastAPI server is running.")
        st.session_state.feedback_messages.append("Connection Error: FastAPI server not reachable.")
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred while communicating with the backend: {e}")
        st.session_state.feedback_messages.append(f"Request Error: {e}")
    except json.JSONDecodeError:
        st.error("Received an invalid response from the backend.")
        st.session_state.feedback_messages.append("JSON Decode Error: Invalid response from FastAPI.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.session_state.feedback_messages.append(f"Unexpected Error: {e}")

# Optional: Display feedback messages for debugging
# if st.session_state.feedback_messages:
#     st.sidebar.subheader("Debug / Feedback")
#     for msg in st.session_state.feedback_messages:
#         st.sidebar.markdown(f"- {msg}")


"""

if __name__ == "__main__":
    logger.info("Starting application. Instructions for running:")
    logger.info("1. To run the FastAPI backend: `python customer_support_agent.py` (then navigate to http://127.0.0.1:8000/docs for API docs)")
    logger.info("2. To run the Streamlit frontend: Create a separate file (e.g., `streamlit_app.py`), paste the `streamlit_code` above into it, and then run `streamlit run streamlit_app.py`")
    logger.info("Ensure OPENAI_API_KEY is set in your environment variables or a .env file.")

    # Run FastAPI application directly if this script is executed
    # This is primarily for ease of demonstration; in production, you'd use a WSGI server like Gunicorn.
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        logger.error(f"Error starting FastAPI server: {e}")


