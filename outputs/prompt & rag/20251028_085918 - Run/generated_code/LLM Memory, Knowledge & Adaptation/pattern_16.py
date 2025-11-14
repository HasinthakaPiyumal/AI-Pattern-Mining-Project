
import os
import json
import sqlite3
from flask import Flask, request, jsonify
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_models import ChatOpenAI # Using ChatOpenAI as an example, replace with desired LLM
from dotenv import load_dotenv

# Load environment variables for API keys
load_dotenv()

app = Flask(__name__)

# --- Configuration ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_DB_PATH = "./chroma_db"
SQLITE_DB_PATH = "./customer_data.db"

# --- Global Components (initialized once) ---
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# Initialize LLM - Using ChatOpenAI as a placeholder. You'll need an OPENAI_API_KEY in your .env file.
# For local or open-source LLMs, you would replace this with appropriate `llm` initialization.
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0) # Or use ChatGoogleGenerativeAI, LlamaCpp, etc.

# --- SQLite Database Setup for User Data ---
def init_sqlite_db():
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""CREATE TABLE IF NOT EXISTS customers (
                        id INTEGER PRIMARY KEY,
                        name TEXT NOT NULL,
                        email TEXT UNIQUE NOT NULL
                    )""")
    cursor.execute("""CREATE TABLE IF NOT EXISTS orders (
                        order_id TEXT PRIMARY KEY,
                        customer_id INTEGER,
                        product_name TEXT NOT NULL,
                        status TEXT NOT NULL,
                        FOREIGN KEY (customer_id) REFERENCES customers(id)
                    )""")
    
    # Insert dummy data if tables are empty
    cursor.execute("INSERT OR IGNORE INTO customers (id, name, email) VALUES (?, ?, ?)", (1, "Alice Smith", "alice@example.com"))
    cursor.execute("INSERT OR IGNORE INTO customers (id, name, email) VALUES (?, ?, ?)", (2, "Bob Johnson", "bob@example.com"))
    cursor.execute("INSERT OR IGNORE INTO orders (order_id, customer_id, product_name, status) VALUES (?, ?, ?, ?)", ("ORD001", 1, "Laptop X1", "Shipped"))
    cursor.execute("INSERT OR IGNORE INTO orders (order_id, customer_id, product_name, status) VALUES (?, ?, ?, ?)", ("ORD002", 1, "Mouse Pro", "Processing"))
    cursor.execute("INSERT OR IGNORE INTO orders (order_id, customer_id, product_name, status) VALUES (?, ?, ?, ?)", ("ORD003", 2, "Keyboard Pro", "Delivered"))

    conn.commit()
    conn.close()

# --- ChromaDB Knowledge Base Setup ---
def init_chroma_db():
    # Check if DB exists, if not, create and populate
    if not os.path.exists(CHROMA_DB_PATH) or not Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)._collection.count():
        print("Initializing ChromaDB with knowledge base...")
        documents_data = [
            {"page_content": "Our refund policy allows returns within 30 days of purchase for a full refund. Items must be in original condition.", "metadata": {"source": "policy"}},
            {"page_content": "The SuperWidget 3000 features a 12MP camera, 6.2-inch display, and 256GB storage. It costs $799.", "metadata": {"source": "product"}},
            {"page_content": "To reset your password, visit the 'Forgot Password' link on the login page and follow the instructions.", "metadata": {"source": "faq"}},
            {"page_content": "Shipping usually takes 3-5 business days for standard delivery within the US.", "metadata": {"source": "shipping"}},
            {"page_content": "The MegaSpeaker Xtreme offers 360-degree sound and 20-hour battery life. It's waterproof and priced at $199.", "metadata": {"source": "product"}},
        ]
        
        # LangChain's Chroma.from_documents expects `Document` objects or dictionaries with 'page_content' and 'metadata'
        # We'll use the dictionary format directly.
        db = Chroma.from_documents(documents_data, embeddings, persist_directory=CHROMA_DB_PATH)
        print("ChromaDB initialized.")
    else:
        print("ChromaDB already exists, loading...")
        db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
    return db

# --- Query Complexity Classifier (Placeholder) ---
def classify_query_complexity(query: str) -> str:
    query = query.lower()
    if "refund" in query or "return" in query or "policy" in query:
        return "policy_inquiry"
    elif "product" in query or "features" in query or "specs" in query or "cost" in query or "price" in query:
        return "product_inquiry"
    elif "order" in query or "status" in query or "track" in query:
        return "order_status"
    elif "password" in query or "account" in query or "login" in query:
        return "account_management"
    elif "shipping" in query or "delivery" in query:
        return "shipping_inquiry"
    else:
        return "general_inquiry"

# --- Customer Support Agent Class ---
class AdaptiveCustomerSupportAgent:
    def __init__(self, llm, embeddings, chroma_db, sqlite_db_path):
        self.llm = llm
        self.embeddings = embeddings
        self.chroma_db = chroma_db
        self.sqlite_db_path = sqlite_db_path
        self.session_memories = {}

        # RAG chain setup
        self.retriever = self.chroma_db.as_retriever()
        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful e-commerce customer support assistant. Answer the user's questions truthfully and concisely based *only* on the provided context."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        self.document_chain = create_stuff_documents_chain(self.llm, self.qa_prompt)
        self.retrieval_chain = create_retrieval_chain(self.retriever, self.document_chain)

    def _get_session_memory(self, session_id: str):
        if session_id not in self.session_memories:
            # Use ConversationBufferWindowMemory to keep a window of recent interactions
            self.session_memories[session_id] = ConversationBufferWindowMemory(
                llm=self.llm, k=5, return_messages=True, memory_key="chat_history", output_key="answer"
            )
        return self.session_memories[session_id]

    def _get_customer_order_info(self, customer_identifier: str):
        conn = sqlite3.connect(self.sqlite_db_path)
        cursor = conn.cursor()
        customer_info = None
        orders_info = []

        # Try to find customer by email or ID (simplified for example)
        cursor.execute("SELECT id, name, email FROM customers WHERE email = ? OR id = ?", (customer_identifier, customer_identifier))
        customer_row = cursor.fetchone()

        if customer_row:
            customer_info = {"id": customer_row[0], "name": customer_row[1], "email": customer_row[2]}
            cursor.execute("SELECT order_id, product_name, status FROM orders WHERE customer_id = ?", (customer_info["id"],))
            orders_rows = cursor.fetchall()
            orders_info = [{"order_id": o[0], "product_name": o[1], "status": o[2]} for o in orders_rows]

        conn.close()
        return customer_info, orders_info

    def handle_query(self, session_id: str, query: str):
        memory = self._get_session_memory(session_id)
        chat_history = memory.load_memory_variables({})["chat_history"]
        query_complexity = classify_query_complexity(query)

        response_text = "I'm sorry, I couldn't process your request at this moment. Please try again later."
        context_info = {}

        try:
            if query_complexity == "order_status" or query_complexity == "account_management":
                # Attempt to extract customer identifier (e.g., email or ID) from query or history
                # For simplicity, we'll use a dummy customer_identifier for now
                customer_id_or_email = "alice@example.com" # In a real app, extract from query/session
                customer_info, orders_info = self._get_customer_order_info(customer_id_or_email)
                
                if customer_info:
                    context_info["customer_info"] = customer_info
                    context_info["orders_info"] = orders_info
                    
                    # Formulate a prompt to the LLM with structured data
                    system_prompt = (
                        f"You are an e-commerce customer support agent. Here is the customer's information: "
                        f"Customer Name: {customer_info['name']}, Email: {customer_info['email']}. "
                        f"Their orders: {json.dumps(orders_info)}. "
                        f"Answer the user's question about their orders or account based on this information. "
                        f"If the information is not present, state that you don't have it." 
                    )
                    user_prompt = f"The customer is asking: {query}"
                    
                    # Create a temporary chain for this specific query
                    specific_prompt = ChatPromptTemplate.from_messages([
                        ("system", system_prompt),
                        MessagesPlaceholder(variable_name="chat_history"),
                        ("human", user_prompt)
                    ])
                    specific_chain = specific_prompt | self.llm
                    response_text = specific_chain.invoke({"chat_history": chat_history, "input": user_prompt}).content

                else:
                    response_text = "I couldn't find any information for the customer provided. Please ensure the details are correct."

            elif query_complexity in ["policy_inquiry", "product_inquiry", "shipping_inquiry", "general_inquiry"]:
                # Use RAG for general knowledge base queries
                # The retrieval_chain automatically combines retrieval with the LLM.
                result = self.retrieval_chain.invoke({"input": query, "chat_history": chat_history})
                response_text = result["answer"]
            
            else:
                # Fallback for unhandled complexities or general inquiries
                # This can be a direct LLM call if no specific RAG or DB lookup is needed
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a helpful e-commerce customer support assistant. Provide relevant and helpful information."),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}")
                ])
                chain = prompt | self.llm
                response_text = chain.invoke({"chat_history": chat_history, "input": query}).content

        except Exception as e:
            print(f"Error processing query: {e}")
            response_text = "An error occurred while trying to process your request. Our team has been notified."

        # Update short-term memory with current interaction
        memory.save_context({"input": query}, {"answer": response_text})
        return response_text

# --- Flask Routes ---
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_query = data.get("query")
    session_id = data.get("session_id", "default_session") # Use a default session if not provided

    if not user_query:
        return jsonify({"error": "Query is required."}), 400

    response = agent.handle_query(session_id, user_query)
    return jsonify({"response": response})

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "message": "Adaptive Customer Support Agent is running."})

# --- Main Execution --- 
if __name__ == "__main__":
    # Initialize databases and agent
    init_sqlite_db()
    chroma_db_instance = init_chroma_db()
    agent = AdaptiveCustomerSupportAgent(llm, embeddings, chroma_db_instance, SQLITE_DB_PATH)
    
    print("\nAdaptive Customer Support Agent is ready!")
    print(f"Access the API at http://127.0.0.1:5000/chat")
    print(f"Health check at http://127.0.0.1:5000/health")
    print("\nExample usage (POST to /chat with JSON body):")
    print("{\"query\": \"What is your refund policy?\", \"session_id\": \"user123\"}")
    print("{\"query\": \"Tell me about the SuperWidget 3000.\", \"session_id\": \"user123\"}")
    print("{\"query\": \"What is the status of my order?\", \"session_id\": \"user456\"}")
    print("\nStarting Flask app...")
    app.run(debug=True) # Set debug=False in production
