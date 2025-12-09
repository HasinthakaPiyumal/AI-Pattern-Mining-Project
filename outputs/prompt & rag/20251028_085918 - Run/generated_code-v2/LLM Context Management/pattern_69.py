import os
import sqlite3
from typing import List, Dict, Any

from fastapi import FastAPI
from pydantic import BaseModel

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from dotenv import load_dotenv

load_dotenv()

class ChatRequest(BaseModel):
    customer_id: str
    message: str

# --- Database Initialization ---
def init_sqlite_db():
    conn = sqlite3.connect("customer_support.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            id TEXT PRIMARY KEY,
            name TEXT,
            preferences TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT,
            user_message TEXT,
            agent_response TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            resolution_status TEXT
        )
    """)
    conn.commit()
    conn.close()

    # Add a dummy customer if not exists
    conn = sqlite3.connect("customer_support.db")
    cursor = conn.cursor()
    cursor.execute("INSERT OR IGNORE INTO customers (id, name, preferences) VALUES (?, ?, ?)", ('customer_123', 'Alice Smith', 'Prefers email updates, often asks about billing.'))
    conn.commit()
    conn.close()


# --- Memory Manager Class ---
class MemoryManager:
    def __init__(self, db_path: str = "customer_support.db"):
        self.db_path = db_path
        self.embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.chroma_collection = Chroma(client_settings=None, 
                                         collection_name="customer_memories", 
                                         embedding_function=self.embeddings_model)

    def get_customer_profile(self, customer_id: str) -> Dict[str, Any]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, preferences FROM customers WHERE id = ?", (customer_id,))
        profile = cursor.fetchone()
        conn.close()
        if profile:
            return {"id": profile[0], "name": profile[1], "preferences": profile[2]}
        return {"id": customer_id, "name": "Unknown Customer", "preferences": "None"}

    def retrieve_memories(self, customer_id: str, query_text: str, k: int = 3) -> List[str]:
        docs = self.chroma_collection.similarity_search(query_text, k=k, filter={"customer_id": customer_id})
        return [doc.page_content for doc in docs]

    def store_interaction(self, customer_id: str, user_message: str, agent_response: str, resolution_status: str = "ongoing"):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO conversations (customer_id, user_message, agent_response, resolution_status) VALUES (?, ?, ?, ?)",
                       (customer_id, user_message, agent_response, resolution_status))
        conversation_id = cursor.lastrowid
        conn.commit()
        conn.close()

        # Store in ChromaDB
        full_interaction = f"Customer: {user_message}\nAgent: {agent_response}\nResolution Status: {resolution_status}"
        self.chroma_collection.add_texts(
            texts=[full_interaction],
            metadatas=[{"customer_id": customer_id, "conversation_id": conversation_id}]
        )


# --- FastAPI Application ---
app = FastAPI()
init_sqlite_db()
memory_manager = MemoryManager()

# --- LLM and Prompt Setup ---
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, api_key=os.getenv("OPENAI_API_KEY"))

# Define the prompt template for the LLM
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful and personalized AI customer support agent. Use the provided customer profile and past interactions to give the best possible support. If you don't know an answer, politely state that you cannot assist with that specific query."),
    ("human", "Customer Profile:\n{customer_profile}\n\nPast Interactions:\n{past_memories}\n\nCurrent Customer Query:\n{query}")
])

# Define the LangChain processing chain
rag_chain = (
    RunnablePassthrough.assign(
        past_memories=lambda x: "\n".join(memory_manager.retrieve_memories(x["customer_id"], x["query"])),
        customer_profile=lambda x: str(memory_manager.get_customer_profile(x["customer_id"]))
    )
    | prompt_template
    | llm
    | StrOutputParser()
)

@app.post("/chat")
async def chat_with_agent(request: ChatRequest):
    customer_id = request.customer_id
    user_message = request.message

    # Invoke the RAG chain
    response = rag_chain.invoke({"customer_id": customer_id, "query": user_message})
    agent_response = response

    # Store the current interaction
    memory_manager.store_interaction(customer_id, user_message, agent_response)

    return {"customer_id": customer_id, "agent_response": agent_response}

# To run this FastAPI application, save it as `main.py` (or any other name) and use:
# uvicorn main:app --reload

# Example usage with curl:
# curl -X POST "http://127.0.0.1:8000/chat" -H "Content-Type: application/json" -d '{"customer_id": "customer_123", "message": "What is my current billing status?"}'
# curl -X POST "http://127.0.0.1:8000/chat" -H "Content-Type: application/json" -d '{"customer_id": "customer_123", "message": "I forgot my password, can you help me reset it?"}'
