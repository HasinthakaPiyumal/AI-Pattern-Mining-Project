import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory, VectorStoreRetrieverMemory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# --- 1. Simulated E-commerce Data ---
simulated_ecommerce_data = {
    "products": {
        "101": {"name": "Laptop Pro", "price": 1200, "category": "Electronics"},
        "102": {"name": "Wireless Mouse", "price": 25, "category": "Accessories"},
        "103": {"name": "Mechanical Keyboard", "price": 75, "category": "Accessories"},
        "201": {"name": "Adventure Backpack", "price": 60, "category": "Outdoor"},
        "202": {"name": "Hiking Boots", "price": 110, "category": "Outdoor"},
    },
    "orders": {
        "ORD001": {"customer_id": "CUST001", "items": [{"product_id": "101", "qty": 1}], "status": "Shipped"},
        "ORD002": {"customer_id": "CUST002", "items": [{"product_id": "102", "qty": 2}], "status": "Processing"},
    },
    "customer_preferences": {
        "CUST001": "Prefers high-performance electronics and fast shipping.",
        "CUST002": "Interested in durable outdoor gear and discounts.",
    }
}

# --- 2. Embedding Model & ChromaDB (Long-Term Memory) ---
# Initialize embedding model
@st.cache_resource
def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = get_embeddings_model()

# Initialize ChromaDB
# For simplicity, using an in-memory ChromaDB. For persistence, specify persist_directory
@st.cache_resource
def get_chroma_db(embeddings_model):
    # Initial data for long-term memory
    initial_memory_docs = [
        "Customer CUST001 previously purchased Laptop Pro and was concerned about warranty.",
        "Customer CUST002 asked about return policy for hiking boots last month.",
        "Common issue: Customers often ask about shipping times for electronics.",
        "Resolved issue: For 'Laptop Pro' display flickering, recommend driver update and checking cable connections.",
        "Product 'Wireless Mouse' is frequently bought with 'Mechanical Keyboard'."
    ]
    db = Chroma.from_texts(initial_memory_docs, embeddings_model)
    return db

vectordb = get_chroma_db(embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# --- 3. LangChain Components ---
# LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7, openai_api_key=os.getenv("OPENAI_API_KEY"))

# Short-term memory (Conversation Buffer)
conversation_memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    return_messages=True,
    k=5  # Keep a window of 5 recent conversational turns
)

# Long-term memory (Vector Store Retriever Memory)
# This will integrate the ChromaDB retriever into the chain's memory
long_term_memory = VectorStoreRetrieverMemory(retriever=retriever)

# Define a custom prompt template to incorporate retrieved context and simulated data
prompt_template = """You are an intelligent customer support agent for an e-commerce platform.
Answer the customer's questions based on the provided context, chat history, and any relevant e-commerce data.
If you don't know the answer, politely state that you cannot assist with that specific query or need more information.

Simulated E-commerce Data:
Products: {products_data}
Orders: {orders_data}
Customer Preferences: {preferences_data}

Retrieved Long-Term Memory (Past Interactions/Knowledge):
{long_term_memory}

Chat History:
{chat_history}

Customer: {question}
Agent:"""

QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["products_data", "orders_data", "preferences_data", "long_term_memory", "chat_history", "question"],
    template=prompt_template,
)

# Create a function to format e-commerce data for the prompt
def format_ecommerce_data():
    products_str = "\n".join([f"ID: {pid}, Name: {data['name']}, Price: ${data['price']}, Category: {data['category']}" for pid, data in simulated_ecommerce_data['products'].items()])
    orders_str = "\n".join([f"ID: {oid}, Customer: {data['customer_id']}, Status: {data['status']}" for oid, data in simulated_ecommerce_data['orders'].items()])
    preferences_str = "\n".join([f"Customer {cid}: {pref}" for cid, pref in simulated_ecommerce_data['customer_preferences'].items()])
    return products_str, orders_str, preferences_str

# Conversational Chain
# We'll manually pass the memories and formatted data to the LLM call rather than using a direct chain with multiple memories
# This gives more control over how the memories are presented to the LLM.

# --- 4. Streamlit Frontend ---
st.title("🛒 E-commerce Support Chatbot")
st.write("Ask me about products, orders, or any support queries!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How can I help you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Get formatted e-commerce data
        products_data, orders_data, preferences_data = format_ecommerce_data()

        # Retrieve relevant long-term memories
        relevant_long_term_memory = long_term_memory.load_memory_variables({"query": prompt})["retriever_memory"]

        # Get short-term chat history
        chat_history = conversation_memory.load_memory_variables({})["chat_history"]
        formatted_chat_history = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history])

        # Prepare inputs for the LLM prompt
        inputs = {
            "products_data": products_data,
            "orders_data": orders_data,
            "preferences_data": preferences_data,
            "long_term_memory": relevant_long_term_memory,
            "chat_history": formatted_chat_history,
            "question": prompt,
        }

        # Generate response using LLM with custom prompt
        full_prompt = QA_CHAIN_PROMPT.format(**inputs)
        response = llm.invoke(full_prompt).content
        st.markdown(response)

        # Update short-term conversation memory
        conversation_memory.save_context({"input": prompt}, {"output": response})
        st.session_state.messages.append({"role": "assistant", "content": response})

