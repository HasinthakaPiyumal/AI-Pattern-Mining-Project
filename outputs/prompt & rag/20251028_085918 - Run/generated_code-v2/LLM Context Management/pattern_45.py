import streamlit as st
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found. Please set it as an environment variable.")
    st.stop()

# --- Simulated Customer Data ---
customer_profiles = {
    "customer_123": {
        "name": "Alice Smith",
        "email": "alice.smith@example.com",
        "purchase_history": [
            "Order #2023-001: Laptop (paid)",
            "Order #2023-005: Wireless Mouse (paid)"
        ],
        "reported_issues": [
            "Issue #ISSUE-456: Laptop charger not working (resolved on 2023-10-20)"
        ]
    },
    "customer_456": {
        "name": "Bob Johnson",
        "email": "bob.j@example.com",
        "purchase_history": [
            "Order #2023-010: Smartwatch (pending shipment)"
        ],
        "reported_issues": []
    }
}

# --- ChromaDB Setup (Long-Term Memory) ---
# Initialize embeddings
embedding_function = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Initialize ChromaDB persistent client
CHROMA_DB_DIR = "./chroma_db"
vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embedding_function)
vectorstore.persist()

# --- LLM and Chain Setup ---
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7, openai_api_key=OPENAI_API_KEY)

# Conversation memory for short-term context within the current session
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Custom prompt template to include customer context and long-term memory
custom_template = """You are an AI customer support assistant for an e-commerce platform. Your goal is to provide helpful, accurate, and personalized support.

Customer ID: {customer_id}
Customer Name: {customer_name}
Customer Email: {customer_email}
Purchase History:
{purchase_history}
Reported Issues:
{reported_issues}

Relevant past conversations or knowledge from long-term memory:
{long_term_memory}

Chat History:
{chat_history}
Human: {question}
AI:"""

CUSTOM_PROMPT = PromptTemplate(input_variables=["customer_id", "customer_name", "customer_email", "purchase_history", "reported_issues", "long_term_memory", "chat_history", "question"], template=custom_template)

def get_customer_context(customer_id):
    profile = customer_profiles.get(customer_id, {})
    return (
        profile.get("name", "N/A"),
        profile.get("email", "N/A"),
        "\n".join(profile.get("purchase_history", ["No purchase history."])),
        "\n".join(profile.get("reported_issues", ["No reported issues."]))    )

def get_long_term_memory(query, k=3):
    # Retrieve similar documents from ChromaDB
    docs = vectorstore.similarity_search(query, k=k)
    return "\n".join([doc.page_content for doc in docs]) if docs else "No relevant past interactions found."

# --- Streamlit App ---
st.set_page_config(page_title="Adaptive E-commerce Chatbot", layout="centered")
st.title("🛒 Adaptive E-commerce Support Chatbot")
st.markdown("Hello! I'm your AI assistant. I remember our past conversations and your purchase history to provide personalized support.")

# Simulated Customer ID input
current_customer_id = st.sidebar.text_input("Enter Customer ID (e.g., customer_123):", "customer_123")

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
        with st.spinner("Thinking..."):
            # Get customer-specific context
            customer_name, customer_email, purchase_history, reported_issues = get_customer_context(current_customer_id)

            # Get relevant long-term memory
            long_term_memory_content = get_long_term_memory(prompt)

            # Prepare chat history for prompt
            chat_history_str = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.messages if msg['role'] != 'assistant' and msg['content'] != prompt])

            # Create the final prompt
            formatted_prompt = CUSTOM_PROMPT.format(
                customer_id=current_customer_id,
                customer_name=customer_name,
                customer_email=customer_email,
                purchase_history=purchase_history,
                reported_issues=reported_issues,
                long_term_memory=long_term_memory_content,
                chat_history=chat_history_str,
                question=prompt
            )

            # Get LLM response
            ai_response_obj = llm.invoke(formatted_prompt)
            ai_response = ai_response_obj.content
            st.markdown(ai_response)

            # Store current interaction in ChromaDB for long-term memory
            full_interaction = f"Human: {prompt}\nAI: {ai_response}"
            vectorstore.add_texts([full_interaction], metadatas=[{"customer_id": current_customer_id, "type": "interaction"}])
            vectorstore.persist()

    st.session_state.messages.append({"role": "assistant", "content": ai_response})
