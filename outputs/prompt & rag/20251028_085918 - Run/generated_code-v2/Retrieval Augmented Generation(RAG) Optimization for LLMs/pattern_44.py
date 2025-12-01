import os
import uvicorn
import asyncio
import requests # Needed for Streamlit client to talk to FastAPI
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

import streamlit as st

# LangChain components
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- FastAPI Backend Application ---
# This part defines the API server.
# It should be run separately (e.g., `uvicorn main:fastapi_app --reload`)
# assuming this code is in `main.py`.

fastapi_app = FastAPI(title="Customer Support Chatbot API")

# Global variables for RAG system (initialized once on startup)
vectorstore_instance = None
retrieval_chain_instance = None

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str

def initialize_rag_system():
    global vectorstore_instance, retrieval_chain_instance

    # Ensure a 'data' directory and a sample document exist for demonstration
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists("data/sample_doc.txt"):
        with open("data/sample_doc.txt", "w") as f:
            f.write("This is a sample document for customer support. It contains information about product features and common troubleshooting steps. For example, if your device is not turning on, please check the power cable. Our new product X has feature Y and Z. Contact support at support@example.com for more details.")
            f.write("\nAnother document part: Our refund policy states that returns are accepted within 30 days of purchase with a valid receipt. Special conditions apply for digital products.")
            f.write("\nDocument 3: For issues with login, please reset your password using the 'Forgot Password' link. If the problem persists, ensure your account is active.")

    # 3. Core RAG System (LangChain)
    # Document Loader
    loader = DirectoryLoader("./data", glob="**/*.txt", loader_cls=TextLoader)
    docs = loader.load()

    # Text Splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    # Embedding Model (using a local HuggingFace model for easier setup)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Vector Store (ChromaDB for simplicity, in-memory by default)
    # For persistence, uncomment: persist_directory = "./chroma_db"
    # vectorstore_instance = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=persist_directory)
    # If using persist_directory, you might need to load from it:
    # vectorstore_instance = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    vectorstore_instance = Chroma.from_documents(documents=splits, embedding=embeddings)


    # Retriever Configuration: Optimal Document Count
    # This directly implements the "Optimal Document Count for In-Context Learning" pattern.
    # Empirically set k (number of documents) to a small value (e.g., 2 or 3)
    # for initial gains, as suggested by the pattern.
    optimal_document_count = 2 # This value can be tuned based on performance evaluation
    retriever = vectorstore_instance.as_retriever(search_kwargs={"k": optimal_document_count})

    # Language Model (LLM)
    # Requires OPENAI_API_KEY environment variable to be set.
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # Prompt Template
    system_prompt_template = """You are a helpful customer support assistant.
    Answer the user's question concisely based *only* on the provided context.
    If the context does not contain the answer, politely state that you cannot find the answer in the provided information.
    Do not invent information.

    Context:
    {context}
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt_template),
            ("human", "{input}"),
        ]
    )

    # RAG Chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain_instance = create_retrieval_chain(retriever, question_answer_chain)

    print(f"RAG system initialized with optimal document count: {optimal_document_count}")

@fastapi_app.on_event("startup")
async def startup_event_fastapi():
    initialize_rag_system()

@fastapi_app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if retrieval_chain_instance is None:
        return ChatResponse(response="RAG system is not initialized. Please wait or check server logs.")
    
    try:
        response = retrieval_chain_instance.invoke({"input": request.query})
        return ChatResponse(response=response["answer"])
    except Exception as e:
        print(f"Error during RAG chain invocation: {e}")
        return ChatResponse(response="I encountered an error while processing your request. Please try again.")

# --- Streamlit Frontend Application ---
# This part defines the web UI.
# It should be run separately (e.g., `streamlit run main.py`)
# assuming this code is in `main.py`.

def streamlit_frontend():
    st.set_page_config(page_title="Customer Support Chatbot")
    st.header("Customer Support Chatbot with Optimized Retrieval")

    # URL for the FastAPI backend (assuming it's running on http://localhost:8000)
    FASTAPI_BACKEND_URL = "http://localhost:8000/chat"

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
            with st.spinner("Finding the best answer..."):
                try:
                    # Make a POST request to the FastAPI backend
                    response = requests.post(FASTAPI_BACKEND_URL, json={"query": prompt})
                    response.raise_for_status() # Raise an exception for HTTP errors
                    chatbot_response = response.json()["response"]
                    st.markdown(chatbot_response)
                    st.session_state.messages.append({"role": "assistant", "content": chatbot_response})
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to the chatbot backend. Please ensure the FastAPI server is running.")
                    st.session_state.messages.append({"role": "assistant", "content": "I'm sorry, I can't connect to my knowledge base right now. Please check if the server is running."})
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": "I apologize, an error occurred. Please try again later."})

# A simple way to run Streamlit when the script is executed directly
if __name__ == "__main__":
    # If you run this file directly with `python main.py`, it will launch the Streamlit app.
    # The FastAPI app needs to be run separately via `uvicorn`.
    # This design keeps the "single code" requirement while acknowledging execution.
    streamlit_frontend()