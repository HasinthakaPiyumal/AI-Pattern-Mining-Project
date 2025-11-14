"""
This script sets up an intelligent customer support chatbot for an e-commerce platform.
It integrates a FastAPI backend, a Streamlit frontend, LangChain for RAG, and simulates
interaction with a vLLM inference server. The architecture is designed to leverage
KV Cache Reuse, PagedAttention, Replication of Critical KV Cache Nodes (conceptually),
and a Swap-Out-Only-Once Cache Strategy through vLLM.
"""

import os
import uvicorn
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import streamlit as st

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.language_models import BaseLLM

# --- Configuration --- #

# FastAPI server configuration
FASTAPI_HOST = os.getenv("FASTAPI_HOST", "127.0.0.1")
FASTAPI_PORT = int(os.getenv("FASTAPI_PORT", 8000))

# vLLM server configuration (assuming vLLM is running separately)
vLLM_API_BASE = os.getenv("VLLM_API_BASE", "http://localhost:8001/v1") # Default vLLM OpenAI-compatible API
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2") # Or any LLM served by vLLM

# ChromaDB configuration
CHROMA_PERSIST_DIRECTORY = "./chroma_db"

# --- 1. ChromaDB Setup (Knowledge Base) --- #

def setup_chroma_db():
    """Initializes ChromaDB with sample e-commerce data if it doesn't exist."""
    if not os.path.exists(CHROMA_PERSIST_DIRECTORY):
        os.makedirs(CHROMA_PERSIST_DIRECTORY, exist_ok=True)
        print("ChromaDB directory created.")

    # Sample E-commerce Data
    ecommerce_data = [
        "Our return policy allows returns within 30 days of purchase with a valid receipt.",
        "Shipping usually takes 3-5 business days for standard delivery within the country.",
        "You can track your order using the tracking number provided in your shipping confirmation email.",
        "We accept major credit cards (Visa, Mastercard, Amex) and PayPal.",
        "To reset your password, click on 'Forgot Password' on the login page and follow the instructions.",
        "Our customer support is available Monday to Friday, 9 AM to 5 PM EST."
    ]

    # Write data to a temporary file to use TextLoader
    with open("ecommerce_faq.txt", "w") as f:
        for line in ecommerce_data:
            f.write(line + "\n")

    # Load and split documents
    loader = TextLoader("ecommerce_faq.txt")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    # Initialize embeddings
    # Use a local embedding model to avoid API keys for demonstration
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create and persist the vector store
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIRECTORY
    )
    vectorstore.persist()
    print(f"ChromaDB initialized and data loaded to {CHROMA_PERSIST_DIRECTORY}")

    # Clean up temporary file
    os.remove("ecommerce_faq.txt")
    return vectorstore, embeddings

vectorstore, embeddings = setup_chroma_db()

# --- 2. Custom LangChain LLM Wrapper for vLLM --- #

# This class wraps the vLLM API to be compatible with LangChain's BaseLLM
class VLLMHttpLLM(BaseLLM):
    """Custom LLM wrapper to interact with a vLLM server via HTTP API."""
    vllm_api_base: str
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 512

    @property
    def _llm_type(self) -> str:
        return "vllm_http"

    def _call(self, prompt: str, stop=None, **kwargs) -> str:
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False # LangChain expects a single response
        }
        if stop: # vLLM uses 'stop' token, not 'stop_sequences'
            payload["stop"] = stop

        try:
            response = requests.post(f"{self.vllm_api_base}/completions", json=payload, headers=headers, timeout=60)
            response.raise_for_status() # Raise an exception for HTTP errors
            return response.json()["choices"][0]["text"]
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Error communicating with vLLM server: {e}")

    @property
    def _identifying_params(self) -> dict:
        return {
            "vllm_api_base": self.vllm_api_base,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

# Initialize the custom vLLM wrapper
vllm_llm = VLLMHttpLLM(
    vllm_api_base=vLLM_API_BASE,
    model_name=MODEL_NAME,
    temperature=0.1, # Keep temperature low for factual chatbot responses
    max_tokens=256
)

# --- 3. FastAPI Backend --- #

app = FastAPI(title="E-commerce Chatbot API")

class ChatRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Endpoint for receiving user queries and returning chatbot responses."""
    try:
        # Create RAG chain with LangChain
        # Using an LCEL chain for more flexibility and transparency
        retriever = vectorstore.as_retriever()

        # Define the prompt template
        template = """You are an helpful e-commerce customer support assistant. Answer the user's question based only on the provided context. If you don't know the answer, politely state that you cannot provide that information.

Context: {context}
Question: {question}
Answer:"""
        prompt = ChatPromptTemplate.from_template(template)

        # RAG chain
        rag_chain = (
            {"context": retriever | (lambda docs: "\n\n".join([doc.page_content for doc in docs])), "question": RunnablePassthrough()}
            | prompt
            | vllm_llm # Use the custom vLLM LLM wrapper
            | StrOutputParser()
        )

        response = rag_chain.invoke(request.query)
        return {"response": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- 4. Streamlit Frontend --- #

# Only run Streamlit if __name__ == "__main__" and not launched via uvicorn/FastAPI
# Streamlit part needs to be run separately (e.g., `streamlit run ecommerce_chatbot.py`)
# This section will only execute if the script is run directly, not imported by uvicorn.

def run_streamlit_app():
    st.set_page_config(page_title="E-commerce Chatbot")
    st.header("🛍️ E-commerce Customer Support Chatbot")

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
                try:
                    # Make a request to the FastAPI backend
                    response = requests.post(
                        f"http://{FASTAPI_HOST}:{FASTAPI_PORT}/chat",
                        json={"query": prompt}
                    )
                    response.raise_for_status()
                    chatbot_response = response.json()["response"]
                except requests.exceptions.ConnectionError:
                    chatbot_response = "Could not connect to the backend API. Please ensure the FastAPI server is running."
                except requests.exceptions.RequestException as e:
                    chatbot_response = f"An error occurred with the API request: {e}"

                st.markdown(chatbot_response)
                st.session_state.messages.append({"role": "assistant", "content": chatbot_response})

# --- Running the Applications --- #

if __name__ == "__main__":
    # Instructions for running:
    print("\n--- Instructions ---")
    print("1. **Start vLLM Server (if not already running):**")
    print(f"   Example: `python -m vllm.entrypoints.api_server --model {MODEL_NAME} --port 8001`")
    print("   Adjust --model and --port as needed. Ensure vLLM_API_BASE in config matches.")
    print("2. **Start FastAPI Backend:**")
    print(f"   Open a new terminal and run: `uvicorn ecommerce_chatbot:app --host {FASTAPI_HOST} --port {FASTAPI_PORT}`")
    print("3. **Start Streamlit Frontend:**")
    print("   Open another terminal and run: `streamlit run ecommerce_chatbot.py`")
    print("--------------------\n")

    # This part is primarily for running Streamlit when the script is executed directly.
    # FastAPI is typically run via `uvicorn`. We'll print instructions for both.
    run_streamlit_app()


