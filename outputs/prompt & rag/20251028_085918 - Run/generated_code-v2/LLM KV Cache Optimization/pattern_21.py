from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import streamlit as st
import requests

from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import VLLM

# --- 1. Knowledge Base (ChromaDB) --- #
# For demonstration, we'll use an in-memory Chroma instance
# In a real application, you'd persist this or connect to a running Chroma server.
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

docs = [
    "Our product offers 24/7 customer support via chat and email.",
    "To reset your password, visit our website and click 'Forgot Password'.",
    "Billing cycles are monthly, and payments are due on the 15th of each month.",
    "Our premium plan includes unlimited storage and priority support.",
    "Returns are accepted within 30 days of purchase with a valid receipt."
]

vectorstore = Chroma.from_texts(docs, embedding_function, collection_name="customer_support_kb")
retriever = vectorstore.as_retriever()

# --- 2. LLM Serving with KV Cache Reuse (vLLM Placeholder) --- #
# In a real deployment, vLLM would be running as a separate server.
# Here, we're demonstrating how Langchain would connect to it.
# You would replace 'http://localhost:8000/generate' with your actual vLLM endpoint.
# vLLM inherently handles KV cache reuse for common prefixes.
llm = VLLM(
    model="microsoft/Phi-3-mini-4k-instruct", # Replace with your chosen model
    trust_remote_code=True,  # Recommended for Phi-3
    tensor_parallel_size=1,  # Adjust based on your GPU setup
    max_new_tokens=256,
    temperature=0.7,
)

# --- 3. Conversational Logic & RAG Orchestration (Langchain) --- #
memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

# Define the prompt template for RAG and conversational context
SYSTEM_TEMPLATE = """
You are a helpful customer support assistant. Answer the user's questions truthfully 
based on the context provided. If you don't know the answer, state that you don't 
know. Keep your answers concise.

{context}

{chat_history}
"""

question_answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_TEMPLATE),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    RunnablePassthrough.assign(context=(lambda x: x["question"] | retriever | format_docs))
    | question_answer_prompt
    | llm
    | StrOutputParser()
)

# --- 4. FastAPI Backend --- #
app = FastAPI()

class ChatInput(BaseModel):
    user_message: str

@app.post("/chat")
async def chat_endpoint(chat_input: ChatInput):
    user_message = chat_input.user_message

    # Load chat history from memory before processing the current message
    inputs = {"question": user_message}
    history_for_rag = memory.load_memory_variables({})
    inputs.update(history_for_rag)

    # Invoke the RAG chain
    response = rag_chain.invoke(inputs)

    # Save current interaction to memory
    memory.save_context({"inputs": user_message}, {"outputs": response})

    return {"response": response}

# --- 5. Streamlit Frontend --- #
if st.session_state.get("run_streamlit", True):
    st.title("KV Cache Reusing Customer Support Chatbot")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("How can I help you?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Assuming FastAPI server is running on localhost:8000
                    backend_url = "http://localhost:8000/chat"
                    response = requests.post(backend_url, json={"user_message": prompt})
                    response.raise_for_status() # Raise an exception for HTTP errors
                    bot_response = response.json()["response"]
                    st.markdown(bot_response)
                    st.session_state.messages.append({"role": "assistant", "content": bot_response})
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to the backend server. Please ensure FastAPI is running.")
                except requests.exceptions.RequestException as e:
                    st.error(f"An error occurred: {e}")

# Instructions to run:
# 1. Save this code as `chatbot_system.py`.
# 2. Make sure you have `pip install fastapi uvicorn streamlit langchain-chroma langchain-community sentence-transformers`.
# 3. To run the FastAPI backend:
#    `uvicorn chatbot_system:app --host 0.0.0.0 --port 8000`
# 4. To run the Streamlit frontend (in a separate terminal):
#    `streamlit run chatbot_system.py -- --run_streamlit True`
# Note: This code includes a placeholder for vLLM. In a real setup, you would start a vLLM server separately.
# For example: `python -m vllm.entrypoints.api_server --model microsoft/Phi-3-mini-4k-instruct`

