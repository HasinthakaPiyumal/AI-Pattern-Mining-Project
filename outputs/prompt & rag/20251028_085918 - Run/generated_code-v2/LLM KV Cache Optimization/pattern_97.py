import streamlit as st
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import threading
import httpx
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

# --- Configuration --- #
VLLM_SERVER_URL = os.getenv("VLLM_SERVER_URL", "http://localhost:8000/generate")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_COLLECTION_NAME = "customer_support_kb"

# --- FastAPI Backend --- #
app = FastAPI()

# Initialize embedding model and ChromaDB
@st.cache_resource
def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

@st.cache_resource
def get_chroma_client():
    client = chromadb.Client()
    collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
    
    # Add some dummy data to ChromaDB for demonstration
    if collection.count() == 0:
        documents = [
            "Our product Alpha is a revolutionary AI assistant that helps with daily tasks. It costs $99.",
            "Beta is our advanced analytics platform, offering real-time insights for businesses. It has a subscription model of $49/month.",
            "To reset your password, please visit our website and click on 'Forgot Password' link.",
            "Our customer support is available 24/7 via chat and email. Phone support is from 9 AM to 5 PM EST.",
            "Product Alpha comes with a 1-year warranty. Beta has continuous software updates."
        ]
        metadatas = [
            {"source": "product_faq"},
            {"source": "product_details"},
            {"source": "account_management"},
            {"source": "contact_info"},
            {"source": "warranty_info"}
        ]
        ids = [f"doc{i}" for i in range(len(documents))]
        
        print("Adding dummy documents to ChromaDB...")
        collection.add(documents=documents, metadatas=metadatas, ids=ids)
        print("Dummy documents added.")
    return collection

embedding_model = get_embedding_model()
chroma_collection = get_chroma_client()

# Pydantic model for chat request
class ChatRequest(BaseModel):
    user_query: str
    conversation_history: list[dict]

# Pydantic model for vLLM API request (simplified)
class VLLMRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # 1. Embed user query
        query_embedding = embedding_model.encode(request.user_query).tolist()

        # 2. RAG - Retrieve relevant documents from ChromaDB
        retrieved_docs = chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )
        
        context_docs = "\n".join([doc for sublist in retrieved_docs['documents'] for doc in sublist])

        # 3. Construct prompt using LangChain template
        system_template = (
            "You are a helpful customer support assistant. Answer the user's questions based on the provided context and conversation history. "
            "If you don't know the answer, state that you don't have enough information. Keep your answers concise."
            "Context: {context}\n"
            "Conversation History: {history}"
        )
        human_template = "User: {query}"

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", human_template),
        ])
        
        # Format conversation history for prompt
        history_str = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in request.conversation_history])

        formatted_prompt = prompt.format(
            context=context_docs,
            history=history_str,
            query=request.user_query
        )

        # 4. Call vLLM inference server (simulated or actual)
        async with httpx.AsyncClient() as client:
            vllm_request_payload = VLLMRequest(prompt=formatted_prompt).model_dump_json()
            
            # For demonstration, if vLLM server is not running, simulate a response
            try:
                vllm_response = await client.post(VLLM_SERVER_URL, 
                                                  content=vllm_request_payload,
                                                  headers={"Content-Type": "application/json"}, 
                                                  timeout=60)
                vllm_response.raise_for_status() # Raise an exception for HTTP errors
                llm_response_data = vllm_response.json()
                # vLLM response structure can vary, assuming text is in 'text' key within 'outputs'
                if llm_response_data and "text" in llm_response_data['outputs'][0]:
                    llm_generated_text = llm_response_data['outputs'][0]['text'][0] # Adjust based on actual vLLM output format
                else:
                    llm_generated_text = "Error: Could not parse LLM response."
            except httpx.RequestError as e:
                print(f"vLLM server connection error: {e}")
                llm_generated_text = f"(Simulated LLM response due to vLLM server error: Could not connect to LLM. Context: {context_docs}. Query: {request.user_query})\nAnswer to \"{request.user_query}\" based on retrieved documents: Our product Alpha costs $99. To reset your password, visit our website." # Fallback simulation
            except httpx.HTTPStatusError as e:
                print(f"vLLM server returned HTTP error: {e.response.status_code} - {e.response.text}")
                llm_generated_text = f"(Simulated LLM response due to vLLM HTTP error: {e.response.status_code}. Query: {request.user_query})\nAnswer to \"{request.user_query}\" based on retrieved documents: Our customer support is available 24/7. Beta offers real-time insights."
            except Exception as e:
                print(f"Unexpected error processing vLLM response: {e}")
                llm_generated_text = f"(Simulated LLM response due to unexpected error: {e}. Query: {request.user_query})\nAnswer to \"{request.user_query}\" based on retrieved documents: Please refer to our FAQ section for more details."

        return {"response": llm_generated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Streamlit Frontend --- #
st.set_page_config(page_title="Smart Customer Support Chatbot")
st.title("🤖 Smart Customer Support Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask me a question about our products or services..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Send query to FastAPI backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Use httpx for Streamlit to FastAPI communication
                response = httpx.post(
                    "http://localhost:8001/chat", # Assuming FastAPI runs on 8001
                    json={"user_query": prompt, "conversation_history": st.session_state.messages[:-1]} # Exclude current prompt from history sent to LLM for RAG context
                )
                response.raise_for_status() # Raise an exception for HTTP errors
                llm_response = response.json()["response"]
            except httpx.RequestError as e:
                llm_response = f"Error connecting to backend: {e}. Please ensure the backend is running." 
                st.error(llm_response)
            except httpx.HTTPStatusError as e:
                llm_response = f"Backend returned an error: {e.response.status_code} - {e.response.text}"
                st.error(llm_response)
            except Exception as e:
                llm_response = f"An unexpected error occurred: {e}"
                st.error(llm_response)

        st.markdown(llm_response)
        st.session_state.messages.append({"role": "assistant", "content": llm_response})


# --- Uvicorn Server Thread for FastAPI --- #
def run_fastapi():
    # This assumes the script is run in a way that Streamlit doesn't block this
    # In a real deployment, FastAPI and Streamlit would be separate processes/containers
    uvicorn.run(app, host="0.0.0.0", port=8001)

# Start FastAPI in a separate thread if not already running
# This is a hack for demonstration; not for production
if __name__ == "__main__":
    # Check if a FastAPI thread is already running (simple check)
    fastapi_thread_exists = False
    for thread in threading.enumerate():
        if thread.name == "FastAPI_Server_Thread":
            fastapi_thread_exists = True
            break

    if not fastapi_thread_exists:
        print("Starting FastAPI server in a separate thread...")
        fastapi_thread = threading.Thread(target=run_fastapi, name="FastAPI_Server_Thread")
        fastapi_thread.daemon = True # Allow the main program to exit even if thread is running
        fastapi_thread.start()
        print("FastAPI server thread started. Access Streamlit at http://localhost:8501")

    # Streamlit will automatically run the rest of the script if this is the entry point
    # The main Streamlit app code runs here.
