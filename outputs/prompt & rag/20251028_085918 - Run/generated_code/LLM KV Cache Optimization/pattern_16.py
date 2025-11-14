import os
import uvicorn
import httpx # For the Streamlit client to talk to FastAPI
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any

# For LLM Inference
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    print("vLLM not found. Please install with 'pip install vllm'. Running in mock mode.")
    VLLM_AVAILABLE = False

# For RAG
try:
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    print("LangChain and Chroma/SentenceTransformers not found. Please install with 'pip install langchain-community chromadb sentence-transformers'. Running in mock mode.")
    LANGCHAIN_AVAILABLE = False

# --- FastAPI Backend ---

app = FastAPI(title="E-commerce Chatbot with Optimized KV Cache")

# Global variables for LLM, Embeddings, and Vector Store
llm_model = None
tokenizer = None
embeddings_model = None
vector_store = None

# Conceptual in-memory store for critical KV cache replication
# In a real system, this would be a distributed store like Redis or a shared memory segment via Ray
CRITICAL_KV_CACHE_STORE: Dict[str, str] = {} # Storing simplified representations for demo

class KVCacheManager:
    """
    Conceptual manager for KV Cache related optimizations and fault tolerance.
    In a real scenario, this would interact with vLLM's internal mechanisms,
    Ray for distributed shared memory/actors, and Redis for persistence.
    """
    def __init__(self):
        self.critical_prefixes = ["hello", "hi", "what is your name", "order status"] # Example critical prefixes

    def replicate_critical_kv_node(self, conversation_id: str, prefix_text: str, generated_response: str):
        """
        Simulates replication of a critical KV cache node.
        In a real system, this would involve extracting KV tensors from vLLM,
        serializing them, and pushing to a persistent store.
        Here, we store a simplified representation.
        """
        if any(prefix_text.lower().startswith(p) for p in self.critical_prefixes):
            print(f"Simulating replication for critical prefix: '{prefix_text}'")
            # Store a simplified representation or a hash of the actual KV state
            CRITICAL_KV_CACHE_STORE[conversation_id + "_" + prefix_text] = generated_response # Store response as a proxy
        else:
            print(f"Prefix '{prefix_text}' not marked as critical for replication.")

    def recover_from_replication(self, conversation_id: str, prefix_text: str) -> str:
        """
        Simulates recovery from a replicated critical KV cache node.
        In a real system, this would involve loading serialized KV tensors
        and injecting them back into a new vLLM instance or a specific request context.
        Here, we retrieve the simplified representation.
        """
        key = conversation_id + "_" + prefix_text
        if key in CRITICAL_KV_CACHE_STORE:
            print(f"Simulating recovery for critical prefix: '{prefix_text}'")
            return CRITICAL_KV_CACHE_STORE[key]
        return None

kv_cache_manager = KVCacheManager()

@app.on_event("startup")
async def startup_event():
    global llm_model, tokenizer, embeddings_model, vector_store
    print("Initializing LLM, Embeddings, and Vector Store...")

    if VLLM_AVAILABLE:
        # Load LLM model with vLLM
        # Replace 'mistralai/Mistral-7B-Instruct-v0.2' with your preferred model
        # Ensure the model is downloaded or accessible to vLLM
        try:
            llm_model = LLM(model="mistralai/Mistral-7B-Instruct-v0.2",
                            tensor_parallel_size=1, # Adjust based on GPU setup
                            gpu_memory_utilization=0.9)
            print("vLLM LLM initialized.")
        except Exception as e:
            print(f"Failed to initialize vLLM LLM: {e}. Running vLLM in mock mode.")
            llm_model = None # Fallback to mock behavior
    else:
        print("vLLM not available, LLM will be mocked.")

    if LANGCHAIN_AVAILABLE:
        # Initialize Embeddings
        embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        print("SentenceTransformerEmbeddings initialized.")

        # Initialize ChromaDB (in-memory for demonstration)
        vector_store = Chroma.from_texts(
            texts=[
                "The latest iPhone model is the iPhone 15 Pro, featuring a A17 Bionic chip and a new titanium design. Price starts at $999.",
                "Our return policy allows returns within 30 days of purchase, provided the item is in its original condition. Some electronics have a 15-day return window.",
                "Shipping usually takes 3-5 business days for standard delivery within the US. Express shipping options are available for an extra cost.",
                "To check your order status, please log in to your account and navigate to 'My Orders', or enter your order number on our 'Track Order' page.",
                "We accept Visa, Mastercard, American Express, PayPal, and Apple Pay.",
                "Our customer support is available 24/7 via chat, email, and phone. Email us at support@ecommerce.com or call us at 1-800-123-4567.",
                "The new product line for summer includes lightweight apparel and outdoor gear.",
                "You can reset your password by clicking 'Forgot Password' on the login page and following the instructions sent to your email.",
                "Warranty for electronics is typically one year from the date of purchase, covering manufacturing defects."
            ],
            embedding=embeddings_model
        )
        print("ChromaDB vector store initialized with dummy data.")
    else:
        print("LangChain not available, Embeddings and Vector Store will be mocked.")


class ChatRequest(BaseModel):
    user_message: str
    conversation_id: str = "default_conversation" # For tracking conversations and fault tolerance
    history: List[Dict[str, str]] = [] # [{"role": "user", "content": "..."}]


class ChatResponse(BaseModel):
    bot_message: str
    conversation_id: str


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    user_message = request.user_message
    conversation_id = request.conversation_id
    chat_history = request.history

    print(f"Received message for conversation '{conversation_id}': {user_message}")

    # 1. Simulate KV Cache Recovery for Critical Prefixes
    recovered_response = kv_cache_manager.recover_from_replication(conversation_id, user_message)
    if recovered_response:
        print(f"Recovered response from simulated KV cache: {recovered_response}")
        # In a real scenario, this recovered_response might be used to prime the LLM's KV cache
        # or directly return if the query perfectly matches a cached prefix.
        # For this demo, we'll just return it directly if a perfect match is found for a critical prefix.
        # Otherwise, proceed with RAG and LLM inference.
        return ChatResponse(bot_message=recovered_response, conversation_id=conversation_id)


    # 2. RAG - Retrieve relevant context
    context = ""
    if LANGCHAIN_AVAILABLE and vector_store:
        try:
            retriever = vector_store.as_retriever()
            docs = retriever.invoke(user_message)
            context = "\n".join([doc.page_content for doc in docs])
            print(f"Retrieved context:\n{context}")
        except Exception as e:
            print(f"Error during RAG retrieval: {e}")
            context = "I couldn't retrieve specific information at this moment."
    else:
        print("RAG components not initialized. Skipping retrieval.")

    # 3. Construct Prompt with History and Context
    template = """
    You are an AI customer support assistant for an e-commerce platform.
    Answer the user's question based on the provided context and conversation history.
    If you don't know the answer, politely state that you cannot assist with that specific query.

    Conversation History:
    {history}

    Context:
    {context}

    User: {question}
    Assistant:
    """

    # Format history for the prompt
    formatted_history = "\n".join([f"{msg['role'].title()}: {msg['content']}" for msg in chat_history])

    prompt = ChatPromptTemplate.from_template(template)
    chain = (
        {"context": RunnablePassthrough() if not context else RunnableLambda(lambda x: context),
         "history": RunnableLambda(lambda x: formatted_history),
         "question": RunnablePassthrough()}
        | prompt
        | StrOutputParser()
    )

    # 4. LLM Inference using vLLM
    bot_message = "I am currently experiencing technical difficulties. Please try again later."
    if VLLM_AVAILABLE and llm_model:
        try:
            full_prompt = chain.invoke({"question": user_message})
            sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=256)
            output = llm_model.generate(prompt_or_messages=[full_prompt], sampling_params=sampling_params)
            bot_message = output[0].outputs[0].text.strip()
            print(f"LLM generated response: {bot_message}")

            # 5. Simulate KV Cache Replication for Critical Prefixes
            kv_cache_manager.replicate_critical_kv_node(conversation_id, user_message, bot_message)

        except Exception as e:
            print(f"Error during vLLM inference: {e}")
            bot_message = "I'm sorry, I'm having trouble processing your request right now due to an internal error."
    else:
        # Mock LLM response
        print("LLM not available or in mock mode. Returning mock response.")
        mock_responses = {
            "hello": "Hello! How can I assist you with your shopping today?",
            "hi": "Hi there! How can I help?",
            "order status": "To check your order status, please log in to your account or provide your order number.",
            "return policy": "Our return policy allows returns within 30 days of purchase for most items.",
            "default": f"This is a mock response to '{user_message}'. Please ensure vLLM and LangChain are installed and configured for full functionality."
        }
        bot_message = mock_responses.get(user_message.lower(), mock_responses["default"])


    return ChatResponse(bot_message=bot_message, conversation_id=conversation_id)


# --- Streamlit Frontend (conceptual client for FastAPI) ---

# This part would typically be in a separate `streamlit_app.py` file.
# Included here for the "single file" request, but should be run as:
# `streamlit run chatbot_app.py` (after starting FastAPI with `python chatbot_app.py`)
# Or, if running only Streamlit, ensure FastAPI is running separately.

# To run FastAPI: `uvicorn chatbot_app:app --host 0.0.0.0 --port 8000`
# Then, to run Streamlit: `streamlit run chatbot_app.py` (or a separate file)

# In a real setup, Streamlit would make HTTP requests to the FastAPI endpoint.
# We'll include a simple client logic here.
# Note: Streamlit part will only run if `st` is imported and `streamlit run` is used.
# If running `python chatbot_app.py`, only FastAPI will start.
if os.getenv("RUN_STREAMLIT") == "True":
    try:
        import streamlit as st
        STREAMLIT_AVAILABLE = True
    except ImportError:
        print("Streamlit not found. Please install with 'pip install streamlit'. Frontend will not run.")
        STREAMLIT_AVAILABLE = False
else:
    STREAMLIT_AVAILABLE = False


if STREAMLIT_AVAILABLE:
    st.set_page_config(page_title="E-commerce Chatbot")
    st.title("🛍️ E-commerce Customer Support Chatbot")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = "streamlit_user_" + os.urandom(8).hex()

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ask about products, orders, or policies..."):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Prepare history for the backend
        history_for_backend = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]] # Exclude current prompt

        # Send message to FastAPI backend
        try:
            with httpx.Client(base_url="http://localhost:8000") as client:
                response = client.post(
                    "/chat",
                    json={
                        "user_message": prompt,
                        "conversation_id": st.session_state.conversation_id,
                        "history": history_for_backend
                    }
                )
                response.raise_for_status()
                bot_response = ChatResponse(**response.json()).bot_message
        except httpx.RequestError as e:
            bot_response = f"Could not connect to the chatbot service. Is the FastAPI backend running? Error: {e}"
        except httpx.HTTPStatusError as e:
            bot_response = f"Error from chatbot service: {e.response.status_code} - {e.response.text}"
        except Exception as e:
            bot_response = f"An unexpected error occurred: {e}"


        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(bot_response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": bot_response})


if __name__ == "__main__":
    if os.getenv("RUN_STREAMLIT") != "True":
        print("\n--- Starting FastAPI Backend ---")
        print("To run the Streamlit frontend, set the environment variable RUN_STREAMLIT=True and run 'streamlit run chatbot_app.py' in a separate terminal.")
        print("FastAPI will be available at http://localhost:8000")
        print("Swagger UI: http://localhost:8000/docs")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        # Streamlit will handle its own execution if RUN_STREAMLIT is True
        print("--- Streamlit Frontend will start ---")
        print("Ensure FastAPI backend is running on http://localhost:8000")
        # No uvicorn.run here, Streamlit takes over the main execution when run with `streamlit run`
        # The streamlit part above will be executed by `streamlit run`.
