import uvicorn
import asyncio
import uuid
import pickle
import os
import time
from collections import defaultdict

# --- FastAPI Backend (Server) ---
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("chatbot_backend")

# --- Configuration ---
PERSISTENT_KV_CACHE_PATH = "critical_kv_cache.pkl"
REPLICATION_INTERVAL_SECONDS = 30
MOCK_VLLM_RESPONSE_DELAY = 1 # Simulate LLM processing time

class MockVLLM:
    """
    A mock vLLM engine to simulate LLM inference and KV cache operations.
    In a real application, this would be a client to a running vLLM server.
    """
    def __init__(self, model_name="Mock Llama 2"):
        self.model_name = model_name
        self._kv_cache_replication_data = {"last_replicated_state": f"Initial KV state for {model_name}"}
        self._evicted_kv_pages = set() # Stores identifiers of KV pages already swapped out
        logger.info(f"MockVLLM initialized with model: {self.model_name}")

    async def generate(self, prompt: str, session_id: str):
        """
        Simulates LLM generation.
        Includes conceptual logging for KV Cache Reuse and PagedAttention.
        """
        logger.info(f"MockVLLM: Generating response for session '{session_id}' with prompt: '{prompt[:50]}...'")
        
        # Simulate KV Cache Reuse for common prefixes
        if "hello" in prompt.lower() or "hi" in prompt.lower():
            logger.info("MockVLLM: Detected common prefix, conceptually reusing KV cache.")
        else:
            logger.info("MockVLLM: No common prefix detected for KV cache reuse.")

        # Simulate PagedAttention benefit (memory efficiency)
        if len(prompt) > 100:
            logger.info("MockVLLM: Long prompt detected, PagedAttention conceptually managing memory efficiently.")
        else:
            logger.info("MockVLLM: Short prompt, PagedAttention still active for page-level management.")

        # Simulate the 'Swap-Out-Only-Once' strategy (conceptual)
        # For simplicity, let's say a 'page' is generated per user prompt.
        current_kv_page_id = f"session_{session_id}_prompt_{hash(prompt)}"
        if current_kv_page_id not in self._evicted_kv_pages:
            # Simulate a new eviction to host memory
            self._evicted_kv_pages.add(current_kv_page_id)
            logger.info(f"MockVLLM: KV page '{current_kv_page_id[:30]}...' conceptually swapped out to host memory for the first time.")
        else:
            logger.info(f"MockVLLM: KV page '{current_kv_page_id[:30]}...' already swapped out, avoiding re-copy.")


        await asyncio.sleep(MOCK_VLLM_RESPONSE_DELAY) # Simulate inference time
        response = f"This is a mock response from {self.model_name} for your query: '{prompt}'. " \
                   f"Session ID: {session_id}. I am designed to offer fast and reliable E-commerce support."
        logger.info(f"MockVLLM: Generated response for session '{session_id}'.")
        return response

    def get_critical_kv_cache_state(self):
        """
        Simulates getting critical KV cache nodes for replication.
        In a real scenario, this would involve accessing vLLM's internal state.
        """
        # Update the state to reflect recent activity
        self._kv_cache_replication_data["last_replicated_state"] = f"KV state updated at {time.ctime()}"
        self._kv_cache_replication_data["active_sessions_count"] = len(session_store)
        return self._kv_cache_replication_data

    def load_critical_kv_cache_state(self, state):
        """
        Simulates loading critical KV cache nodes.
        """
        self._kv_cache_replication_data = state
        logger.info(f"MockVLLM: Critical KV cache state loaded: {state.get('last_replicated_state', 'N/A')}")


class ChatRequest(BaseModel):
    session_id: str = None
    query: str

class ChatResponse(BaseModel):
    session_id: str
    response: str
    chat_history: list

app = FastAPI(
    title="E-commerce Chatbot Backend",
    description="Intelligent customer support chatbot powered by optimized LLM inference."
)

mock_vllm_engine = MockVLLM()
session_store = defaultdict(list) # Stores conversation history: {session_id: [("user", query), ("bot", response)]}

async def replicate_critical_kv_cache_nodes():
    """
    Background task to periodically replicate critical KV cache nodes to persistent storage.
    """
    logger.info("Starting background task for critical KV cache replication.")
    while True:
        try:
            critical_state = mock_vllm_engine.get_critical_kv_cache_state()
            with open(PERSISTENT_KV_CACHE_PATH, "wb") as f:
                pickle.dump(critical_state, f)
            logger.info(f"Critical KV cache replicated to '{PERSISTENT_KV_CACHE_PATH}'. State: {critical_state.get('last_replicated_state', 'N/A')}")

            # Simulate recovery if a previous state exists on startup
            if os.path.exists(PERSISTENT_KV_CACHE_PATH) and not hasattr(app.state, "kv_cache_loaded"):
                with open(PERSISTENT_KV_CACHE_PATH, "rb") as f:
                    loaded_state = pickle.load(f)
                    mock_vllm_engine.load_critical_kv_cache_state(loaded_state)
                    app.state.kv_cache_loaded = True # Mark as loaded to prevent repeated loading
                    logger.info("Successfully loaded critical KV cache on startup (simulated recovery).")

        except Exception as e:
            logger.error(f"Error during KV cache replication: {e}", exc_info=True)
        await asyncio.sleep(REPLICATION_INTERVAL_SECONDS)

@app.on_event("startup")
async def startup_event():
    """
    On startup, try to load any previously replicated KV cache and start the replication task.
    """
    logger.info("FastAPI startup event triggered.")
    # Try to load critical KV cache at startup for simulated fast recovery
    if os.path.exists(PERSISTENT_KV_CACHE_PATH):
        try:
            with open(PERSISTENT_KV_CACHE_PATH, "rb") as f:
                loaded_state = pickle.load(f)
                mock_vllm_engine.load_critical_kv_cache_state(loaded_state)
                app.state.kv_cache_loaded = True
                logger.info(f"Successfully loaded critical KV cache from '{PERSISTENT_KV_CACHE_PATH}' during startup.")
        except Exception as e:
            logger.warning(f"Could not load critical KV cache during startup: {e}")
    else:
        logger.info("No existing critical KV cache found to load at startup.")

    # Start the background replication task
    asyncio.create_task(replicate_critical_kv_cache_nodes())


@app.get("/")
async def read_root():
    return {"message": "E-commerce Chatbot Backend is running!"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    session_id = request.session_id
    if not session_id:
        session_id = str(uuid.uuid4())
        logger.info(f"New session created: {session_id}")

    user_query = request.query
    session_store[session_id].append(("user", user_query))
    logger.info(f"Session {session_id}: User query received: {user_query[:50]}...")

    try:
        # Simulate LLM inference
        bot_response = await mock_vllm_engine.generate(user_query, session_id)
        session_store[session_id].append(("bot", bot_response))
        logger.info(f"Session {session_id}: Bot response sent: {bot_response[:50]}...")

        return ChatResponse(
            session_id=session_id,
            response=bot_response,
            chat_history=session_store[session_id]
        )
    except Exception as e:
        logger.error(f"Error during chat processing for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred while processing your request.")

# --- Streamlit Frontend (Client) ---
# This part would typically be in a separate file (e.g., `streamlit_app.py`)
# but is included here as per the "all files in single code" request.
# To run this, save the entire content as `app.py` and then run:
# 1. `uvicorn app:app --reload` in one terminal (for FastAPI backend)
# 2. `streamlit run app.py` in another terminal (for Streamlit frontend)

import streamlit as st
import requests
import json # Used for pretty printing JSON for debugging if needed

# Frontend Configuration
BACKEND_URL = "http://localhost:8000"

def streamlit_frontend_code():
    st.set_page_config(page_title="E-commerce Support Chatbot", layout="centered")
    st.title("🛍️ E-commerce Customer Support Chatbot")
    st.markdown(
        """
        Welcome to our intelligent customer support chatbot!
        Ask me anything about our products, orders, or services.
        This chatbot leverages advanced LLM inference optimizations for speed and reliability.
        """
    )

    # Initialize session state for chat history and session_id
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = None

    # Display chat messages
    for speaker, message in st.session_state.chat_history:
        if speaker == "user":
            with st.chat_message("user"):
                st.markdown(message)
        else:
            with st.chat_message("assistant"):
                st.markdown(message)

    # Chat input
    if prompt := st.chat_input("How can I help you today?"):
        st.session_state.chat_history.append(("user", prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response_data = {
                        "session_id": st.session_state.session_id,
                        "query": prompt
                    }
                    response = requests.post(f"{BACKEND_URL}/chat", json=response_data)
                    response.raise_for_status() # Raise an exception for bad status codes
                    
                    chat_response = response.json()
                    bot_message = chat_response["response"]
                    st.session_state.session_id = chat_response["session_id"]
                    st.session_state.chat_history = chat_response["chat_history"]

                    st.markdown(bot_message)
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to the chatbot backend. Please ensure the backend server is running.")
                except requests.exceptions.RequestException as e:
                    st.error(f"An error occurred: {e}")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")

    # Optional: Display session ID for debugging
    if st.session_state.session_id:
        st.sidebar.info(f"Current Session ID: {st.session_state.session_id}")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **Backend Optimizations (Conceptual):**
        - **KV Cache Reuse:** For common query prefixes.
        - **PagedAttention:** Efficient KV memory management.
        - **Swap-Out-Only-Once:** Reduces data transfer for evicted KV pages.
        - **Critical KV Cache Replication:** For fast recovery from failures.
        """
    )


# --- How to run this code ---
#
# This single file contains both the FastAPI backend and Streamlit frontend code.
#
# To run the *FastAPI Backend*:
# 1. Save this file as `chatbot_app.py` (or any `.py` name).
# 2. Open your terminal and navigate to the directory where you saved the file.
# 3. Install necessary libraries:
#    `pip install fastapi uvicorn 'pydantic<2' 'uvicorn[standard]' requests streamlit`
#    (Note: pydantic<2 is to ensure compatibility with older FastAPI versions if issues arise, though latest should be fine)
# 4. Run the FastAPI server:
#    `uvicorn chatbot_app:app --reload`
#    (The `--reload` flag is optional but useful for development)
#    You should see output indicating the server is running on `http://127.0.0.1:8000`.
#
# To run the *Streamlit Frontend*:
# 1. Make sure the FastAPI Backend is already running as described above.
# 2. Open *another* terminal (keep the FastAPI terminal running).
# 3. Navigate to the same directory.
# 4. Run the Streamlit application:
#    `streamlit run chatbot_app.py`
#    Streamlit will open a new tab in your web browser, typically at `http://localhost:8501`.
#
# **Explanation for "all files in single code":**
# The Python interpreter will execute the code sequentially.
# When `uvicorn chatbot_app:app` is run, it imports `app` (the FastAPI instance)
# and starts the server. The `if __name__ == "__main__":` block for Streamlit
# is NOT executed in this context.
#
# When `streamlit run chatbot_app.py` is run, the entire script is executed.
# However, Streamlit specifically looks for Streamlit commands (`st.`) to build the UI.
# The FastAPI `app` object is defined, but `uvicorn.run()` is inside a condition
# that is not met when `streamlit run` executes the script.
#
# Therefore, by using the `if __name__ == "__main__":` block to conditionally
# run `uvicorn.run()` (which would block the script) and defining the Streamlit
# code as a function `streamlit_frontend_code()` that is called within a similar
# conditional block, we can achieve the "single file" request while maintaining
# runnable components that are launched separately.
#
# In this specific implementation, `uvicorn.run()` is not explicitly called within
# `if __name__ == "__main__":` for the FastAPI part because `uvicorn app:app`
# directly imports and runs it. The `streamlit_frontend_code()` is called if
# the script is executed as a Streamlit app.

if __name__ == "__main__":
    # This block will be executed if the script is run directly (e.g., `python chatbot_app.py`)
    # However, to run the FastAPI app, you typically use `uvicorn chatbot_app:app`
    # and to run Streamlit, you use `streamlit run chatbot_app.py`.
    #
    # The current setup allows Streamlit to run successfully when `streamlit run` is used,
    # as the `streamlit_frontend_code()` function is defined and then called.
    # The FastAPI `app` object is still defined and accessible for `uvicorn`.
    #
    # To demonstrate running the Streamlit part from here if not launched via `streamlit run`,
    # one might add `streamlit_frontend_code()` call here.
    # But for the typical deployment, `streamlit run` handles the execution.
    #
    # The key is that `uvicorn` and `streamlit` are separate entry points for the same file.
    # The user instruction explains how to run them separately.
    logger.info("Script executed directly. This is typically for Streamlit or for testing.")
    if "streamlit" in os.environ.get("STREAMLIT_SERVER_CMDLINE", ""):
        # This is a heuristic to check if running via `streamlit run`
        streamlit_frontend_code()
    else:
        # If run directly without streamlit, it might be for backend testing or other purposes
        # For actual uvicorn server, it's `uvicorn chatbot_app:app`
        logger.info("To run the FastAPI backend, use: `uvicorn chatbot_app:app --reload`")
        logger.info("To run the Streamlit frontend, use: `streamlit run chatbot_app.py`")