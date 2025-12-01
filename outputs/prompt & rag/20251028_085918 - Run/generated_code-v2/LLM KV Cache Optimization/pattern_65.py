import streamlit as st
from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn
import threading
import requests
import os
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

# --- FastAPI Backend Setup ---
app = FastAPI()

# Mock LLM for local testing if vLLM server is not available
class MockLLM:
    def __init__(self):
        pass
    def __call__(self, prompt: str) -> str:
        return f"Mock response for: {prompt[:100]}..."

mock_llm_instance = MockLLM()

# In a real scenario, this would be your connection to the vLLM server
# For simplicity, we'll use requests.post to a hypothetical vLLM endpoint
# You would run vLLM separately, e.g., `python -m vllm.entrypoints.api_server --model your/model --port 8000`
vllm_api_url = os.getenv("VLLM_API_URL", "http://localhost:8000/generate")

def get_llm_response(prompt: str) -> str:
    try:
        headers = {"Content-Type": "application/json"}
        data = {"prompt": prompt, "max_tokens": 150, "temperature": 0.7}
        response = requests.post(vllm_api_url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        # vLLM's /generate endpoint returns a list of outputs
        if result and "text" in result["outputs"][0]:
            return result["outputs"][0]["text"]
        return "No text generated."
    except requests.exceptions.ConnectionError:
        print("vLLM server not reachable. Using mock LLM.")
        return mock_llm_instance(prompt)
    except Exception as e:
        print(f"Error calling vLLM: {e}. Using mock LLM.")
        return mock_llm_instance(prompt)


# LangChain memory and chain setup
# Using ConversationBufferWindowMemory for simplicity, storing last 5 turns
# In a real app, RedisChatMessageHistory or similar would be used for persistence.
memory = ConversationBufferWindowMemory(k=5)

_DEFAULT_TEMPLATE = """The following is a friendly conversation between a human and an AI.
The AI is helpful, courteous, and provides customer support.

{history}
Human: {input}
AI:"""
PROMPT = PromptTemplate(input_variables=["history", "input"], template=_DEFAULT_TEMPLATE)

# Custom LLM Chain to integrate with our get_llm_response function
class CustomLLMChain(LLMChain):
    def _call(self, inputs: dict) -> dict:
        full_prompt = PROMPT.format(**inputs)
        response_text = get_llm_response(full_prompt)
        return {"text": response_text}

llm_chain = CustomLLMChain(prompt=PROMPT, llm=mock_llm_instance, memory=memory) # mock_llm_instance is a placeholder, CustomLLMChain calls get_llm_response

class ChatMessage(BaseModel):
    message: str
    session_id: str = "default_session"

@app.post("/chat")
async def chat_endpoint(msg: ChatMessage):
    # For demonstration, we'll store history in the global memory object
    # For multiple users, session_id would be used to retrieve specific history from a persistent store (e.g., Redis)
    # We're simulating shared prefixes by just feeding the full conversation history to LangChain.
    # vLLM's internal mechanisms then handle the KV cache reuse for common parts of that history.
    
    # Add current message to memory before prediction
    # Note: LangChain's memory manages the 'history' part of the prompt
    
    response = llm_chain.predict(input=msg.message)
    return {"response": response}

# --- Streamlit Frontend Setup ---
st.set_page_config(page_title="KV Cache Reuse Chatbot")
st.title("Intelligent Customer Support Chatbot")
st.subheader("Leveraging KV Cache Reuse for LLM Efficiency")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = os.urandom(16).hex()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How can I help you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.spinner("Thinking..."):
        try:
            # Send message to FastAPI backend
            fastapi_url = os.getenv("FASTAPI_URL", "http://localhost:8001/chat")
            data = {"message": prompt, "session_id": st.session_state.session_id}
            response = requests.post(fastapi_url, json=data)
            response.raise_for_status()
            ai_response = response.json()["response"]
        except requests.exceptions.ConnectionError:
            ai_response = "Error: Could not connect to the backend server. Please ensure FastAPI is running."
        except Exception as e:
            ai_response = f"An error occurred: {e}"

    st.session_state.messages.append({"role": "assistant", "content": ai_response})
    with st.chat_message("assistant"):
        st.markdown(ai_response)


# --- How to run (Instructions will be in explanation) ---
# To run the FastAPI server:
# uvicorn chatbot_app:app --host 0.0.0.0 --port 8001
#
# To run the Streamlit frontend:
# streamlit run chatbot_app.py
#
# To run vLLM server (separately):
# python -m vllm.entrypoints.api_server --model <your-model-name> --tensor-parallel-size <num-gpus> --port 8000