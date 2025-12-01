from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import uvicorn
from vllm import LLM, SamplingParams
import asyncio

app = FastAPI()

llm = LLM(model="HuggingFaceH4/zephyr-7b-beta", trust_remote_code=True)

class ChatRequest(BaseModel):
    user_message: str
    chat_history: List[List[str]]

sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512)

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    conversation_history = ""
    for user_msg, bot_resp in request.chat_history:
        conversation_history += f"User: {user_msg}\nAssistant: {bot_resp}\n"
    
    full_prompt = f"{conversation_history}User: {request.user_message}\nAssistant:"

    outputs = llm.generate([full_prompt], sampling_params)
    
    if outputs and outputs[0].outputs:
        generated_text = outputs[0].outputs[0].text.strip()
        return {"response": generated_text}
    else:
        return {"response": "Sorry, I couldn't generate a response."}

import streamlit as st
import requests
import json

FASTAPI_URL = "http://localhost:8000/chat"

st.set_page_config(page_title="KV Cache Optimized Chatbot")
st.title("Customer Support Chatbot (KV Cache Optimized)")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How can I help you today?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    chat_history_for_backend = []
    temp_user_msg = None
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            temp_user_msg = msg["content"]
        elif msg["role"] == "assistant" and temp_user_msg is not None:
            chat_history_for_backend.append([temp_user_msg, msg["content"]])
            temp_user_msg = None
    
    if temp_user_msg is not None and temp_user_msg == prompt:
         pass

    try:
        payload = {
            "user_message": prompt,
            "chat_history": chat_history_for_backend
        }
        headers = {"Content-Type": "application/json"}
        
        with st.spinner("Thinking..."):
            response = requests.post(FASTAPI_URL, data=json.dumps(payload), headers=headers)
            response.raise_for_status()
            bot_response = response.json().get("response", "Error: No response from chatbot.")

        with st.chat_message("assistant"):
            st.markdown(bot_response)
        st.session_state.messages.append({"role": "assistant", "content": bot_response})

    except requests.exceptions.ConnectionError:
        st.error(f"Could not connect to the FastAPI backend. Make sure it's running at {FASTAPI_URL}.")
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred: {e}")