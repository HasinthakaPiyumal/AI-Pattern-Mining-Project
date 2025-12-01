import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import streamlit as st
import time

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

class AgentAssistRequest(BaseModel):
    customer_issue: str

class MockVLLM:
    def __init__(self, model_name: str = "mock-llm", openai_api_base: str = "http://localhost:8000/v1"):
        self.model_name = model_name
        self.openai_api_base = openai_api_base

    def invoke(self, prompt: str) -> str:
        if "summarize the customer issue" in prompt.lower():
            issue_part = prompt.split("issue: ")[-1]
            return f"Summary: Customer concern is about {issue_part.split(' and suggest')[0].strip()}. Suggested actions: Check return policy, provide shipping updates, escalate to specialized agent."
        return f"AI Assistant: You mentioned '{prompt}'. How can I further assist you with this?"

llm = MockVLLM()

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        response = llm.invoke(request.message)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agent-assist")
async def agent_assist_endpoint(request: AgentAssistRequest):
    try:
        prompt = f"Summarize the customer issue: {request.customer_issue} and suggest 2-3 responses."
        agent_response = llm.invoke(prompt)
        return {"summary_and_suggestions": agent_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def run_streamlit_app():
    st.set_page_config(page_title="Intelligent Customer Support")
    st.title("📞 Intelligent Customer Support Platform")

    st.header("Chat with AI Assistant")
    user_query = st.text_area("Your Message:", key="chat_input")
    if st.button("Send Message", key="send_chat"):
        if user_query:
            st.info("Sending message to AI...")
            try:
                response = requests.post("http://127.0.0.1:8000/chat", json={"message": user_query})
                if response.status_code == 200:
                    st.text_area("AI Response:", value=response.json().get("response", "Error getting response"), height=100, disabled=True)
                else:
                    st.error(f"Error from backend: {response.status_code} - {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to FastAPI backend. Please ensure the backend is running on http://127.0.0.1:8000.")
        else:
            st.warning("Please enter a message.")

    st.markdown("---")

    st.header("Agent Assist Tool")
    customer_issue = st.text_area("Customer Issue Description:", key="agent_input")
    if st.button("Get Agent Assist", key="get_assist"):
        if customer_issue:
            st.info("Generating agent assist...")
            try:
                response = requests.post("http://127.0.0.1:8000/agent-assist", json={"customer_issue": customer_issue})
                if response.status_code == 200:
                    st.text_area("Agent Assist (Summary & Suggestions):", value=response.json().get("summary_and_suggestions", "Error getting suggestions"), height=200, disabled=True)
                else:
                    st.error(f"Error from backend: {response.status_code} - {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to FastAPI backend. Please ensure the backend is running on http://127.0.0.1:8000.")
        else:
            st.warning("Please enter a customer issue.")

