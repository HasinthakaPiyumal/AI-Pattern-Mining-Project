import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor, Tool, initialize_agent, AgentType
from langchain.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv
import os

load_dotenv()

# --- Custom Tools ---
class KnowledgeBaseSearchTool:
    def run(self, query: str) -> str:
        # Simulate searching a knowledge base
        knowledge_base = {
            "password reset": "To reset your password, visit our website and click 'Forgot Password'. Follow the instructions.",
            "billing inquiry": "For billing inquiries, please check your account dashboard or contact our billing department directly at billing@example.com.",
            "technical issue": "Please describe your technical issue in more detail so I can assist you better. Provide error messages or steps to reproduce.",
            "product features": "Our product offers features like A, B, and C. For a full list, visit our product page.",
            "shipping status": "Please provide your order number to check the shipping status."
        }
        query_lower = query.lower()
        for keyword, response in knowledge_base.items():
            if keyword in query_lower:
                return response
        return "I couldn't find a direct answer in our knowledge base. Would you like me to try rephrasing or escalate to a human?"

class EscalateToHumanTool:
    def run(self, issue_description: str) -> str:
        return f"Okay, I'm escalating your issue: '{issue_description}' to a human agent. Please hold while I connect you."

# --- Agent Initialization ---
def initialize_support_agent():
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=os.getenv("OPENAI_API_KEY"))

    tools = [
        Tool(
            name="KnowledgeBaseSearch",
            func=KnowledgeBaseSearchTool().run,
            description="Useful for searching the knowledge base for common customer issues and solutions."
        ),
        Tool(
            name="EscalateToHuman",
            func=EscalateToHumanTool().run,
            description="Useful when the agent cannot resolve the issue and needs to escalate to a human support agent."
        )
    ]

    # Using ConversationBufferWindowMemory to maintain a memory of recent interactions
    memory = ConversationBufferWindowMemory(memory_key="chat_history", k=5, return_messages=True)

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
        handle_parsing_errors=True
    )
    return agent

# --- Streamlit UI ---
st.set_page_config(page_title="Intelligent Customer Support Agent", layout="centered")
st.title("🤖 Intelligent Customer Support Agent")

if "agent" not in st.session_state:
    st.session_state.agent = initialize_support_agent()
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("How can I help you today?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.agent.run(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_message = f"An error occurred: {e}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
