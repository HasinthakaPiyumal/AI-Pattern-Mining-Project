import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from tools import medical_search, medical_database_query, medical_knowledge_graph_lookup

# Load environment variables (e.g., OPENAI_API_KEY)
from dotenv import load_dotenv
load_dotenv()

# Streamlit UI
st.set_page_config(page_title="Medical Diagnostic Assistant", layout="wide")
st.title("🧠 Medical Diagnostic Assistant")
st.markdown("This assistant leverages an LLM augmented with external medical tools to provide diagnostic support and treatment recommendations.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Define the tools
tools = [medical_search, medical_database_query, medical_knowledge_graph_lookup]

# Create the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful Medical Diagnostic Assistant. You have access to specialized medical tools to provide accurate and up-to-date information. Always strive to provide detailed and evidence-based responses. If a specific medical condition is mentioned, try to use the medical knowledge graph tool to provide contextual information. If recent information is needed, use the medical search tool. If specific guidelines or structured data are required, use the medical database query tool."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# Create the agent
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def invoke_agent(query):
    with st.spinner("Thinking..."):
        response = agent_executor.invoke({"input": query, "chat_history": []})
        return response["output"]

if prompt_input := st.chat_input("How can I assist you with medical information today?"):
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    response = invoke_agent(prompt_input)

    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
