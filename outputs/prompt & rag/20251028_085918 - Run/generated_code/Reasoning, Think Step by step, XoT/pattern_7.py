
import streamlit as st
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.retrievers import MultiQueryRetriever
from langchain.agents import AgentExecutor, create_react_agent, tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
import os

# --- Configuration --- #
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
# You might need to initialize ChromaDB with your actual medical knowledge base
# For demonstration, we'll use a dummy in-memory ChromaDB

# --- Embedding Model --- #
@st.cache_resource
def get_embedding_function():
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = get_embedding_function()

# --- ChromaDB (Dummy for demonstration) --- #
# In a real application, you would load your pre-built medical knowledge base
@st.cache_resource
def get_chroma_db():
    # This is a dummy for demonstration. Replace with your actual medical data.
    sample_docs = [
        "Symptoms of common cold include runny nose, sore throat, cough, and congestion.",
        "Influenza (flu) symptoms are similar to a cold but often more severe, with fever, body aches, and fatigue.",
        "Diabetes mellitus is a chronic condition that affects how your body turns food into energy. Symptoms include increased thirst, frequent urination, and unexplained weight loss.",
        "Hypertension, or high blood pressure, often has no symptoms but can lead to serious health problems like heart disease and stroke.",
        "Asthma is a condition in which your airways narrow and swell and may produce extra mucus. This can make breathing difficult and trigger coughing, a whistling sound (wheezing) when you breathe out, and shortness of breath."
    ]
    from langchain.schema import Document
    docs = [Document(page_content=d) for d in sample_docs]
    db = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db_medical")
    return db

vectordb = get_chroma_db()

# --- LLM Setup --- #
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2)

# --- RAG Tool --- #
def retrieve_medical_info(query: str) -> str:
    """Searches the medical knowledge base for information relevant to the query."""
    retriever = vectordb.as_retriever()
    docs = retriever.invoke(query)
    return "\n".join([doc.page_content for doc in docs])

medical_info_tool = Tool(
    name="medical_knowledge_retriever",
    func=retrieve_medical_info,
    description="Useful for retrieving factual medical information from a knowledge base. Input should be a concise medical query."
)

# --- Chain-of-Thought (CoT) & Verification Agent --- #
# This agent will simulate CoT and verification steps

COT_VERIFICATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a medical diagnostic assistant. Your goal is to provide a differential diagnosis and reasoning based on patient symptoms. Always think step-by-step and verify information using the available tools. Be cautious and state limitations."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

# Define tools for the agent
tools = [medical_info_tool]

agent = create_react_agent(llm, tools, COT_VERIFICATION_PROMPT)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# --- Streamlit Application --- #
st.set_page_config(page_title="Medical Diagnosis Assistant", layout="wide")
st.title("🩺 Medical Diagnosis Assistant with Enhanced Reasoning")

st.markdown("This assistant uses advanced AI reasoning and verification to provide potential diagnoses based on your symptoms. **This is for informational purposes only and not a substitute for professional medical advice.**")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def format_chat_history(messages):
    formatted = []
    for msg in messages:
        if msg["role"] == "user":
            formatted.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            formatted.append(AIMessage(content=msg["content"]))
    return formatted

# React to user input
if prompt := st.chat_input("Describe the patient's symptoms or ask a medical question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking and verifying..."):
            try:
                # Format history for the agent
                chat_history_for_agent = format_chat_history(st.session_state.messages[:-1]) # Exclude current prompt
                response = agent_executor.invoke({"input": prompt, "chat_history": chat_history_for_agent})
                full_response = response["output"]
            except Exception as e:
                full_response = f"An error occurred: {e}. Please try again or simplify your query."
                st.error(full_response)

            st.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

st.sidebar.header("Disclaimer")
st.sidebar.info("This AI tool is designed to assist with medical information and potential diagnoses based on the provided symptoms and a limited knowledge base. It is **not a substitute for professional medical advice, diagnosis, or treatment**. Always seek the advice of a qualified healthcare provider for any medical concerns.")
st.sidebar.info("Knowledge base is dummy for demonstration. In a real application, it would be a comprehensive and regularly updated medical database.")

