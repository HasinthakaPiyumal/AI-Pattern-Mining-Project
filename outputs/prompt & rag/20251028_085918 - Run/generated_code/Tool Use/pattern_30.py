import streamlit as st
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory

import os

# Mock API Wrappers
@tool
def get_order_details(order_id: str) -> str:
    if order_id == "ORD123":
        return "Order ORD123: Product A, Quantity 1, Status: Shipped, Estimated Delivery: 2024-08-10"
    return f"Order {order_id} not found."

@tool
def get_delivery_status(tracking_id: str) -> str:
    if tracking_id == "TRK456":
        return "Tracking TRK456: Package is currently in transit, last scanned in New York."
    return f"Tracking ID {tracking_id} not found."

@tool
def check_payment_status(transaction_id: str) -> str:
    if transaction_id == "PAY789":
        return "Transaction PAY789: Status: Completed, Amount: $50.00."
    return f"Transaction ID {transaction_id} not found."

@tool
def process_refund(transaction_id: str, amount: float) -> str:
    if transaction_id == "PAY789" and amount <= 50.00:
        return f"Refund of ${amount} for transaction {transaction_id} successfully initiated."
    return f"Failed to process refund for transaction {transaction_id}. Invalid transaction or amount."

# Knowledge Base Interface
def setup_knowledge_base():
    documents = [
        "FAQ: Our return policy allows returns within 30 days of purchase with a valid receipt.",
        "FAQ: To reset your password, visit our website and click 'Forgot Password'.",
        "Policy: All electronics come with a 1-year warranty from the date of purchase.",
        "Policy: Shipping usually takes 3-5 business days for domestic orders.",
    ]
    persist_directory = "./chroma_db"
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)
    
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma.from_texts(documents, embeddings, persist_directory=persist_directory)
    return vectordb

vectordb = setup_knowledge_base()
retriever = vectordb.as_retriever()

# Internal Scripting Tool (Simple Python Interpreter)
@tool
def execute_python_script(script: str) -> str:
    try:
        # WARNING: This is a simplified, UNSAFE implementation for demonstration.
        # In a real application, a properly sandboxed environment (e.g., using ` RestrictedPython ` or a dedicated microservice) is CRITICAL.
        global_vars = {}
        local_vars = {}
        exec(script, global_vars, local_vars)
        return str(local_vars.get("result", "Script executed, no explicit result."))
    except Exception as e:
        return f"Error executing script: {e}"

# Initialize LLM and Agent
llm = ChatOpenAI(temperature=0, model_name="gpt-4", openai_api_key=os.environ.get("OPENAI_API_KEY")) # Replace with your actual API key or model

# Create a RAG chain for the knowledge base
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

@tool
def query_knowledge_base(query: str) -> str:
    return qa_chain.invoke({"query": query})["result"]

# Define the tools the agent can use
tools = [
    get_order_details,
    get_delivery_status,
    check_payment_status,
    process_refund,
    query_knowledge_base,
    execute_python_script
]

# Prompt for the agent
prompt_template = PromptTemplate.from_template(
    """You are an intelligent customer support agent. You have access to various tools to help customers.
    Answer the user's questions as accurately and helpfully as possible.

    TOOLS:
    {tools}

    Remember to use the appropriate tools to find the information needed.
    If a customer asks a question that can be answered by the knowledge base, use the `query_knowledge_base` tool.
    If they ask about order details, use `get_order_details`.
    If they ask about delivery status, use `get_delivery_status`.
    If they ask about payment status, use `check_payment_status`.
    If they ask to process a refund, use `process_refund` (ensure you have transaction ID and amount).
    If a simple calculation or data manipulation is needed, use `execute_python_script`.

    Always think step-by-step before using a tool.

    {agent_scratchpad}

    Human: {input}
    Chat History: {chat_history}
    AI:"""
)

# Create the agent
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
agent = create_react_agent(llm, tools, prompt_template)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory, handle_parsing_errors=True)

# Streamlit UI
st.title("Intelligent Customer Support Agent")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How can I help you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = agent_executor.invoke({"input": prompt})
            st.markdown(response["output"])
            st.session_state.messages.append({"role": "assistant", "content": response["output"]})
