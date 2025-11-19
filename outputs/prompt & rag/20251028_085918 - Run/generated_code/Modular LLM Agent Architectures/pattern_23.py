import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.tools import Tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

# --- 1. LLM Core and Embeddings ---
llm = ChatOpenAI(model="gpt-4o", temperature=0)
embeddings = OpenAIEmbeddings()

# --- 2. Tool Execution Module ---
def get_customer_info(customer_id: str) -> str:
    if customer_id == "CUST123":
        return "Customer CUST123: John Doe, Email: john.doe@example.com, Phone: 555-1234, Membership: Premium"
    return f"Customer with ID {customer_id} not found."

def get_order_status(order_id: str) -> str:
    if order_id == "ORD456":
        return "Order ORD456: Status: Shipped, Tracking: TRACK789, Estimated Delivery: 2024-07-25"
    elif order_id == "ORD789":
        return "Order ORD789: Status: Processing, Expected Ship Date: 2024-07-20"
    return f"Order with ID {order_id} not found."

# Create LangChain tools
tools = [
    Tool(
        name="GetCustomerInfo",
        func=get_customer_info,
        description="Useful for getting detailed information about a customer given their customer ID."
    ),
    Tool(
        name="GetOrderStatus",
        func=get_order_status,
        description="Useful for getting the current status of an order given an order ID."
    ),
]

# --- 3. Knowledge Retrieval Module (RAG) ---
# Dummy documents for the knowledge base
docs = [
    "Our return policy allows returns within 30 days of purchase for a full refund. Items must be in original condition.",
    "For exchanges, please contact our support team within 60 days of purchase. We offer even exchanges for items of equal value.",
    "Shipping usually takes 3-5 business days for standard delivery. Expedited shipping options are available at checkout.",
    "We accept major credit cards, PayPal, and Apple Pay.",
    "To reset your password, visit our website and click 'Forgot Password' on the login page."
]

# Initialize Chroma vector store
vectorstore = Chroma.from_documents(
    documents=[{"page_content": doc} for doc in docs], # Chroma expects dicts for docs
    embedding=embeddings,
    collection_name="customer_support_kb",
    persist_directory="./chroma_db"
)
retriever = vectorstore.as_retriever()

# Add retriever as a tool
tools.append(Tool(
    name="KnowledgeBase",
    func=lambda query: retriever.invoke(query),
    description="Useful for answering questions about company policies, products, and general information. Input should be a question."
))

# --- 4. Conversational Memory Module ---
memory = ConversationBufferWindowMemory(k=5, return_messages=True, memory_key="chat_history")

# --- 5. Planning Module & Agent Orchestration ---
# Define the prompt for the agent
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful customer support agent for an e-commerce company. Answer user questions thoroughly and accurately. If a tool requires an ID, try to ask the user for it if not provided."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# Create the agent
agent = create_tool_calling_agent(llm, tools, prompt)

# Create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- 6. Integrate Memory with Agent Executor ---
# This will store chat history in the session
def get_session_history(session_id: str) -> ConversationBufferWindowMemory:
    return memory

with_message_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# --- 7. User Interface (CLI) ---
print("Welcome to the Smart Customer Support Agent! Type 'exit' to quit.")

session_id = "user_session_1" # For a single user CLI, we can use a fixed session ID

while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Agent: Goodbye!")
        break
    
    try:
        response = with_message_history.invoke(
            {"input": user_input},
            config={
                "configurable": {"session_id": session_id}
            },
        )
        print(f"Agent: {response['output']}")
    except Exception as e:
        print(f"Agent Error: An error occurred - {e}")
        print("Agent: I apologize, I encountered an issue. Please try again or rephrase your request.")
