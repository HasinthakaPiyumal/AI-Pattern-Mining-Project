import os
from contextlib import asynccontextmanager
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_core.documents import Document

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


class Settings(BaseSettings):
    openai_api_key: str
    model_name: str = "gpt-3.5-turbo"
    embedding_model_name: str = "all-MiniLM-L6-v2"
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()


def get_llm():
    return ChatOpenAI(
        model=settings.model_name,
        openai_api_key=settings.openai_api_key,
        temperature=0.7
    )


def get_memory():
    return ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True, 
        output_key="output"
    )


class KnowledgeBase:
    def __init__(self):
        self.vectorstore = None
        self.retriever = None

    def initialize(self):
        docs_content = [
            "Our return policy states that items can be returned within 30 days of purchase with a valid receipt. Customized items are non-refundable.",
            "Shipping typically takes 3-5 business days for standard delivery. Expedited shipping options are available at an additional cost.",
            "To reset your password, please visit our website and click on 'Forgot Password' link on the login page. An email will be sent to your registered address with instructions.",
            "Our customer support team is available Monday to Friday, 9 AM to 5 PM EST. You can reach us via phone at 1-800-123-4567 or email at support@example.com."
        ]
        
        documents = [Document(page_content=content) for content in docs_content]

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model_name)
        
        self.vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=embeddings, 
            persist_directory="./chroma_db_temp"
        )
        self.retriever = self.vectorstore.as_retriever()

    def retrieve_context(self, query: str) -> str:
        if not self.retriever:
            raise ValueError("Knowledge base not initialized.")
        docs = self.retriever.invoke(query)
        return "\n".join([doc.page_content for doc in docs])

knowledge_base = KnowledgeBase()


@tool
def check_order_status(order_id: str) -> str:
    if not order_id.startswith("ORD"):
        return "Invalid order ID format. Please provide an ID starting with 'ORD'."
    
    import random
    statuses = ["Processing", "Shipped", "Delivered", "Cancelled"]
    return f"Order {order_id} status: {random.choice(statuses)}"

@tool
def create_support_ticket(issue: str, user_id: str = "anonymous") -> str:
    import uuid
    ticket_id = str(uuid.uuid4())[:8]
    return f"Support ticket '{ticket_id}' created for user '{user_id}' with issue: '{issue}'."

@tool
def knowledge_retrieval_tool(query: str) -> str:
    return knowledge_base.retrieve_context(query)

tools = [check_order_status, create_support_ticket, knowledge_retrieval_tool]


SYSTEM_PROMPT = """You are a helpful and friendly customer support assistant.\nYour goal is to assist customers with their queries by providing accurate information, checking order statuses, and creating support tickets when necessary.\nAlways try to use the tools available to you to gather information or perform actions.\nIf you use the 'knowledge_retrieval_tool', summarize the relevant information concisely.\nIf you need an order ID or detailed issue description for a tool, ask the user for it.\nMaintain a polite and professional tone.\n"""

def get_agent_executor(llm, tools, memory):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory, handle_parsing_errors=True)
    return agent_executor


app = FastAPI(
    title="LLM Augmented Customer Support Chatbot",
    description="A plug-and-play LLM augmentation framework for customer support.",
    version="0.1.0",
)

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default_session"

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing knowledge base...")
    knowledge_base.initialize()
    print("Knowledge base initialized. App ready.")
    yield
    print("Shutting down application.")

app.router.lifespan_context = lifespan

llm = None
memory = None
agent_executor = None

@app.on_event("startup")
async def startup_event():
    global llm, memory, agent_executor
    llm = get_llm()
    memory = get_memory()
    agent_executor = get_agent_executor(llm, tools, memory)
    print("Chatbot components initialized and ready.")

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        response = agent_executor.invoke({"input": request.message})
        return {"response": response["output"]}
    except Exception as e:
        print(f"Error during chat processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))
