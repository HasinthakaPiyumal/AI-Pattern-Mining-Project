import os
from typing import Dict, Any, List
from pydantic import BaseModel, Field

from langchain.chains import RetrievalQA, create_qa_with_sources_chain, LLMChain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.tools import tool
from langchain.vectorstores import FAISS
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

# Set your OpenAI API key
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

class NotebookWriteTool: # Custom tool to simulate NotebookWrite
    def __init__(self):
        self.notes = []

    @tool("Notebook Write")
    def write_note(self, text: str) -> str:
        """Write a note to the agent's short-term memory (notebook). Useful for remembering key details or constraints across turns."""
        self.notes.append(text)
        return f"Note '{text}' recorded in notebook."

    def get_notes(self) -> List[str]:
        return self.notes

    def clear_notes(self):
        self.notes = []

# Mock CRM and Knowledge Base Data
mock_crm_data = {
    "customer_101": {"name": "Alice Smith", "email": "alice@example.com", "last_purchase": "Laptop", "issue_history": "wifi connectivity issues"},
    "customer_102": {"name": "Bob Johnson", "email": "bob@example.com", "last_purchase": "Monitor", "issue_history": "damaged screen on delivery"},
    "customer_103": {"name": "Charlie Brown", "email": "charlie@example.com", "last_purchase": "Keyboard", "issue_history": "sticky keys, resolved with replacement"},
}

mock_kb_articles = [
    "Article 1: Troubleshooting Wi-Fi Connectivity: Ensure your router is on. Restart your device. Check network settings. If problems persist, contact support.",
    "Article 2: Monitor Setup Guide: Connect HDMI/DisplayPort cable. Plug in power. Select correct input on monitor. Adjust resolution in OS settings.",
    "Article 3: Return Policy: Items can be returned within 30 days of purchase for a full refund. Original packaging required. Some exclusions apply.",
    "Article 4: Laptop Battery Care: Avoid overcharging. Use original charger. Calibrate monthly. Store at 50% charge if not used for long periods.",
    "Article 5: Keyboard Cleaning Tips: Use compressed air. Gently wipe with a microfiber cloth. Avoid liquid spills."
]

# Initialize LLM and Embeddings
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
embeddings = OpenAIEmbeddings()

# --- Long-Term Memory (RAG) Setup ---

# CRM Vector Store
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
crm_documents = [Document(page_content=f"Customer ID: {cid}, Info: {info}") for cid, info in mock_crm_data.items()]
crm_texts = text_splitter.split_documents(crm_documents)
crm_vectorstore = FAISS.from_documents(crm_texts, embeddings)
crm_retriever = crm_vectorstore.as_retriever()

# Knowledge Base Vector Store
kb_documents = [Document(page_content=article) for article in mock_kb_articles]
kb_texts = text_splitter.split_documents(kb_documents)
kb_vectorstore = FAISS.from_documents(kb_texts, embeddings)
kb_retriever = kb_vectorstore.as_retriever()

# QA Chains for RAG
crm_qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=crm_retriever)
kb_qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=kb_retriever)

# Memory Summarization Chain
summarize_chain = load_summarize_chain(llm, chain_type="stuff")

def summarize_docs(docs: List[Document]) -> str:
    if not docs:
        return "No relevant information found."
    return summarize_chain.run(docs)

# --- Short-Term Memory Setup ---

conversation_memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=5)
notebook_tool = NotebookWriteTool()

# --- Agent Orchestration ---

tools = [
    Tool(
        name="CRM Customer History",
        func=lambda query: crm_qa_chain.run(query + " Return full customer information."),
        description="Useful for retrieving detailed history and information about a specific customer. Input should be a customer identifier or name."
    ),
    Tool(
        name="Knowledge Base Articles",
        func=lambda query: kb_qa_chain.run(query),
        description="Useful for finding general product information, troubleshooting guides, or policy details. Input should be a specific question or topic."
    ),
    notebook_tool.write_note,
]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=conversation_memory,
    handle_parsing_errors=True,
)

# --- Simulation Loop ---

print("Smart Customer Support Agent activated. Type 'exit' to end the conversation.")

while True:
    user_input = input("\nCustomer: ")
    if user_input.lower() == 'exit':
        print("Agent: Goodbye!")
        break

    # Include notebook notes in the current context for the agent
    current_notebook_notes = "\n" + "\n".join([f"NOTE: {note}" for note in notebook_tool.get_notes()]) if notebook_tool.get_notes() else ""
    
    try:
        response = agent.run(input=user_input + current_notebook_notes)
        print(f"Agent: {response}")
    except Exception as e:
        print(f"Agent Error: {e}")
        print("Agent: I encountered an error. Please try rephrasing your request.")

    # Clear notebook notes after each turn if they are meant to be 'working memory' for a single turn
    # For longer persistence, modify NotebookWriteTool to manage notes differently or integrate with ConversationBufferWindowMemory directly.
    # For this example, we'll clear them to simulate ephemeral working memory for specific prompts.
    # If notebook items are long-term, they would be added to a dedicated long-term memory module (e.g., another vector store).
    notebook_tool.clear_notes()