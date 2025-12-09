import os
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


# --- Configuration and Initialization ---

# Set OpenAI API Key from environment variable
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") # Ensure OPENAI_API_KEY is set in your environment

app = FastAPI()

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# Initialize Embeddings
embeddings = OpenAIEmbeddings()

# --- Knowledge Base Setup (ChromaDB) ---

# Sample knowledge base content
sample_docs_content = [
    "Our return policy allows for returns within 30 days of purchase with a valid receipt. Items must be unused and in original packaging.",
    "To reset your password, visit the login page and click on 'Forgot Password'. Follow the instructions sent to your registered email address.",
    "Shipping usually takes 5-7 business days for standard delivery within the continental US. Expedited shipping options are available at checkout.",
    "You can contact customer support via live chat on our website, by email at support@example.com, or by calling 1-800-123-4567 during business hours.",
    "Our loyalty program offers points for every purchase, which can be redeemed for discounts on future orders. Sign up on our website to join.",
    "To update your billing information, log in to your account, navigate to 'Account Settings', and then select 'Payment Methods'."
]

# Create a dummy text loader for the sample content
class DummyTextLoader:
    def load(self):
        from langchain_core.documents import Document
        return [Document(page_content=doc) for doc in sample_docs_content]

# Load and split documents
loader = DummyTextLoader()
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splitted_docs = text_splitter.split_documents(docs)

# Create Chroma vector store
vectorstore = Chroma.from_documents(documents=splitted_docs, embedding=embeddings)
retriever = vectorstore.as_retriever()

# --- Query Complexity Classifier (Simulated) ---

def classify_query(query: str) -> Literal["straightforward", "moderate", "complex"]:
    query_lower = query.lower()
    if "hello" in query_lower or "hi" in query_lower or "how are you" in query_lower or "thank you" in query_lower:
        return "straightforward"
    if "return policy" in query_lower or "shipping time" in query_lower or "reset password" in query_lower or "contact support" in query_lower or "loyalty program" in query_lower or "billing information" in query_lower:
        return "moderate"
    if "optimize system performance" in query_lower or "deep technical issue" in query_lower or "integrate with third-party api" in query_lower:
        return "complex"
    return "moderate" # Default for unclassified queries

# --- Strategy Orchestrator Logic ---

def create_rag_chain():
    template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)
    
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()} 
        | prompt 
        | llm 
        | StrOutputParser()
    )
    return rag_chain

rag_chain = create_rag_chain()

async def handle_straightforward(query: str) -> str:
    prompt = ChatPromptTemplate.from_template("You are a helpful customer support agent. Be concise. Question: {question}")
    chain = prompt | llm | StrOutputParser()
    return await chain.ainvoke({"question": query})

async def handle_moderate(query: str) -> str:
    return await rag_chain.ainvoke(query)

async def handle_complex(query: str) -> str:
    # For complex queries, we simulate escalation or a more involved RAG
    # In a real system, this would involve multi-step reasoning, multi-document RAG,
    # or routing to a human agent/ specialized LLM.
    escalation_message = (
        f"Your query '{query}' is complex and requires further assistance. "
        "I am escalating this to a specialist. Please provide your contact details if you haven't already."
    )
    return escalation_message

# --- FastAPI Endpoint ---

class QueryRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat_with_bot(request: QueryRequest):
    query = request.query
    complexity = classify_query(query)

    if complexity == "straightforward":
        response = await handle_straightforward(query)
    elif complexity == "moderate":
        response = await handle_moderate(query)
    else:
        response = await handle_complex(query)
    
    return {"query": query, "complexity": complexity, "response": response}

# To run the application:
# 1. Save this file as adaptive_chatbot_backend.py
# 2. Make sure you have the required libraries installed: 
#    pip install "fastapi[all]" langchain-openai langchain-chroma langchain-community pydantic "sentence-transformers>=2.2.0" openai
# 3. Set your OPENAI_API_KEY environment variable.
# 4. Run from your terminal: uvicorn adaptive_chatbot_backend:app --reload
