
import os
from dotenv import load_dotenv
from loguru import logger
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
import functools

# LangChain imports
from langchain_community.document_loaders import TextLoader # Using TextLoader for simplicity
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnablePassthrough

# --- 0. Environment Setup ---
load_dotenv()
logger.add("file.log", rotation="500 MB")
logger.info("Application starting up...")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not found. Please set it in your .env file.")
    raise ValueError("OPENAI_API_KEY not found.")

# --- 1. Data Ingestion and Knowledge Base (Medical Corpus) ---

# Mock medical documents for demonstration
mock_medical_docs = [
    "A common symptom of influenza is a fever, body aches, headache, and fatigue. Treatment often involves rest, fluids, and antiviral medications.",
    "Diabetes mellitus is a chronic condition that affects how your body turns food into energy. Type 1 diabetes is an autoimmune disease, while Type 2 is often linked to lifestyle factors. Management includes diet, exercise, and insulin or other medications.",
    "Hypertension, or high blood pressure, increases the risk of heart disease and stroke. It often has no symptoms. Treatment involves lifestyle changes and medications like ACE inhibitors or diuretics.",
    "Pneumonia is an infection that inflames air sacs in one or both lungs, which may fill with fluid. Symptoms include cough with phlegm, fever, chills, and difficulty breathing. Antibiotics are commonly prescribed for bacterial pneumonia.",
    "Migraine headaches are severe headaches often accompanied by throbbing pain, sensitivity to light and sound, and nausea. Triggers can vary, and treatment options include pain relievers, triptans, and preventive medications."
]

def initialize_knowledge_base(docs: List[str]):
    logger.info("Initializing medical knowledge base...")
    # Save mock docs to a temporary file to use TextLoader
    with open("medical_corpus.txt", "w") as f:
        for doc in docs:
            f.write(doc + "\n\n")

    loader = TextLoader("medical_corpus.txt")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    split_documents = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    
    # Ensure the persist_directory exists
    persist_directory = "./chroma_db"
    os.makedirs(persist_directory, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=split_documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    logger.info(f"Knowledge base initialized with {len(split_documents)} chunks.")
    return vectorstore

vectorstore = initialize_knowledge_base(mock_medical_docs)
retriever = vectorstore.as_retriever()

# --- 2. LLM Integration ---
llm = ChatOpenAI(model="gpt-4o", temperature=0.0, openai_api_key=OPENAI_API_KEY)

# --- 3. Prompt Engineering ---
SYSTEM_PROMPT = (
    "You are a helpful medical assistant for healthcare professionals. "
    "Use the retrieved medical context to provide accurate diagnoses and treatment recommendations. "
    "If the information is not in the provided context, state that you don't have enough information, and avoid hallucinating. "
    "Answer concisely and professionally."
    "\n\nRetrieved Context: {context}"
)

question_answering_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ]
)

# --- 4. RAG Orchestration (Core Logic) ---

def should_retrieve(query: str) -> bool:
    """Adaptive decision-making: decide if retrieval is necessary."""
    # Simple rule-based logic: trigger retrieval for longer or specific queries
    # In a real system, this could be a classification model or more complex heuristics.
    keywords = ["diagnose", "symptoms", "treatment", "causes", "prognosis", "medical"]
    if len(query.split()) > 5 or any(keyword in query.lower() for keyword in keywords):
        logger.info(f"Query '{query}' triggered retrieval.")
        return True
    logger.info(f"Query '{query}' did not trigger retrieval (using LLM's general knowledge).")
    return False

# Stuff documents chain (for combining retrieved docs with prompt)
document_chain = create_stuff_documents_chain(llm, question_answering_prompt)

# Define the RAG chain with adaptive retrieval
def medical_rag_chain(query: str):
    if should_retrieve(query):
        # Use the full RAG chain
        retrieval_augmented_chain = create_retrieval_chain(retriever, document_chain)
        response = retrieval_augmented_chain.invoke({"input": query})
        return response["answer"]
    else:
        # Directly invoke LLM without retrieval
        response = llm.invoke(question_answering_prompt.format(context="No additional context retrieved.", input=query))
        return response.content

# --- 5. System-Level Optimizations (Caching) ---
# Cache the RAG function for identical queries
@functools.lru_cache(maxsize=128) # Cache up to 128 recent queries
def cached_medical_rag_chain(query: str) -> str:
    logger.info(f"Processing query (possibly from cache): {query}")
    return medical_rag_chain(query)

# --- 6. User Interface (FastAPI) ---
app = FastAPI(
    title="Medical Diagnosis and Treatment Recommendation System",
    description="A RAG-powered system for healthcare professionals to get accurate medical information."
)

class MedicalQuery(BaseModel):
    query: str

class MedicalResponse(BaseModel):
    diagnosis: str
    source_documents: List[Dict[str, Any]] = []

@app.post("/diagnose", response_model=MedicalResponse)
async def diagnose_medical_case(medical_query: MedicalQuery):
    logger.info(f"Received diagnosis request for query: {medical_query.query}")
    
    # Adaptive decision-making and RAG execution
    if should_retrieve(medical_query.query):
        # For retrieving source documents, we need to invoke the retrieval_augmented_chain directly
        # Or modify medical_rag_chain to return sources. 
        # For simplicity in this example, we'll run the full chain and extract documents if present.
        retrieval_augmented_chain = create_retrieval_chain(retriever, document_chain)
        response_obj = retrieval_augmented_chain.invoke({"input": medical_query.query})
        answer = response_obj["answer"]
        source_docs = []
        if "context" in response_obj:
            source_docs = [{
                "page_content": doc.page_content,
                "metadata": doc.metadata
            } for doc in response_obj["context"]]
    else:
        answer = llm.invoke(question_answering_prompt.format(context="No additional context retrieved.", input=medical_query.query)).content
        source_docs = [] # No specific sources if retrieval was skipped

    # The cached_medical_rag_chain returns only the answer string. 
    # If we want to return source documents, we need to adjust the caching strategy
    # or re-run the retrieval part (which defeats partial caching).
    # For this example, if should_retrieve is True, we run the chain to get sources.
    # If should_retrieve is False, we just use the LLM and have no sources.

    logger.info(f"Diagnosis completed for query: {medical_query.query}")
    return MedicalResponse(diagnosis=answer, source_documents=source_docs)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

# To run the application:
# 1. Save this code as `main.py`
# 2. Create a `.env` file in the same directory with `OPENAI_API_KEY="your_openai_api_key"`
# 3. Install dependencies: `pip install fastapi uvicorn python-dotenv loguru langchain-community langchain-text-splitters langchain-openai chromadb pydantic`
# 4. Run from your terminal: `uvicorn main:app --reload`
# 5. Access the API at http://127.0.0.1:8000/docs
