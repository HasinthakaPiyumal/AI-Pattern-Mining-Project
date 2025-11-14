import os
import asyncio
import functools
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it.")

# --- 1. Data Ingestion & Indexing ---

def load_and_index_documents(pdf_paths: list[str], persist_directory: str = "./chroma_db") -> Chroma:
    """Loads PDF documents, splits them, creates embeddings, and stores them in ChromaDB."""
    print("Loading and indexing documents...")
    all_docs = []
    for pdf_path in pdf_paths:
        try:
            loader = PyPDFLoader(pdf_path)
            all_docs.extend(loader.load())
        except Exception as e:
            print(f"Could not load {pdf_path}: {e}")

    if not all_docs:
        raise ValueError("No documents were loaded. Please check PDF paths.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = text_splitter.split_documents(all_docs)

    embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Check if a persistent ChromaDB exists, if not, create and persist
    if os.path.exists(persist_directory):
        print("Loading existing ChromaDB...")
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings_model)
    else:
        print("Creating new ChromaDB...")
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings_model, persist_directory=persist_directory)
        vectorstore.persist()
        print("ChromaDB created and persisted.")

    print("Document indexing complete.")
    return vectorstore

# --- 2. RAG Pipeline --- 

@functools.lru_cache(maxsize=128)
def _get_llm_response_cached(prompt_template: str, context: str, query: str) -> str:
    """Cached function for getting LLM response."""
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2, api_key=OPENAI_API_KEY)
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm
    return chain.invoke({"context": context, "question": query}).content

def should_retrieve(query: str) -> bool:
    """Adaptive decision-making: Heuristic to decide if retrieval is necessary.
    Retrieves if query contains medical keywords or asks for specific information.
    """
    medical_keywords = ["diagnose", "symptoms", "condition", "treatment", "research", "guidelines", "patient case"]
    if any(keyword in query.lower() for keyword in medical_keywords):
        return True
    # A more sophisticated model could be trained here
    return False

async def medical_diagnostic_assistant(query: str, vectorstore: Chroma):
    """Runs the RAG pipeline for medical diagnosis assistance."""
    print(f"\nUser Query: {query}")

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2, api_key=OPENAI_API_KEY)

    # Adaptive Decision-Making
    if should_retrieve(query):
        print("Performing intelligent retrieval...")
        # Intelligent Retrieval & Context Conditioning
        retriever = vectorstore.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
        
        # Asynchronous execution for potential pipelining benefit (conceptual here)
        # In a real pipeline, retrieval and LLM calls could run in parallel if independent
        response = await asyncio.to_thread(qa_chain.invoke, {"query": query})
        
        answer = response["result"]
        source_docs = response["source_documents"]
        print("\n--- Retrieved Information ---")
        for i, doc in enumerate(source_docs):
            print(f"Source {i+1}: {doc.metadata.get('source', 'N/A')}\nContent: {doc.page_content[:200]}...")
        print("-----------------------------")

    else:
        print("Skipping retrieval, relying on LLM's base knowledge.")
        # Fallback to direct LLM if retrieval is skipped
        prompt_template = "You are a helpful medical assistant. Answer the following question: {question}"
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | llm
        answer = await asyncio.to_thread(chain.invoke, {"question": query}).content
        source_docs = [] # No source documents if retrieval is skipped
    
    print("\n--- Diagnostic Assistant Response ---")
    print(answer)
    print("-------------------------------------")
    return answer, source_docs

# --- Main Execution --- 
async def main():
    # Ensure you have some PDF files in a 'data' directory or specify paths directly
    # Example: create a 'data' directory and put 'medical_paper1.pdf', 'clinical_guideline.pdf' inside.
    pdf_files = [
        "./data/medical_paper1.pdf", 
        "./data/clinical_guideline.pdf",
        # Add more paths to your medical PDF documents
    ]

    # Create a dummy data directory and dummy PDF files for demonstration if they don't exist
    os.makedirs("./data", exist_ok=True)
    for f in pdf_files:
        if not os.path.exists(f):
            with open(f, "w") as dummy_file:
                dummy_file.write(f"This is a dummy medical document for {os.path.basename(f)}. It contains information about general medical conditions and treatments to simulate a medical knowledge base. For example, a common symptom of flu is fever and body aches. A suggested treatment might be rest and hydration. Always consult a healthcare professional for actual medical advice. This document also discusses advanced topics in cardiology and oncology, providing insights into recent research findings and clinical trials related to heart diseases and cancer therapies.")

    try:
        vectorstore = load_and_index_documents(pdf_files)
    except ValueError as e:
        print(f"Error during document loading and indexing: {e}")
        print("Please ensure your PDF files exist in the specified paths.")
        return

    # Example queries
    queries = [
        "What are the common symptoms of influenza and what is the recommended treatment?",
        "Summarize the latest research on cardiology and oncology.",
        "What is the capital of France?", # Example of a query that might skip retrieval
        "How to diagnose a complex neurological condition based on patient symptoms?"
    ]

    for query in queries:
        await medical_diagnostic_assistant(query, vectorstore)
        # A small delay to make output readable for multiple queries
        await asyncio.sleep(1)

if __name__ == "__main__":
    # To run this code:
    # 1. pip install -r requirements.txt (where requirements.txt contains: 
    #    langchain-openai
    #    langchain-community
    #    sentence-transformers
    #    chromadb
    #    pypdf
    #    python-dotenv
    #    openai
    # 2. Create a .env file in the same directory with OPENAI_API_KEY='your_openai_api_key'
    # 3. Create a 'data' directory and place your medical PDF documents inside, 
    #    or ensure the dummy files are created.
    # 4. python medical_diagnostic_assistant.py
    asyncio.run(main())
