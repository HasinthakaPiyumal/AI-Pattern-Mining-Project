import os
import shutil
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
# For demonstration, we'll use a placeholder for the LLM.
# In a real application, you would integrate with an actual LLM service (e.g., OpenAI, Ollama, HuggingFace Inference).
from langchain_core.language_models import BaseChatModel
from typing import List, Dict, Any


# --- Configuration ---
DATA_DIR = "medical_data"
VECTOR_DB_PATH = "faiss_index"

# --- Placeholder LLM (for demonstration purposes) ---
class MockLLM(BaseChatModel):
    """A mock LLM that simply concatenates retrieved documents with a generic answer."""
    def _generate(self, messages: List[List[Dict[str, Any]]], stop: List[str] | None = None, **kwargs: Any) -> Dict[str, Any]:
        # Extract the last user message from the input
        user_message = messages[-1][-1]["content"]
        # In a real LLM, this would involve a complex generation process.
        # Here, we just simulate a response by acknowledging the query and retrieved context.
        # The actual generation based on retrieved documents would happen in the RAG chain's prompt.
        return {"generations": [[{"text": "Based on the provided medical context, here is a summary of the relevant information.\n"}]], "llm_output": {}}

    async def _agenerate(self, messages: List[List[Dict[str, Any]]], stop: List[str] | None = None, **kwargs: Any) -> Dict[str, Any]:
        return self._generate(messages, stop, **kwargs)
    
    @property
    def _llm_type(self) -> str:
        return "mock_llm"


def load_and_index_documents(data_dir: str, vector_db_path: str):
    """
    Loads text documents from a directory, chunks them, embeds them,
    and creates a FAISS vector store.
    """
    if not os.path.exists(data_dir):
        print(f"Warning: Data directory '{data_dir}' not found. Please create it and add .txt files.")
        os.makedirs(data_dir, exist_ok=True)
        # Create a dummy file for initial setup
        with open(os.path.join(data_dir, "sample_guideline.txt"), "w") as f:
            f.write("Medical guideline for Type 2 Diabetes: Initial treatment often involves lifestyle modifications. Metformin is a first-line pharmacological agent. Regular monitoring of blood glucose levels is crucial. Newer drugs like SGLT2 inhibitors and GLP-1 receptor agonists are also used. Always consult a physician for personalized advice.")
        with open(os.path.join(data_dir, "drug_info_metformin.txt"), "w") as f:
            f.write("Metformin is an oral antidiabetic drug in the biguanide class. It is primarily used to treat type 2 diabetes, especially in people who are overweight. It works by decreasing glucose production by the liver and improving insulin sensitivity. Common side effects include nausea, diarrhea, and abdominal discomfort. Lactic acidosis is a rare but serious side effect.")
        print(f"Created sample documents in '{data_dir}'. Please populate this directory with your medical texts.")
        print("Run the script again to index these documents.")
        return None # Indicate that indexing didn't happen yet due to missing data

    print(f"Loading documents from {data_dir}...")
    loader = DirectoryLoader(data_dir, glob="**/*.txt", show_progress=True)
    documents = loader.load()

    if not documents:
        print(f"No .txt documents found in '{data_dir}'. Please add some medical texts.")
        return None

    print(f"Loaded {len(documents)} documents. Splitting into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    print("Creating embeddings and building FAISS index...")
    # Using a sentence-transformer model from HuggingFace for embeddings
    # Ensure 'sentence-transformers' and 'transformers' are installed
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(vector_db_path)
    print(f"FAISS index created and saved to {vector_db_path}.")
    return db

def get_rag_chain(db_path: str, embeddings_model_name: str):
    """
    Loads the FAISS vector store and sets up the RAG chain.
    """
    if not os.path.exists(db_path):
        print(f"Error: Vector database not found at '{db_path}'. Please run document indexing first.")
        return None

    print(f"Loading FAISS index from {db_path}...")
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 relevant chunks

    # Initialize the LLM (using MockLLM for demonstration)
    # Replace MockLLM with your actual LLM integration (e.g., ChatOpenAI(temperature=0) if using OpenAI)
    llm = MockLLM() 

    # Define the RAG prompt template
    prompt_template = PromptTemplate.from_template(
        """You are a medical knowledge assistant. Use the following context to answer the question concisely and accurately. 
        Always attribute the source documents for the information you provide.
        If you don't know the answer, state that you don't have enough information.

        Context: {context}

        Question: {question}

        Answer:"""
    )

    # Create the RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # 'stuff' combines all docs into one prompt
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template},
    )
    print("RAG chain initialized.")
    return qa_chain

def main():
    print("--- Medical Knowledge Assistant ---")
    print("Initialize or update the knowledge base first.")

    # Initial setup or re-indexing
    knowledge_base = load_and_index_documents(DATA_DIR, VECTOR_DB_PATH)
    if knowledge_base is None:
        # Exit if initial data setup failed, prompt user to add data
        return

    qa_chain = get_rag_chain(VECTOR_DB_PATH, "all-MiniLM-L6-v2")
    if qa_chain is None:
        return

    while True:
        query = input("\nEnter your medical question (or 'reindex' to update, 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        elif query.lower() == 'reindex':
            print("Re-indexing the knowledge base...")
            # Clean up old index before re-indexing
            if os.path.exists(VECTOR_DB_PATH):
                shutil.rmtree(VECTOR_DB_PATH)
            knowledge_base = load_and_index_documents(DATA_DIR, VECTOR_DB_PATH)
            if knowledge_base is None:
                qa_chain = None # Reset chain if re-indexing failed
                continue
            qa_chain = get_rag_chain(VECTOR_DB_PATH, "all-MiniLM-L6-v2")
            if qa_chain is None:
                continue
            print("Knowledge base re-indexed successfully.")
            continue

        if qa_chain is None:
            print("The RAG chain is not initialized. Please reindex if the data directory was empty.")
            continue

        print("Searching for answer...")
        try:
            result = qa_chain({"query": query})
            answer = result["result"]
            source_documents = result["source_documents"]

            print(f"\nAnswer: {answer}")
            print("\n--- Sources ---")
            if source_documents:
                for i, doc in enumerate(source_documents):
                    print(f"Source {i+1}: {doc.metadata.get('source', 'N/A')}")
                    # print(f"  Content Snippet: {doc.page_content[:200]}...") # Optional: print snippet
            else:
                print("No specific source documents retrieved.")
        except Exception as e:
            print(f"An error occurred during query processing: {e}")

    print("Exiting Medical Knowledge Assistant. Goodbye!")

if __name__ == "__main__":
    main()
