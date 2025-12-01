import os
import shutil
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

load_dotenv()

# --- Configuration ---
MEDICAL_DATA_DIR = "medical_data"
CHROMA_DB_DIR = "./chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- Simulate Medical Data (for demonstration if no files exist) ---
def create_dummy_medical_data(directory):
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "article_1.txt"), "w") as f:
        f.write("\n".join([
            "Medical Article 1: Aspirin is a nonsteroidal anti-inflammatory drug (NSAID) used to reduce fever and relieve mild to moderate pain.",
            "It is also used to treat inflammatory conditions such as arthritis. The most common side effects include gastrointestinal upset.",
            "Aspirin works by inhibiting cyclooxygenase (COX) enzymes, which are involved in the synthesis of prostaglandins.",
            "Prostaglandins contribute to inflammation, pain, and fever.",
            "Regular low-dose aspirin is sometimes prescribed to prevent heart attacks and strokes in individuals at high risk."
        ]))
    with open(os.path.join(directory, "guideline_2.txt"), "w") as f:
        f.write("\n".join([
            "Clinical Guideline 2: Diabetes Mellitus Type 2 Management. First-line treatment typically involves lifestyle modifications, including diet and exercise.",
            "Metformin is generally recommended as the initial pharmacological agent for most patients with type 2 diabetes.",
            "Other medications, such as GLP-1 receptor agonists or SGLT2 inhibitors, may be added if glycemic targets are not met.",
            "Regular monitoring of blood glucose levels, HbA1c, and renal function is crucial.",
            "Patient education on self-management, including medication adherence and recognizing symptoms of hypoglycemia, is vital."
        ]))
    with open(os.path.join(directory, "drug_info_3.txt"), "w") as f:
        f.write("\n".join([
            "Drug Information 3: Ibuprofen. Ibuprofen is an NSAID used for pain relief, fever reduction, and anti-inflammatory effects.",
            "It is commonly available over-the-counter. Dosing varies by condition and age.",
            "Potential side effects include stomach upset, heartburn, and dizziness.",
            "It should be used with caution in patients with kidney disease or a history of gastrointestinal bleeding.",
            "Ibuprofen inhibits prostaglandin synthesis similar to aspirin, but generally has a shorter duration of action."
        ]))
    print(f"Created dummy medical data in {directory}")

# --- Data Ingestion and Indexing ---
def initialize_vector_store():
    if not os.path.exists(MEDICAL_DATA_DIR) or not os.listdir(MEDICAL_DATA_DIR):
        create_dummy_medical_data(MEDICAL_DATA_DIR)

    print(f"Loading documents from {MEDICAL_DATA_DIR}...")
    loader = DirectoryLoader(MEDICAL_DATA_DIR, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")

    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks.")

    print("Creating embeddings and indexing into ChromaDB...")
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    # Clear existing ChromaDB if it exists to ensure fresh index
    if os.path.exists(CHROMA_DB_DIR):
        shutil.rmtree(CHROMA_DB_DIR)
        print(f"Removed existing ChromaDB at {CHROMA_DB_DIR}")

    db = Chroma.from_documents(texts, embeddings, persist_directory=CHROMA_DB_DIR)
    db.persist()
    print("ChromaDB initialized and persisted.")
    return db, embeddings

# --- RAG Assistant Functionality ---
def get_medical_assistant():
    db, embeddings = initialize_vector_store()
    retriever = db.as_retriever(search_kwargs={"k": 3})

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

def ask_medical_question(qa_chain, question):
    print(f"\nUser Question: {question}")
    result = qa_chain({"query": question})
    print("\nAI Answer:")
    print(result["result"])
    print("\nSources:")
    for doc in result["source_documents"]:
        print(f"- {doc.metadata['source']} (Page: {doc.metadata.get('page', 'N/A')})")

# --- Main Execution ---
if __name__ == "__main__":
    print("Initializing Medical RAG Assistant...")
    medical_assistant_qa = get_medical_assistant()
    print("Medical RAG Assistant ready to answer questions.\n")

    while True:
        query = input("Enter your medical question (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
        if query.strip():
            ask_medical_question(medical_assistant_qa, query)
        else:
            print("Please enter a valid question.")

    print("Exiting Medical RAG Assistant. Goodbye!")
