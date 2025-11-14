# medical_rag_system.py

import os
from dotenv import load_dotenv
from typing import List, Dict
from tqdm import tqdm

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# --- Configuration --- #
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_DB_PATH = "./chroma_db"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL_NAME = "gpt-3.5-turbo"
TOP_K_RETRIEVAL = 4

# --- Mock Medical Data (replace with actual data loading in a real application) ---
mock_medical_docs = [
    {
        "source": "Medical Journal A",
        "content": """Diabetes Mellitus is a chronic condition characterized by high blood sugar levels. Type 1 diabetes is an autoimmune disease where the body does not produce insulin. Type 2 diabetes occurs when the body either doesn't produce enough insulin or can't effectively use the insulin it produces. Symptoms include increased thirst, frequent urination, unexplained weight loss, and fatigue. Management often involves lifestyle changes, medication, and regular monitoring of blood glucose. Complications can include heart disease, kidney disease, and nerve damage."""
    },
    {
        "source": "Research Paper B",
        "content": """Hypertension, or high blood pressure, is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease. Risk factors include obesity, lack of physical activity, high-salt diet, and family history. Treatment typically involves lifestyle modifications like diet and exercise, and often prescription medications such as diuretics, ACE inhibitors, or calcium channel blockers. Regular monitoring of blood pressure is crucial."""
    },
    {
        "source": "Clinical Guidelines C",
        "content": """Common cold symptoms include a runny nose, sore throat, cough, congestion, slight body aches or a headache, and sneezing. It is caused by viruses and usually lasts for 7 to 10 days. Treatment is mainly symptomatic, focusing on relieving discomfort. This includes rest, hydration, over-the-counter pain relievers, and decongestants. Antibiotics are ineffective against the common cold because it is a viral infection."""
    },
    {
        "source": "Medical Reference D",
        "content": """Migraine is a severe type of headache characterized by throbbing pain or a pulsing sensation, usually on one side of the head. It is often accompanied by nausea, vomiting, and extreme sensitivity to light and sound. Migraine attacks can cause significant pain for hours to days. Triggers can include stress, certain foods, hormonal changes, and lack of sleep. Treatment options include pain-relieving medications (e.g., triptans) and preventive medications (e.g., beta-blockers, antidepressants). Lifestyle adjustments are also recommended."""
    },
    {
        "source": "Public Health Article E",
        "content": """Asthma is a chronic respiratory condition characterized by inflammation and narrowing of the airways, leading to difficulty breathing. Symptoms include wheezing, coughing, chest tightness, and shortness of breath. Triggers can include allergens (pollen, dust mites), irritants (smoke, pollution), exercise, and respiratory infections. Management typically involves bronchodilators to open airways and corticosteroids to reduce inflammation. An asthma action plan is essential for effective self-management."""
    }
]

# --- Data Ingestion Pipeline --- #
def ingest_medical_data(docs: List[Dict], db_path: str) -> Chroma:
    """Loads, splits, embeds, and stores medical documents in ChromaDB."""
    print(f"\n--- Ingesting {len(docs)} medical documents ---")
    
    all_texts = []
    for doc_data in tqdm(docs, desc="Processing documents"):
        # Simulate loading from a TextLoader or similar
        # In a real app, this would be actual file I/O or API calls
        from langchain_core.documents import Document
        doc = Document(page_content=doc_data["content"], metadata={"source": doc_data["source"]})
        all_texts.append(doc)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_texts)
    print(f"Split into {len(chunks)} chunks.")

    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    print("Creating and persisting ChromaDB...")
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=db_path)
    vectorstore.persist()
    print("ChromaDB created and persisted successfully.")
    return vectorstore

# --- Retrieval and Generation Modules --- #
def setup_rag_system(vectorstore: Chroma) -> callable:
    """Sets up the RAG chain with the LLM and retriever."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found. Please set it in a .env file or as an environment variable.")

    llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0)

    # Define the prompt template for the LLM
    # This prompt includes adaptive decision-making instructions
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a highly knowledgeable and experienced medical assistant. Your primary goal is to provide accurate and relevant medical diagnoses and treatment recommendations based *only* on the provided context. If the provided information is insufficient to give a confident answer, state that more information is needed or that you cannot provide a definitive answer. Do not invent information. Prioritize patient safety and recommend consulting a healthcare professional for actual medical advice.\n\nContext: {context}"),
        ("human", "{input}"),
    ])

    # Create a chain to combine documents into a single string for the LLM
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Create the retrieval chain
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K_RETRIEVAL})
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    print("RAG system initialized.")
    return retrieval_chain

# --- Caching --- #
class SimpleCache:
    """A simple in-memory cache for responses."""
    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.max_size = max_size
        self.keys_order = [] # To manage LRU

    def get(self, key: str):
        if key in self.cache:
            # Move to end to mark as recently used
            self.keys_order.remove(key)
            self.keys_order.append(key)
            return self.cache[key]
        return None

    def set(self, key: str, value: str):
        if len(self.cache) >= self.max_size:
            # Remove the least recently used item
            lru_key = self.keys_order.pop(0)
            del self.cache[lru_key]
        self.cache[key] = value
        self.keys_order.append(key)

    def __len__(self):
        return len(self.cache)

    def clear(self):
        self.cache.clear()
        self.keys_order.clear()


# --- Main CLI --- #
def main():
    print("Welcome to the Medical Diagnosis and Treatment Recommendation System!")
    print("Initializing RAG system...")

    # 1. Ingest Data
    # Check if ChromaDB already exists and has data
    vectorstore = None
    if os.path.exists(CHROMA_DB_PATH):
        try:
            embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
            vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
            # A simple check to see if it's not empty, adjust as needed
            if vectorstore._collection.count() > 0:
                print("Loaded existing ChromaDB.")
            else:
                print("Existing ChromaDB is empty, re-ingesting data.")
                vectorstore = ingest_medical_data(mock_medical_docs, CHROMA_DB_PATH)
        except Exception as e:
            print(f"Error loading ChromaDB: {e}. Re-ingesting data.")
            vectorstore = ingest_medical_data(mock_medical_docs, CHROMA_DB_PATH)
    else:
        vectorstore = ingest_medical_data(mock_medical_docs, CHROMA_DB_PATH)

    # 2. Setup RAG Chain
    try:
        rag_chain = setup_rag_system(vectorstore)
    except ValueError as e:
        print(f"Error: {e}")
        print("Please ensure your OPENAI_API_KEY is set in a .env file or as an environment variable.")
        return
    
    cache = SimpleCache(max_size=20)
    print("Type 'exit' or 'quit' to end the session.")

    while True:
        user_query = input("\nPatient Symptoms/Query (e.g., 'What are the symptoms of type 2 diabetes?'): ")
        if user_query.lower() in ["exit", "quit"]:
            print("Thank you for using the Medical RAG System. Goodbye!")
            break

        # Check cache first
        cached_response = cache.get(user_query)
        if cached_response:
            print("\n--- Cached Response ---")
            print(cached_response)
            print("-----------------------")
        else:
            print("\n--- Generating Response ---")
            try:
                response = rag_chain.invoke({"input": user_query})
                result = response["answer"]
                print(result)
                cache.set(user_query, result)
            except Exception as e:
                print(f"An error occurred during generation: {e}")
            print("-----------------------")

if __name__ == "__main__":
    main()
