from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

def setup_knowledge_base(data_dir="./medical_data"):
    """
    Sets up the medical knowledge base by loading documents, splitting them,
    embedding them, and storing them in a FAISS vector store.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created data directory: {data_dir}")
        # Create some dummy medical data files for demonstration
        with open(os.path.join(data_dir, "medical_guideline_1.txt"), "w") as f:
            f.write("Medical guideline 1: \n\nDiabetes management includes regular blood sugar monitoring, diet control, and exercise. Insulin therapy may be required for Type 1 diabetes and some cases of Type 2. \n\nSource: WHO Guidelines 2023.")
        with open(os.path.join(data_dir, "research_paper_heart.txt"), "w") as f:
            f.write("Research Paper Abstract: \n\nA study on cardiovascular disease showed that a Mediterranean diet significantly reduces the risk of heart attacks. Regular physical activity also plays a crucial role. \n\nSource: New England Journal of Medicine, Vol 388, No 18.")
        with open(os.path.join(data_dir, "patient_record_sample.txt"), "w") as f:
            f.write("Patient Record Summary: \n\nPatient A, 45 years old, presented with fatigue and weight loss. Diagnosed with hypothyroidism. Treatment initiated with levothyroxine 50mcg daily. \n\nSource: Internal Clinic Record.")
        print("Created dummy medical data files.")

    documents = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".txt"):
                try:
                    loader = TextLoader(os.path.join(root, file))
                    documents.extend(loader.load())
                except Exception as e:
                    print(f"Error loading {file}: {e}")

    if not documents:
        print("No documents found in the medical_data directory. Please ensure it contains .txt files.")
        return None

    # Split documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    # Initialize embeddings model
    # Using a common sentence transformer model for embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create a FAISS vector store
    vectorstore = FAISS.from_documents(docs, embeddings)
    print("Knowledge base setup complete. FAISS vector store created.")
    return vectorstore

if __name__ == "__main__":
    # Example usage
    medical_vectorstore = setup_knowledge_base()
    if medical_vectorstore:
        query = "What are the recommendations for diabetes management?"
        retrieved_docs = medical_vectorstore.similarity_search(query, k=2)
        print(f"\nRetrieved documents for query: \'{query}\'")
        for i, doc in enumerate(retrieved_docs):
            print(f"--- Document {i+1} ---")
            print(doc.page_content)
            print("-------------------")