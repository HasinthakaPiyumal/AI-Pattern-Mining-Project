import os
import glob
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

MEDICAL_DOCS_DIR = "medical_docs"
FAISS_INDEX_PATH = "faiss_index"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def setup_knowledge_base():
    if not os.path.exists(MEDICAL_DOCS_DIR):
        os.makedirs(MEDICAL_DOCS_DIR)
        # Create some dummy medical documents for demonstration
        with open(os.path.join(MEDICAL_DOCS_DIR, "diabetes_overview.txt"), "w") as f:
            f.write("Diabetes mellitus is a chronic metabolic disease characterized by high blood sugar levels. Symptoms include frequent urination, increased thirst, and unexplained weight loss. Type 1 diabetes is an autoimmune condition, while Type 2 is often linked to lifestyle factors. Management involves diet, exercise, and medication like insulin or metformin.")
        with open(os.path.join(MEDICAL_DOCS_DIR, "hypertension_guidelines.txt"), "w") as f:
            f.write("Hypertension, or high blood pressure, is a common condition that can lead to serious health problems like heart disease and stroke. Normal blood pressure is typically less than 120/80 mmHg. Lifestyle modifications such as reduced sodium intake, regular physical activity, and maintaining a healthy weight are crucial. Medications include ACE inhibitors, ARBs, and diuretics.")
        with open(os.path.join(MEDICAL_DOCS_DIR, "flu_symptoms.txt"), "w") as f:
            f.write("Influenza (flu) is a contagious respiratory illness caused by flu viruses. Symptoms include fever, cough, sore throat, body aches, headache, and fatigue. It is recommended to get an annual flu vaccine. Treatment often involves rest, fluids, and antiviral medications in some cases.")
        print(f"Created '{MEDICAL_DOCS_DIR}' directory with sample documents.")

    print(f"Loading documents from '{MEDICAL_DOCS_DIR}'...")
    loader = DirectoryLoader(MEDICAL_DOCS_DIR, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()

    if not documents:
        print("No documents found in the medical_docs directory. Please add some text files.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    print(f"Split {len(documents)} documents into {len(docs)} chunks.")

    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    print("Creating/Updating FAISS index...")
    if os.path.exists(FAISS_INDEX_PATH):
        db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        # For simplicity, we'll rebuild the index entirely on update. For large KBs, incremental updates would be better.
        print("Rebuilding FAISS index with updated documents.")
        new_db = FAISS.from_documents(docs, embeddings)
        new_db.save_local(FAISS_INDEX_PATH)
        db = new_db # Update the current db object
    else:
        db = FAISS.from_documents(docs, embeddings)
        db.save_local(FAISS_INDEX_PATH)
        print("FAISS index created.")
    
    print("Knowledge base setup complete.")
    return db, embeddings

def retrieve_medical_info(db, embeddings, query: str, k: int = 3):
    if db is None:
        print("Knowledge base not initialized. Please run 'update_kb' first.")
        return []
    print(f"Retrieving information for query: '{query}'")
    docs = db.similarity_search(query, k=k)
    return docs

def simulate_llm_response(query: str, retrieved_docs: list):
    if not retrieved_docs:
        return "I couldn't find relevant information in the knowledge base for your query. Please try rephrasing or updating the medical documents."

    context = "\n\n".join([f"Source: {d.metadata.get('source', 'Unknown')}\nContent: {d.page_content}" for d in retrieved_docs])

    # This is a placeholder for actual LLM interaction.
    # In a real application, you would send `query` and `context` to an LLM.
    # Example with a conceptual prompt:
    # prompt = f"Based on the following medical information, answer the question '{query}':\n\n{context}\n\nAnswer:"
    # llm_response = actual_llm_call(prompt)
    
    response = f"Based on the available medical knowledge, here is some information related to your query:\n\n---"
    response += f"\nQuery: {query}\n"
    response += f"\nRetrieved Evidence:\n{context}"
    response += "\n\n---\nDisclaimer: This is a simulated response based on retrieved documents and should not replace professional medical advice. Always consult a qualified healthcare provider for diagnosis and treatment."

    return response

def main():
    print("\nWelcome to the Medical Diagnostic Assistant CLI!")
    print("Type your medical query or commands: 'update_kb', 'exit'")
    
    db_and_embeddings = setup_knowledge_base()
    if db_and_embeddings is None:
        print("Exiting due to empty knowledge base.")
        return
    db, embeddings = db_and_embeddings

    while True:
        user_input = input("\nDoctor> ").strip()

        if user_input.lower() == 'exit':
            print("Exiting Medical Diagnostic Assistant. Goodbye!")
            break
        elif user_input.lower() == 'update_kb':
            print("Initiating knowledge base update...")
            db_and_embeddings = setup_knowledge_base()
            if db_and_embeddings is None:
                print("Knowledge base update failed. Please check medical_docs directory.")
                # If update fails, we might want to keep the old db or handle it more robustly
            else:
                db, embeddings = db_and_embeddings
                print("Knowledge base updated successfully!")
        elif user_input:
            retrieved_docs = retrieve_medical_info(db, embeddings, user_input)
            llm_output = simulate_llm_response(user_input, retrieved_docs)
            print(llm_output)
        else:
            print("Please enter a query or a valid command.")

if __name__ == "__main__":
    main()