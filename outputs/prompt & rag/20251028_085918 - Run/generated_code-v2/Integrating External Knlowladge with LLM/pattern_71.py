from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.documents import Document
import os

def ingest_medical_data(collection_name: str, persist_directory: str):
    """
    Ingests dummy medical text data, converts it into embeddings, and stores it
    in a Chroma vector database.
    """
    # Create a directory for persistence if it doesn't exist
    os.makedirs(persist_directory, exist_ok=True)

    # Dummy medical data - in a real application, this would come from files or a database
    medical_texts = [
        "Diabetes mellitus is a metabolic disease that causes high blood sugar. The hormone insulin moves sugar from the blood into your cells to be stored for energy.",
        "Hypertension, or high blood pressure, is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease.",
        "Symptoms of a common cold include a runny nose, sore throat, cough, congestion, slight body aches or a mild headache, and sneezing. It's caused by a virus.",
        "A myocardial infarction, commonly known as a heart attack, occurs when blood flow to a part of your heart is blocked for a long enough time that part of the heart muscle is damaged or dies.",
        "The flu (influenza) is a contagious respiratory illness caused by influenza viruses that infect the nose, throat, and sometimes the lungs. It can cause mild to severe illness, and at times can lead to death.",
        "Asthma is a condition in which your airways narrow and swell and may produce extra mucus. This can make breathing difficult and trigger coughing, a whistling sound (wheezing) when you breathe out and shortness of breath.",
        "Pneumonia is an infection that inflames the air sacs in one or both lungs. The air sacs may fill with fluid or pus (purulent material), causing cough with phlegm or pus, fever, chills, and difficulty breathing.",
        "Migraine is a type of headache characterized by recurrent moderate to severe headaches often associated with a number of autonomic nervous system symptoms. The word migraine is derived from the Greek word hemikrania, meaning 'half a head'.",
        "Appendicitis is an inflammation of the appendix, a finger-shaped pouch that projects from your colon on the lower right side of your abdomen. Appendicitis causes pain in your lower right abdomen. However, in most people, pain begins around the navel and then moves.",
        "Chronic Obstructive Pulmonary Disease (COPD) is a chronic inflammatory lung disease that causes obstructed airflow from the lungs. Symptoms include breathing difficulty, cough, mucus (sputum) production and wheezing."
    ]

    # Convert texts to Document objects
    documents = [Document(page_content=text, metadata={"source": f"medical_doc_{i}"}) for i, text in enumerate(medical_texts)]

    # Initialize embeddings model
    # Using a sentence-transformer model suitable for medical texts if available, otherwise a general one
    # For demonstration, 'all-MiniLM-L6-v2' is a good general-purpose model.
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Initialize Chroma vector store
    print(f"Initializing ChromaDB collection '{collection_name}' in '{persist_directory}'...")
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory
    )
    vectordb.persist()
    print(f"Successfully ingested {len(documents)} medical documents into ChromaDB.")
    return vectordb

if __name__ == "__main__":
    # Example usage
    COLLECTION_NAME = "medical_knowledge"
    PERSIST_DIRECTORY = "./chroma_db"
    ingest_medical_data(COLLECTION_NAME, PERSIST_DIRECTORY)
    # To verify, you can try to load it again
    # from langchain_community.vectorstores import Chroma
    # from langchain_community.embeddings import SentenceTransformerEmbeddings
    # embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    # loaded_db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings, collection_name=COLLECTION_NAME)
    # print(f"Loaded DB with {loaded_db._collection.count()} items.")