from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

def setup_medical_knowledge_base():
    """
    Sets up an in-memory Chroma vector database with dummy medical information.
    In a real-world scenario, this would involve ingesting data from
    PubMed, clinical trial databases, and official guidelines.
    """
    print("Setting up medical knowledge base...")

    # Dummy medical documents for demonstration
    medical_docs = [
        Document(page_content="Aspirin is commonly used as an analgesic to relieve minor aches and pains and to reduce fever. It is also an anti-inflammatory drug and can be used as an anticoagulant.", metadata={"source": "Drug Database"}),
        Document(page_content="Type 2 diabetes is a chronic condition that affects the way your body processes blood sugar (glucose). With type 2 diabetes, your body either doesn't produce enough insulin, or it resists the effects of insulin.", metadata={"source": "Medical Guidelines"}),
        Document(page_content="COVID-19 symptoms can range from mild to severe. Common symptoms include fever, cough, fatigue, and loss of taste or smell. Serious symptoms include difficulty breathing and persistent chest pain.", metadata={"source": "WHO"}),
        Document(page_content="The liver plays a vital role in detoxification, protein synthesis, and the production of biochemicals necessary for digestion. It is located in the upper right quadrant of the abdomen.", metadata={"source": "Anatomy Textbook"}),
        Document(page_content="A new study published in 'The Lancet' suggests that a novel drug, 'Medicab', shows promising results in treating a specific type of autoimmune disease, reducing inflammation by 30% in phase 3 trials.", metadata={"source": "The Lancet"})
    ]

    # Initialize HuggingFace embeddings
    # Using a common embedding model for demonstration. For production, consider domain-specific embeddings.
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create a Chroma vector store from the documents
    # This will create an in-memory vector store for this example
    vectorstore = Chroma.from_documents(documents=medical_docs, embedding=embeddings)

    print("Medical knowledge base setup complete.")
    return vectorstore.as_retriever()

if __name__ == "__main__":
    retriever = setup_medical_knowledge_base()
    print(f"Retriever created: {type(retriever)}")
    # Example of retrieval
    # results = retriever.invoke("What are the uses of Aspirin?")
    # for doc in results:
    #     print(f"- {doc.page_content}")