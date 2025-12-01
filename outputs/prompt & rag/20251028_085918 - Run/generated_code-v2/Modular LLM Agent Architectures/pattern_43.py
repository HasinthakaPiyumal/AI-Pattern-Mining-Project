from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

class MedicalKnowledgeRetriever:
    """
    A plug-and-play module for retrieving medical knowledge from a vector store.
    """
    def __init__(self, medical_data: list[str] = None):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        if medical_data is None:
            # Dummy medical data for demonstration
            self.medical_data = [
                "Common cold symptoms include runny nose, sore throat, cough, and congestion. It is caused by viruses and usually resolves within a week.",
                "Influenza (flu) symptoms are similar to a cold but often more severe, including fever, body aches, fatigue, and headache. Vaccination is recommended.",
                "Allergies can cause sneezing, itchy eyes, runny nose, and skin rashes, often triggered by pollen, dust mites, or pet dander. Antihistamines can help.",
                "Diabetes Mellitus Type 2 is characterized by high blood sugar due to insulin resistance or insufficient insulin production. Symptoms include increased thirst, frequent urination, and blurred vision. Management involves diet, exercise, and medication.",
                "Hypertension (high blood pressure) often has no symptoms but can lead to serious health problems like heart attack or stroke. Regular monitoring and lifestyle changes are crucial.",
                "Migraine headaches are severe headaches often accompanied by throbbing pain, sensitivity to light and sound, and nausea. Triggers can vary, and treatments include pain relievers and preventive medications.",
                "Appendicitis is an inflammation of the appendix, causing severe pain in the lower right abdomen, nausea, vomiting, and fever. It typically requires surgery."
            ]
        else:
            self.medical_data = medical_data
        
        self.vectorstore = self._create_vectorstore()

    def _create_vectorstore(self):
        """Creates a FAISS vector store from the medical data."""
        docs = [Document(page_content=text) for text in self.medical_data]
        # FAISS expects an embedding function that returns a list of floats
        class CustomEmbeddings:
            def __init__(self, model):
                self.model = model

            def embed_documents(self, texts):
                return self.model.encode(texts).tolist()
            
            def embed_query(self, text):
                return self.model.encode([text])[0].tolist()

        return FAISS.from_documents(docs, CustomEmbeddings(self.model))

    def retrieve(self, query: str, k: int = 3) -> list[str]:
        """
        Retrieves the top k most relevant medical facts for a given query.
        Args:
            query (str): The user's query (e.g., symptoms).
            k (int): The number of top relevant documents to retrieve.
        Returns:
            list[str]: A list of retrieved medical facts.
        """
        results = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in results]

# Example Usage (for testing the module independently)
if __name__ == "__main__":
    retriever = MedicalKnowledgeRetriever()
    symptoms = "fever, body aches, very tired"
    relevant_info = retriever.retrieve(symptoms)
    print(f"Query: {symptoms}")
    print("Retrieved Medical Information:")
    for info in relevant_info:
        print(f"- {info}")

    symptoms = "lower right abdominal pain, nausea"
    relevant_info = retriever.retrieve(symptoms)
    print(f"\nQuery: {symptoms}")
    print("Retrieved Medical Information:")
    for info in relevant_info:
        print(f"- {info}")