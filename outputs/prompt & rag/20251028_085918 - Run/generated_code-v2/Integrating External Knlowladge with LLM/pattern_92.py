from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
import os

class MedicalResearchAssistant:
    def __init__(self, embedding_model_name="all-MiniLM-L6-v2", faiss_index_path="medical_faiss_index"):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        self.faiss_index_path = faiss_index_path
        self.vectorstore = None

    def _get_embeddings(self, texts):
        return self.embedding_model.encode(texts, convert_to_numpy=True)

    def ingest_documents(self, documents_path="./medical_docs"):
        print(f"Ingesting documents from {documents_path}...")
        all_texts = []
        for filename in os.listdir(documents_path):
            if filename.endswith(".txt"):
                filepath = os.path.join(documents_path, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    raw_text = f.read()
                chunks = self.text_splitter.split_text(raw_text)
                all_texts.extend(chunks)
        
        if not all_texts:
            print("No documents found or processed.")
            return

        embeddings = self._get_embeddings(all_texts)
        self.vectorstore = FAISS.from_embeddings(
            text_embeddings=list(zip(all_texts, embeddings)),
            embedding=self.embedding_model, # FAISS expects an embedding function if using from_texts, but from_embeddings takes pre-computed
            # We pass the model here for consistency, though it's not strictly used by FAISS.from_embeddings directly for generating. 
            # A custom `VectorStore` impl would be cleaner if FAISS didn't have this expectation.
            # For Langchain's FAISS, it expects an embeddings object that has an `embed_query` method.
            # Let's adjust to use FAISS.from_texts which takes the embedding model.
            # self.vectorstore = FAISS.from_texts(all_texts, self.embedding_model) # This would be if SentenceTransformer was wrapped as a Langchain Embeddings object.
            # For simplicity, we create directly from pre-computed embeddings and then wrap for query.
        )
        
        # Correct way to integrate SentenceTransformer with Langchain's FAISS if directly using from_texts for simplicity
        # Need a wrapper for SentenceTransformer to expose `embed_query` and `embed_documents`
        from langchain_community.embeddings import SentenceTransformerEmbeddings
        embeddings_model_for_langchain = SentenceTransformerEmbeddings(model_name=self.embedding_model.model_name)
        self.vectorstore = FAISS.from_texts(all_texts, embeddings_model_for_langchain)
        
        self.vectorstore.save_local(self.faiss_index_path)
        print(f"Documents ingested and FAISS index saved to {self.faiss_index_path}")

    def load_vectorstore(self):
        from langchain_community.embeddings import SentenceTransformerEmbeddings
        embeddings_model_for_langchain = SentenceTransformerEmbeddings(model_name=self.embedding_model.model_name)
        if os.path.exists(self.faiss_index_path):
            self.vectorstore = FAISS.load_local(self.faiss_index_path, embeddings_model_for_langchain, allow_dangerous_deserialization=True)
            print(f"FAISS index loaded from {self.faiss_index_path}")
        else:
            print("No existing FAISS index found. Please ingest documents first.")

    def query_knowledge_base(self, query, k=5):
        if not self.vectorstore:
            print("Vector store not initialized. Please ingest or load documents.")
            return []

        print(f"Searching for: '{query}'...")
        docs = self.vectorstore.similarity_search(query, k=k)
        
        results = []
        for i, doc in enumerate(docs):
            results.append(f"--- Retrieved Document Chunk {i+1} ---")
            results.append(doc.page_content)
            results.append("---------------------------------------")
        return results

if __name__ == "__main__":
    # Create a dummy directory for medical documents
    if not os.path.exists("medical_docs"):
        os.makedirs("medical_docs")
    
    # Create some dummy medical documents
    with open("medical_docs/diabetes_treatment.txt", "w", encoding="utf-8") as f:
        f.write("Recent advancements in Type 2 Diabetes treatment include SGLT2 inhibitors and GLP-1 receptor agonists, which have shown benefits beyond glycemic control, including cardiovascular and renal protection. Metformin remains a first-line therapy. Personalized medicine approaches are gaining traction, considering patient comorbidities and risk factors.")
    with open("medical_docs/pediatric_cough.txt", "w", encoding="utf-8") as f:
        f.write("Persistent cough in pediatric patients can be caused by various factors such as post-nasal drip, asthma, gastroesophageal reflux, or even habit cough. A thorough medical history and physical examination are crucial for accurate diagnosis. Imaging studies or pulmonary function tests might be indicated in certain cases.")
    with open("medical_docs/hypertension_guidelines.txt", "w", encoding="utf-8") as f:
        f.write("The latest hypertension guidelines emphasize lifestyle modifications, including diet and exercise, as foundational. Pharmacological interventions often start with thiazide diuretics, ACE inhibitors, ARBs, or calcium channel blockers. Target blood pressure varies based on patient age and comorbidities. Regular monitoring is essential.")

    assistant = MedicalResearchAssistant()
    
    # Ingest documents and build vector store
    assistant.ingest_documents()

    # Or load existing vector store if available (uncomment to test loading)
    # assistant.load_vectorstore()

    print("\nMedical Research Assistant Ready. Type 'exit' to quit.")
    while True:
        query = input("\nEnter your medical query: ")
        if query.lower() == 'exit':
            break
        
        results = assistant.query_knowledge_base(query)
        if results:
            for res in results:
                print(res)
        else:
            print("No relevant information found.")

    # Clean up dummy documents and index
    if os.path.exists("medical_docs"):
        import shutil
        shutil.rmtree("medical_docs")
    if os.path.exists(assistant.faiss_index_path):
        import shutil
        shutil.rmtree(assistant.faiss_index_path)
    
    print("Exiting Medical Research Assistant.")