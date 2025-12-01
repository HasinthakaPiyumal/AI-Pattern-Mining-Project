import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

class KnowledgeRetrievalModule:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
        self.document_embeddings = None
        print(f"KnowledgeRetrievalModule initialized with model: {model_name}")

    def add_documents(self, docs):
        if not docs:
            print("No documents to add.")
            return

        print(f"Adding {len(docs)} documents to the knowledge base...")
        new_embeddings = self.model.encode(docs, convert_to_numpy=True)
        
        if self.document_embeddings is None:
            self.document_embeddings = new_embeddings
        else:
            self.document_embeddings = np.vstack([self.document_embeddings, new_embeddings])

        self.documents.extend(docs)

        # Initialize or update FAISS index
        dimension = self.document_embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dimension)
        
        # Clear and re-add all documents to the FAISS index for simplicity
        # For a production system, consider incremental updates or a more robust index management
        self.index = faiss.IndexFlatL2(dimension) # Re-initialize for simplicity
        self.index.add(self.document_embeddings)
        print(f"Successfully added {len(docs)} documents. Total documents: {len(self.documents)}")


    def retrieve(self, query, top_k=3):
        if self.index is None or len(self.documents) == 0:
            print("Knowledge base is empty. Cannot retrieve documents.")
            return []

        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # Ensure top_k does not exceed the number of available documents
        actual_top_k = min(top_k, len(self.documents))
        
        D, I = self.index.search(query_embedding, actual_top_k)
        
        retrieved_docs = []
        for idx in I[0]:
            if idx < len(self.documents): # Ensure index is valid
                retrieved_docs.append(self.documents[idx])
        
        print(f"Retrieved {len(retrieved_docs)} documents for query: '{query}'")
        return retrieved_docs