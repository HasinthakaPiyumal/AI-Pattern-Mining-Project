import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer

class IndexManager:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.current_index = None
        self.documents = []

    def _get_embeddings(self, texts):
        return self.embedding_model.encode(texts, convert_to_numpy=True)

    def create_index(self, texts):
        if not texts:
            return None, []

        embeddings = self._get_embeddings(texts)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        return index, texts

    def save_index(self, index, file_path):
        faiss.write_index(index, file_path)

    def load_index(self, file_path):
        return faiss.read_index(file_path)

    def hotswap_index(self, new_index, new_documents):
        self.current_index = new_index
        self.documents = new_documents
        print("Index hotswapped successfully!")

class NewsRetriever:
    def __init__(self, index_manager, embedding_model):
        self.index_manager = index_manager
        self.embedding_model = embedding_model

    def _get_query_embedding(self, query):
        return self.embedding_model.encode([query], convert_to_numpy=True)

    def retrieve(self, query, k=3):
        if self.index_manager.current_index is None:
            return []

        query_embedding = self._get_query_embedding(query)
        D, I = self.index_manager.current_index.search(query_embedding, k)
        
        retrieved_docs = []
        for doc_idx in I[0]:
            if 0 <= doc_idx < len(self.index_manager.documents):
                retrieved_docs.append(self.index_manager.documents[doc_idx])
        return retrieved_docs

def simulate_llm_response(question, context_docs):
    if not context_docs:
        return f"I don't have enough information to answer '{question}'."
    
    context_str = "\n".join(context_docs)
    response = f"Based on the following information:\n---\n{context_str}\n---\nI can tell you that regarding '{question}', the relevant details are as above."
    return response

class QASystem:
    def __init__(self, retriever, llm_function):
        self.retriever = retriever
        self.llm_function = llm_function

    def ask_question(self, question):
        retrieved_context = self.retriever.retrieve(question)
        answer = self.llm_function(question, retrieved_context)
        return answer

if __name__ == "__main__":
    # 1. Initialize Components
    print("Initializing embedding model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    index_manager = IndexManager(embedding_model)
    news_retriever = NewsRetriever(index_manager, embedding_model)
    qa_system = QASystem(news_retriever, simulate_llm_response)

    # 2. Initial Setup: Create and load the first news index
    initial_news = [
        "Apple releases new iPhone 15 with A17 Bionic chip.",
        "Google announces Q3 earnings, beating analyst expectations.",
        "Major breakthrough in AI for drug discovery reported.",
        "Global climate summit concludes with new emissions targets."
    ]
    print("\nCreating initial news index...")
    initial_faiss_index, initial_docs = index_manager.create_index(initial_news)
    index_manager.hotswap_index(initial_faiss_index, initial_docs)

    # 3. Q&A with initial knowledge
    print("\n--- Initial Q&A ---")
    question1 = "What is new with Apple's latest phone?"
    print(f"Q: {question1}")
    print(f"A: {qa_system.ask_question(question1)}")

    question2 = "Tell me about recent AI developments."
    print(f"Q: {question2}")
    print(f"A: {qa_system.ask_question(question2)}")
    
    question3 = "Who won the latest football match?"
    print(f"Q: {question3}")
    print(f"A: {qa_system.ask_question(question3)}")

    # 4. Simulate Knowledge Update: New news arrives
    print("\n--- Simulating Knowledge Update ---")
    updated_news = [
        "Apple releases new iPhone 15 with A17 Bionic chip.", # Keep some old
        "Google announces Q3 earnings, beating analyst expectations.",
        "Major breakthrough in AI for drug discovery reported.",
        "Global climate summit concludes with new emissions targets.",
        "Tech giant unveils new foldable smartphone at annual conference.", # New article
        "Government passes new bill on renewable energy incentives.",      # New article
        "Local team wins national football championship title!"            # New article
    ]
    
    # Create a new index with the updated knowledge
    print("Creating new updated index...")
    updated_faiss_index, updated_docs = index_manager.create_index(updated_news)
    
    # Hotswap the index
    index_manager.hotswap_index(updated_faiss_index, updated_docs)

    # 5. Q&A with updated knowledge
    print("\n--- Q&A After Hotswap ---")
    question4 = "What's the latest in smartphone technology?"
    print(f"Q: {question4}")
    print(f"A: {qa_system.ask_question(question4)}")

    question5 = "What are the recent government actions regarding energy?"
    print(f"Q: {question5}")
    print(f"A: {qa_system.ask_question(question5)}")
    
    question6 = "Who won the latest football match?"
    print(f"Q: {question6}")
    print(f"A: {qa_system.ask_question(question6)}")

    # Example of saving and loading an index (optional, for persistence)
    # index_file_path = "updated_news_index.bin"
    # print(f"\nSaving current index to {index_file_path}...")
    # index_manager.save_index(index_manager.current_index, index_file_path)
    # print("Loading index back...")
    # loaded_index = index_manager.load_index(index_file_path)
    # print("Index loaded successfully.")
    # os.remove(index_file_path) # Clean up the saved index file
