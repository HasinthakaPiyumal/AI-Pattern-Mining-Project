from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from datetime import datetime
import uuid
import threading
import time
from typing import List, Dict, Any, Tuple

class NewsScraper:
    def __init__(self):
        self._article_id_counter = 0

    def fetch_new_articles(self, num_articles: int = 5) -> List[Dict[str, Any]]:
        new_articles = []
        for _ in range(num_articles):
            self._article_id_counter += 1
            article_id = str(uuid.uuid4())
            title = f"Breaking News {self._article_id_counter}: Event on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            content = f"This is the content of breaking news article {self._article_id_counter}. It discusses recent developments in the world, updated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}."
            url = f"http://news.example.com/{article_id}"
            timestamp = datetime.now().isoformat()
            new_articles.append({"id": article_id, "title": title, "content": content, "url": url, "timestamp": timestamp})
        return new_articles

class EmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        return self._model.encode(texts)

class VectorIndex:
    def __init__(self, embedding_dimension: int):
        self._index = faiss.IndexFlatL2(embedding_dimension)
        self._id_map = []

    def add(self, embeddings: np.ndarray, ids: List[str]):
        self._index.add(embeddings)
        self._id_map.extend(ids)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[str]:
        distances, indices = self._index.search(query_embedding.reshape(1, -1), k)
        retrieved_ids = []
        for idx in indices[0]:
            if 0 <= idx < len(self._id_map):
                retrieved_ids.append(self._id_map[idx])
        return retrieved_ids

    @property
    def id_map(self) -> List[str]:
        return self._id_map

    @property
    def dimension(self) -> int:
        return self._index.d

class IndexManager:
    def __init__(self):
        self._active_index: VectorIndex = None
        self._active_document_store: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def hotswap_index(self, new_index: VectorIndex, new_document_store: Dict[str, Dict[str, Any]]):
        with self._lock:
            self._active_index = new_index
            self._active_document_store = new_document_store
            print(f"[IndexManager] Hotswapped to a new index with {len(new_document_store)} documents.")

    def get_active_index_and_docs(self) -> Tuple[VectorIndex, Dict[str, Any]]:
        with self._lock:
            return self._active_index, self._active_document_store

class RAGSystem:
    def __init__(self, embedding_model: EmbeddingModel, index_manager: IndexManager):
        self._embedding_model = embedding_model
        self._index_manager = index_manager

    def retrieve(self, query_text: str, k: int = 3) -> List[Dict[str, Any]]:
        query_embedding = self._embedding_model.encode([query_text])
        active_index, active_document_store = self._index_manager.get_active_index_and_docs()

        if active_index is None:
            return []

        retrieved_ids = active_index.search(query_embedding[0], k=k)
        retrieved_documents = [active_document_store[doc_id] for doc_id in retrieved_ids if doc_id in active_document_store]
        return retrieved_documents

    def generate_answer(self, query_text: str, retrieved_documents: List[Dict[str, Any]]) -> str:
        if not retrieved_documents:
            return f"No relevant information found for '{query_text}'."

        answer = f"Based on the latest news, here's what I found related to '{query_text}':\n\n"
        for i, doc in enumerate(retrieved_documents):
            answer += f"Article {i+1}: {doc['title']} (Published: {doc['timestamp']})\n"
            answer += f"URL: {doc['url']}\n"
            answer += f"Content Snippet: {doc['content'][:200]}...\n\n"
        return answer

def update_index_task(scraper: NewsScraper, embedding_model: EmbeddingModel, index_manager: IndexManager, interval_seconds: int = 30):
    while True:
        print(f"\n[Scheduler] Starting index update at {datetime.now().strftime('%H:%M:%S')}")
        # Simulate fetching new articles
        new_articles = scraper.fetch_new_articles(num_articles=10)
        if not new_articles:
            print("[Scheduler] No new articles fetched.")
            time.sleep(interval_seconds)
            continue

        # Prepare data for new index
        article_ids = [article["id"] for article in new_articles]
        article_contents = [article["content"] for article in new_articles]
        article_embeddings = embedding_model.encode(article_contents)

        # Create a new VectorIndex and populate it
        new_vector_index = VectorIndex(embedding_dimension=article_embeddings.shape[1])
        new_vector_index.add(article_embeddings, article_ids)

        new_document_store = {article["id"]: article for article in new_articles}

        # Hotswap the index
        index_manager.hotswap_index(new_vector_index, new_document_store)
        print(f"[Scheduler] Index update complete. Next update in {interval_seconds} seconds.")
        time.sleep(interval_seconds)

def main():
    print("Initializing Dynamic News Aggregator...")
    embedding_model = EmbeddingModel()
    index_manager = IndexManager()
    rag_system = RAGSystem(embedding_model, index_manager)
    scraper = NewsScraper()

    initial_articles = scraper.fetch_new_articles(num_articles=20)
    initial_article_ids = [article["id"] for article in initial_articles]
    initial_article_contents = [article["content"] for article in initial_articles]
    initial_embeddings = embedding_model.encode(initial_article_contents)

    initial_vector_index = VectorIndex(embedding_dimension=initial_embeddings.shape[1])
    initial_vector_index.add(initial_embeddings, initial_article_ids)

    initial_document_store = {article["id"]: article for article in initial_articles}
    index_manager.hotswap_index(initial_vector_index, initial_document_store)
    print(f"Initial index loaded with {len(initial_document_store)} articles.")

    update_thread = threading.Thread(target=update_index_task, args=(scraper, embedding_model, index_manager, 15), daemon=True)
    update_thread.start()

    print("\nNews Aggregator is ready. You can query about current events.")
    print("Type 'exit' to quit.")

    while True:
        query = input("\nEnter your query: ")
        if query.lower() == 'exit':
            print("Exiting News Aggregator.")
            break

        retrieved_docs = rag_system.retrieve(query, k=3)
        answer = rag_system.generate_answer(query, retrieved_docs)
        print("\n" + "="*50)
        print(answer)
        print("="*50)

if __name__ == "__main__":
    main()