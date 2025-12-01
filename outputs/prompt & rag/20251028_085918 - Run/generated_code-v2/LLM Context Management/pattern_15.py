import os
import pickle
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

class NewsIndexer:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def build_index(self, articles_df, index_name):
        print(f"Building index: {index_name}...")
        texts = articles_df["content"].tolist()
        embeddings = self.model.encode(texts, show_progress_bar=True)

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype("float32"))

        index_dir = "./indices"
        os.makedirs(index_dir, exist_ok=True)

        index_path = os.path.join(index_dir, f"{index_name}.faiss")
        metadata_path = os.path.join(index_dir, f"{index_name}_metadata.pkl")

        faiss.write_index(index, index_path)
        with open(metadata_path, "wb") as f:
            pickle.dump(articles_df.to_dict(orient="records"), f)
        print(f"Index {index_name} built and saved.")
        return index_path, metadata_path

    def load_index(self, index_name):
        index_path = os.path.join("./indices", f"{index_name}.faiss")
        metadata_path = os.path.join("./indices", f"{index_name}_metadata.pkl")

        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Index {index_name} not found.")

        index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as f:
            articles_data = pickle.load(f)
        articles_df = pd.DataFrame(articles_data)
        return index, articles_df

class KnowledgeBaseManager:
    def __init__(self):
        self.active_index = None
        self.active_articles = None

    def load_index(self, index_path, metadata_path):
        index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as f:
            articles_data = pickle.load(f)
        articles_df = pd.DataFrame(articles_data)
        return index, articles_df

    def set_active_index(self, index_path, metadata_path):
        print(f"Hotswapping to index from: {index_path}")
        self.active_index, self.active_articles = self.load_index(index_path, metadata_path)
        print("New index is active.")

    def get_active_index(self):
        if self.active_index is None:
            raise ValueError("No active index set. Please call set_active_index first.")
        return self.active_index, self.active_articles

class QAAgent:
    def __init__(self, embedding_model_name="all-MiniLM-L6-v2", kb_manager=None):
        self.model = SentenceTransformer(embedding_model_name)
        self.kb_manager = kb_manager

    def answer_question(self, query, top_k=3):
        if self.kb_manager is None:
            raise ValueError("Knowledge Base Manager not set for QAAgent.")

        query_embedding = self.model.encode([query]).astype("float32")

        index, articles_df = self.kb_manager.get_active_index()

        distances, indices = index.search(query_embedding, top_k)

        retrieved_articles = articles_df.iloc[indices[0]].to_dict(orient="records")

        context = "\n".join([f"Title: {a['title']}\nContent: {a['content']}" for a in retrieved_articles])

        prompt = f"Based on the following information, answer the question:\n\n{context}\n\nQuestion: {query}\nAnswer:"

        return {"answer": f"[SIMULATED LLM RESPONSE] Based on retrieved articles: {query}", "retrieved_articles": retrieved_articles}

def main():
    # Setup directories
    os.makedirs("./indices", exist_ok=True)

    # --- 1. Initial Data and Indexing ---
    print("\n--- Initializing with Old Knowledge Base ---")
    initial_news_data = [
        {"id": 1, "title": "New AI Model Achieves Breakthrough", "content": "Researchers at TechCorp have unveiled a new AI model that sets records in natural language understanding.", "date": "2023-01-15"},
        {"id": 2, "title": "Global Climate Summit Concludes", "content": "Leaders from around the world met to discuss strategies for combating climate change.", "date": "2023-01-20"},
        {"id": 3, "title": "Sports Team Wins Championship", "content": "The local basketball team clinched the national championship in a thrilling final match.", "date": "2023-01-25"},
        {"id": 4, "title": "Old World Leader John Doe Passes Away", "content": "John Doe, the former president of a major nation, has passed away at the age of 90.", "date": "2023-02-01"}
    ]
    initial_df = pd.DataFrame(initial_news_data)

    news_indexer = NewsIndexer()
    initial_index_path, initial_metadata_path = news_indexer.build_index(initial_df, "initial_news")

    kb_manager = KnowledgeBaseManager()
    kb_manager.set_active_index(initial_index_path, initial_metadata_path)

    qa_agent = QAAgent(kb_manager=kb_manager)

    # --- Initial Q&A ---
    print("\n--- Asking questions with Initial Knowledge Base ---")
    q1 = "Who passed away recently?"
    ans1 = qa_agent.answer_question(q1)
    print(f"Question: {q1}")
    print(f"Answer: {ans1['answer']}")
    print("Retrieved Articles (Initial):")
    for art in ans1["retrieved_articles"]:
        print(f"  - {art['title']}")

    q2 = "Tell me about the new AI model."
    ans2 = qa_agent.answer_question(q2)
    print(f"Question: {q2}")
    print(f"Answer: {ans2['answer']}")
    print("Retrieved Articles (Initial):")
    for art in ans2["retrieved_articles"]:
        print(f"  - {art['title']}")

    # --- 2. Updated Data and Hotswapping ---
    print("\n--- Simulating New Knowledge Arriving (Updated Knowledge Base) ---")
    updated_news_data = [
        {"id": 5, "title": "New President Elected in Nation X", "content": "Jane Smith has been elected as the new president of Nation X, promising significant policy changes.", "date": "2023-03-10"},
        {"id": 6, "title": "Breakthrough in Quantum Computing", "content": "Scientists announce a major step forward in building practical quantum computers.", "date": "2023-03-15"},
        {"id": 7, "title": "Market Sees Record Highs", "content": "Stock markets around the world reached unprecedented levels today.", "date": "2023-03-20"},
        {"id": 4, "title": "Old World Leader John Doe Passes Away (Re-confirmed)", "content": "John Doe, the former president of a major nation, has passed away at the age of 90. This is a re-confirmation.", "date": "2023-02-01"}
    ]
    updated_df = pd.DataFrame(updated_news_data)

    updated_index_path, updated_metadata_path = news_indexer.build_index(updated_df, "updated_news")

    # Hotswap the knowledge base
    kb_manager.set_active_index(updated_index_path, updated_metadata_path)

    # --- Q&A with Updated Knowledge ---
    print("\n--- Asking questions with Updated Knowledge Base ---")
    q3 = "Who is the new president of Nation X?"
    ans3 = qa_agent.answer_question(q3)
    print(f"Question: {q3}")
    print(f"Answer: {ans3['answer']}")
    print("Retrieved Articles (Updated):")
    for art in ans3["retrieved_articles"]:
        print(f"  - {art['title']}")

    q4 = "Who passed away recently?"
    ans4 = qa_agent.answer_question(q4)
    print(f"Question: {q4}")
    print(f"Answer: {ans4['answer']}")
    print("Retrieved Articles (Updated):")
    for art in ans4["retrieved_articles"]:
        print(f"  - {art['title']}")

    q5 = "Tell me about global climate summit."
    ans5 = qa_agent.answer_question(q5)
    print(f"Question: {q5}")
    print(f"Answer: {ans5['answer']}")
    print("Retrieved Articles (Updated):")
    for art in ans5["retrieved_articles"]:
        print(f"  - {art['title']}")


if __name__ == "__main__":
    main()