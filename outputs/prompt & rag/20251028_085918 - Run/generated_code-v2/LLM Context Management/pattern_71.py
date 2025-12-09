"""Builds and saves a FAISS index from processed news articles."""

import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import DATA_DIR, INDEX_DIR, EMBEDDING_MODEL_NAME, INDEX_FILE_PREFIX, PROCESSED_NEWS_FILE

def load_processed_news(filepath):
    """Loads processed news articles from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        news_articles = json.load(f)
    return news_articles

def create_embeddings(articles, model):
    """Creates embeddings for news article content."""
    texts = [article["processed_text"] for article in articles]
    print(f"Creating embeddings for {len(texts)} articles using {EMBEDDING_MODEL_NAME}...")
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings

def build_and_save_faiss_index(articles, embeddings, index_name_suffix=""):
    """Builds a FAISS index and saves it along with article metadata."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension) # Using L2 distance for similarity search
    index.add(np.array(embeddings).astype("float32"))

    os.makedirs(INDEX_DIR, exist_ok=True)
    index_filepath = os.path.join(INDEX_DIR, f"{INDEX_FILE_PREFIX}{index_name_suffix}.faiss")
    metadata_filepath = os.path.join(INDEX_DIR, f"{INDEX_FILE_PREFIX}{index_name_suffix}_metadata.json")

    faiss.write_index(index, index_filepath)
    with open(metadata_filepath, "w", encoding="utf-8") as f:
        # Store only necessary metadata to link back to articles
        metadata = [{
            "id": article["id"],
            "title": article["title"],
            "content": article["content"],
            "timestamp": article["timestamp"]
        } for article in articles]
        json.dump(metadata, f, indent=4)

    print(f"FAISS index saved to {index_filepath}")
    print(f"Metadata saved to {metadata_filepath}")
    return index_filepath, metadata_filepath

def main(news_filepath, index_name_suffix=""):
    """Main function to load news, create embeddings, and build/save the index."""
    print(f"Loading news from {news_filepath}")
    articles = load_processed_news(news_filepath)

    print(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    embeddings = create_embeddings(articles, embedding_model)
    build_and_save_faiss_index(articles, embeddings, index_name_suffix)

if __name__ == "__main__":
    # Example usage for initial index
    print("Building initial news index...")
    initial_news_file = PROCESSED_NEWS_FILE.replace(".json", "_initial.json")
    if os.path.exists(initial_news_file):
        main(initial_news_file, "initial")
    else:
        print(f"Warning: {initial_news_file} not found. Please run news_ingestion.py first.")

    # Example usage for an updated index
    print("\nBuilding updated news index...")
    updated_news_file = PROCESSED_NEWS_FILE.replace(".json", "_update_1.json")
    if os.path.exists(updated_news_file):
        main(updated_news_file, "update_1")
    else:
        print(f"Warning: {updated_news_file} not found. Please run news_ingestion.py first.")
