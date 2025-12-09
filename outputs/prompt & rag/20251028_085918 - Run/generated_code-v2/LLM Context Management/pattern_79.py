import os
import requests
import faiss
import numpy as np
import threading
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import pipeline

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "YOUR_NEWS_API_KEY")
NEWS_SOURCES = "bbc-news,reuters,associated-press"
INDEX_DIR = "news_indices"
ACTIVE_INDEX_PATH = os.path.join(INDEX_DIR, "active_index.faiss")
ACTIVE_METADATA_PATH = os.path.join(INDEX_DIR, "active_index_metadata.npy")
STAGING_INDEX_PATH = os.path.join(INDEX_DIR, "staging_index.faiss")
STAGING_METADATA_PATH = os.path.join(INDEX_DIR, "staging_index_metadata.npy")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GENERATION_MODEL_NAME = "google/flan-t5-small"
CHUNK_SIZE = 500
OVERLAP = 50
TOP_K_RETRIEVAL = 3

current_faiss_index = None
current_news_articles_metadata = []
embedding_model = None
generation_pipeline = None

class QueryRequest(BaseModel):
    query: str

class QAResponse(BaseModel):
    answer: str
    sources: list[str]

class SummarizeRequest(BaseModel):
    topic: str

class SummarizeResponse(BaseModel):
    summary: str
    sources: list[str]

class NewsDataService:
    def fetch_news(self, api_key: str, sources: str, query: str = "") -> list[dict]:
        print("Fetching news...")
        if query:
            return [
                {"title": f"Breaking: {query} Update", "description": f"Details about the latest developments on {query}.", "url": "http://example.com/query-news", "content": f"The situation regarding {query} has evolved significantly. Experts are now saying... " * 5},
                {"title": f"Analysis of {query}", "description": f"An in-depth look at {query} and its implications.", "url": "http://example.com/query-analysis", "content": f"Understanding the complexities of {query} requires considering various factors, including... " * 5},
            ]
        return [
            {"title": "Global Markets Rise", "description": "Stock markets around the world saw significant gains today.", "url": "http://example.com/news1", "content": "Major indices closed higher as investor confidence returned after recent economic data releases. Analysts point to strong corporate earnings as a key driver."},
            {"title": "New Tech Breakthrough", "description": "A company announced a revolutionary AI advancement.", "url": "http://example.com/news2", "content": "Scientists have unveiled a groundbreaking AI model capable of understanding complex human emotions. This could revolutionize human-computer interaction."},
            {"title": "Local Election Results", "description": "The mayoral race in Cityville concluded with unexpected results.", "url": "http://example.com/news3", "content": "Candidate Smith narrowly defeated Candidate Jones in a closely watched election. Voter turnout was higher than anticipated."},
            {"title": "Climate Change Report", "description": "New report highlights urgent need for action on climate.", "url": "http://example.com/news4", "content": "The intergovernmental panel on climate change released its latest assessment, emphasizing the need for immediate global cooperation to curb emissions."},
            {"title": "Sports Final: Team A Wins Championship", "description": "Team A clinched the championship in a thrilling final match.", "url": "http://example.com/news5", "content": "In a nail-biting finish, Team A secured victory with a last-minute goal, sending their fans into a frenzy. It was a historic moment for the club."}
        ]

    def preprocess_text(self, text: str) -> str:
        text = text.replace("\n", " ").replace("\r", " ").strip()
        text = " ".join(text.split())
        return text

    def chunk_text(self, text: str, chunk_size: int, overlap: int) -> list[str]:
        chunks = []
        words = text.split()
        if len(words) <= chunk_size:
            return [text]
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

class EmbeddingService:
    def __init__(self):
        global embedding_model
        if embedding_model is None:
            print("Loading embedding model...")
            embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            print("Embedding model loaded.")

    def get_embeddings(self, texts: list[str]) -> np.ndarray:
        return embedding_model.encode(texts, convert_to_numpy=True)

class IndexBuilderService:
    def __init__(self, news_service: NewsDataService, embedding_service: EmbeddingService):
        self.news_service = news_service
        self.embedding_service = embedding_service

    def build_and_save_index(self, articles: list[dict], index_path: str, metadata_path: str):
        print(f"Building new index at {index_path}...")
        documents = []
        metadata = []
        
        for article in articles:
            content = self.news_service.preprocess_text(article.get("content", article.get("description", "")))
            chunks = self.news_service.chunk_text(content, CHUNK_SIZE, OVERLAP)
            for i, chunk in enumerate(chunks):
                documents.append(chunk)
                metadata.append({
                    "title": article.get("title", ""),
                    "url": article.get("url", ""),
                    "chunk_idx": i,
                    "original_content": chunk
                })

        if not documents:
            print("No documents to index. Skipping index creation.")
            return None, None

        embeddings = self.embedding_service.get_embeddings(documents)
        dimension = embeddings.shape[1]
        
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

        faiss.write_index(index, index_path)
        np.save(metadata_path, metadata)
        print(f"Index built and saved to {index_path} and {metadata_path}")
        return index_path, metadata_path

    def run_background_builder(self):
        while True:
            print("Index builder running in background...")
            new_articles = self.news_service.fetch_news(NEWS_API_KEY, NEWS_SOURCES)
            if new_articles:
                staging_index_path, staging_metadata_path = self.build_and_save_index(new_articles, STAGING_INDEX_PATH, STAGING_METADATA_PATH)
                if staging_index_path and staging_metadata_path:
                    global current_faiss_index, current_news_articles_metadata
                    try:
                        new_index = faiss.read_index(staging_index_path)
                        new_metadata = np.load(staging_metadata_path, allow_pickle=True).tolist()
                        
                        current_faiss_index = new_index
                        current_news_articles_metadata = new_metadata
                        print("Index hotswapped successfully by background builder!")
                    except Exception as e:
                        print(f"Error during background hotswap: {e}")
            time.sleep(300)

app = FastAPI()

def load_initial_index():
    global current_faiss_index, current_news_articles_metadata, embedding_model, generation_pipeline

    if not os.path.exists(INDEX_DIR):
        os.makedirs(INDEX_DIR)

    embedding_model_instance = EmbeddingService()
    if generation_pipeline is None:
        print("Loading generation model...")
        generation_pipeline = pipeline("text2text-generation", model=GENERATION_MODEL_NAME)
        print("Generation model loaded.")

    if not os.path.exists(ACTIVE_INDEX_PATH) or not os.path.exists(ACTIVE_METADATA_PATH):
        print("No active index found, building initial index...")
        news_service = NewsDataService()
        initial_articles = news_service.fetch_news(NEWS_API_KEY, NEWS_SOURCES)
        index_builder = IndexBuilderService(news_service, embedding_model_instance)
        active_index_path, active_metadata_path = index_builder.build_and_save_index(initial_articles, ACTIVE_INDEX_PATH, ACTIVE_METADATA_PATH)
        if active_index_path and active_metadata_path:
            current_faiss_index = faiss.read_index(active_index_path)
            current_news_articles_metadata = np.load(active_metadata_path, allow_pickle=True).tolist()
    else:
        print(f"Loading existing active index from {ACTIVE_INDEX_PATH}")
        current_faiss_index = faiss.read_index(ACTIVE_INDEX_PATH)
        current_news_articles_metadata = np.load(ACTIVE_METADATA_PATH, allow_pickle=True).tolist()

@app.on_event("startup")
async def startup_event():
    load_initial_index()
    news_service = NewsDataService()
    embedding_service = EmbeddingService()
    index_builder = IndexBuilderService(news_service, embedding_service)
    builder_thread = threading.Thread(target=index_builder.run_background_builder, daemon=True)
    builder_thread.start()
    print("Background index builder started.")

@app.get("/")
async def read_root():
    return {"message": "Real-time News Q&A and Summarization System is running."}

@app.post("/qa", response_model=QAResponse)
async def answer_question(request: QueryRequest):
    if current_faiss_index is None or not current_news_articles_metadata:
        raise HTTPException(status_code=503, detail="Knowledge base not yet initialized or empty.")

    query_embedding = embedding_model.encode([request.query], convert_to_numpy=True)

    D, I = current_faiss_index.search(query_embedding, TOP_K_RETRIEVAL)
    retrieved_indices = I[0]

    context_chunks = []
    sources = []
    
    for idx in retrieved_indices:
        if 0 <= idx < len(current_news_articles_metadata):
            metadata = current_news_articles_metadata[idx]
            context_chunks.append(metadata["original_content"])
            sources.append(f"{metadata['title']} ({metadata['url']})")

    if not context_chunks:
        return QAResponse(answer="I could not find relevant information in the current news index.", sources=[])

    context_text = "\n".join(context_chunks)
    prompt = f"Context: {context_text}\nQuestion: {request.query}\nAnswer:"

    result = generation_pipeline(prompt, max_new_tokens=150, num_return_sequences=1)
    answer = result[0]["generated_text"].strip()

    return QAResponse(answer=answer, sources=list(set(sources)))

@app.post("/summarize", response_model=SummarizeResponse)
async def summarize_topic(request: SummarizeRequest):
    if current_faiss_index is None or not current_news_articles_metadata:
        raise HTTPException(status_code=503, detail="Knowledge base not yet initialized or empty.")

    news_service = NewsDataService()
    topic_articles = news_service.fetch_news(NEWS_API_KEY, NEWS_SOURCES, query=request.topic)

    documents_to_summarize = []
    summarization_sources = []

    if not topic_articles:
        return SummarizeResponse(summary=f"Could not find relevant news for topic: {request.topic}", sources=[])

    for article in topic_articles:
        content = news_service.preprocess_text(article.get("content", article.get("description", "")))
        documents_to_summarize.append(content)
        summarization_sources.append(f"{article.get('title', 'N/A')} ({article.get('url', 'N/A')})")

    if not documents_to_summarize:
        return SummarizeResponse(summary=f"No content found to summarize for topic: {request.topic}", sources=[])

    full_context_for_summary = " ".join(documents_to_summarize)
    
    max_llm_input_length = generation_pipeline.model.config.max_position_embeddings if hasattr(generation_pipeline.model.config, 'max_position_embeddings') else 512
    if len(full_context_for_summary.split()) > max_llm_input_length * 0.8:
        full_context_for_summary = " ".join(full_context_for_summary.split()[:int(max_llm_input_length * 0.8)])

    prompt = f"Summarize the following news about {request.topic}:\n{full_context_for_summary}\nSummary:"
    
    result = generation_pipeline(prompt, max_new_tokens=200, num_return_sequences=1)
    summary = result[0]["generated_text"].strip()

    return SummarizeResponse(summary=summary, sources=list(set(summarization_sources)))

@app.post("/hotswap_index")
async def hotswap_index_endpoint():
    global current_faiss_index, current_news_articles_metadata
    if not os.path.exists(STAGING_INDEX_PATH) or not os.path.exists(STAGING_METADATA_PATH):
        raise HTTPException(status_code=404, detail="No staging index or metadata found to hotswap.")
    
    try:
        new_index = faiss.read_index(STAGING_INDEX_PATH)
        new_metadata = np.load(STAGING_METADATA_PATH, allow_pickle=True).tolist()
        
        current_faiss_index = new_index
        current_news_articles_metadata = new_metadata

        print("Index hotswapped via API call!")
        return {"message": "Index hotswapped successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during hotswap: {str(e)}")