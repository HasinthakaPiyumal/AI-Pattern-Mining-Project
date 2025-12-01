from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.metrics.pairwise import cosine_similarity
import re

# --- 1. Data Ingestion & Preprocessing Module (Simplified) ---
class NewsArticle(BaseModel):
    article_id: str
    title: str
    text: str
    category: str = "general"

class NewsProcessor:
    def clean_text(self, text: str) -> str:
        text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
        text = text.lower()  # Convert to lowercase
        return text

# --- 2. LLM-based Content Interpretation Module ---
class LLMContentInterpreter:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def get_embedding(self, text: str) -> np.ndarray:
        # Ensure input is a list of strings, even for a single text
        embeddings = self.model.encode([text], convert_to_numpy=True)
        return embeddings[0]

# --- 3. User Profiling Module (Simplified In-Memory) ---
class UserProfileManager:
    def __init__(self):
        self.user_profiles = {}

    def update_user_profile(self, user_id: str, article_embedding: np.ndarray):
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = []
        self.user_profiles[user_id].append(article_embedding)

    def get_user_embedding(self, user_id: str) -> np.ndarray | None:
        if user_id in self.user_profiles and self.user_profiles[user_id]:
            return np.mean(self.user_profiles[user_id], axis=0)
        return None

# --- 4. Recommendation Generation Module ---
class RecommendationEngine:
    def __init__(self, embedding_dim: int):
        self.article_embeddings_index = faiss.IndexFlatL2(embedding_dim)  # L2 distance for similarity
        self.article_metadata = []  # Store (article_id, title) in order of insertion
        self.article_id_to_index = {}

    def add_article(self, article_id: str, title: str, embedding: np.ndarray):
        if article_id not in self.article_id_to_index:
            index = len(self.article_metadata)
            self.article_metadata.append({"article_id": article_id, "title": title})
            self.article_id_to_index[article_id] = index
            self.article_embeddings_index.add(np.array([embedding]))
        else:
            # Update existing article if needed (not fully implemented for brevity)
            pass

    def get_recommendations(self, user_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        if self.article_embeddings_index.ntotal == 0:
            return []
        
        # Reshape user_embedding for Faiss query
        D, I = self.article_embeddings_index.search(np.array([user_embedding]), top_k)
        
        recommended_articles = []
        for i in I[0]:
            if i != -1:  # Faiss returns -1 for not found indices
                article_info = self.article_metadata[i]
                recommended_articles.append(article_info)
        return recommended_articles

# --- 5. Serving Layer (FastAPI) ---
app = FastAPI()

# Initialize core components
news_processor = NewsProcessor()
llm_interpreter = LLMContentInterpreter()
user_manager = UserProfileManager()

EMBEDDING_DIM = llm_interpreter.get_embedding("test").shape[0] # Get dimension dynamically
recommender = RecommendationEngine(EMBEDDING_DIM)

# Simulate databases/storage in memory for demonstration
articles_db = {}

@app.post("/articles")
async def add_news_article(article: NewsArticle):
    cleaned_text = news_processor.clean_text(f"{article.title}. {article.text}")
    embedding = llm_interpreter.get_embedding(cleaned_text)
    recommender.add_article(article.article_id, article.title, embedding)
    articles_db[article.article_id] = article.dict()
    return {"message": "Article added successfully", "article_id": article.article_id}

@app.post("/interact")
async def user_interaction(user_id: str, article_id: str):
    if article_id not in articles_db:
        raise HTTPException(status_code=404, detail="Article not found")
    
    # Retrieve the embedding of the interacted article
    article_index = recommender.article_id_to_index.get(article_id)
    if article_index is None:
        raise HTTPException(status_code=404, detail="Article embedding not found in index.")
    
    # Faiss doesn't directly expose individual embeddings by index easily in FlatL2
    # For simplicity, re-encode or store them separately if direct retrieval is needed often.
    # For this example, we'll re-encode to simulate updating profile with the article's semantic content.
    # In a real system, you'd store embeddings directly or retrieve from a vector DB.
    interacted_article_text = f"{articles_db[article_id]['title']}. {articles_db[article_id]['text']}"
    cleaned_interacted_text = news_processor.clean_text(interacted_article_text)
    interacted_embedding = llm_interpreter.get_embedding(cleaned_interacted_text)

    user_manager.update_user_profile(user_id, interacted_embedding)
    return {"message": f"User {user_id} interacted with article {article_id}"}

@app.get("/recommend/{user_id}")
async def get_recommendations_for_user(user_id: str, top_k: int = 5):
    user_embedding = user_manager.get_user_embedding(user_id)
    if user_embedding is None:
        raise HTTPException(status_code=404, detail="User profile not found or empty.")
    
    recommendations = recommender.get_recommendations(user_embedding, top_k)
    return {"user_id": user_id, "recommendations": recommendations}


# To run the app:
# 1. Save this code as news_recommender.py
# 2. Install dependencies: pip install fastapi uvicorn pandas numpy sentence-transformers faiss-cpu scikit-learn pydantic
# 3. Run: uvicorn news_recommender:app --reload
# 4. Access API at http://127.0.0.1:8000/docs