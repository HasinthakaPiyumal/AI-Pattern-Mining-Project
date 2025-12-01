import requests
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
from fastapi import FastAPI
import uvicorn
import torch
from sentence_transformers import SentenceTransformer
import faiss
import random
import time

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


# Configuration
MODEL_NAME = "all-MiniLM-L6-v2" # A small, efficient sentence transformer
EMBEDDING_DIM = 384 # Dimension for all-MiniLM-L6-v2
BATCH_SIZE = 32

# 1. Data Ingestion & Preprocessing Layer
class NewsScraper:
    def fetch_articles(self, num_articles=10):
        print("Fetching dummy news articles...")
        dummy_articles = [
            {"id": i, "title": f"Article {i}: Latest breakthroughs in AI", "content": f"Scientists are making rapid progress in artificial intelligence, with new models achieving human-level performance in various tasks. This is article content {i}."}
            for i in range(num_articles // 2)
        ] + [
            {"id": i + num_articles // 2, "title": f"Article {i + num_articles // 2}: Global economic trends for 2024", "content": f"Analysts predict a turbulent but potentially rewarding year for global markets, driven by inflation and geopolitical shifts. This is article content {i + num_articles // 2}."}
            for i in range(num_articles // 2)
        ]
        return pd.DataFrame(dummy_articles)

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def preprocess(self, text):
        tokens = word_tokenize(text.lower())
        filtered_tokens = [word for word in tokens if word.isalnum() and word not in self.stop_words]
        return " ".join(filtered_tokens)

# 2. Content Interpretation Layer (LLM Core)
class LLMEmbeddingModel:
    def __init__(self, model_name=MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def get_embeddings(self, texts):
        return self.model.encode(texts, convert_to_tensor=True, show_progress_bar=False).cpu().numpy()

# 3. User Profiling Layer
class UserProfileManager:
    def __init__(self):
        self.user_profiles = {}

    def get_user_profile(self, user_id):
        if user_id not in self.user_profiles:
            # Initialize with a random vector for new users (cold-start)
            self.user_profiles[user_id] = np.random.rand(EMBEDDING_DIM).astype(np.float32)
            print(f"Initialized new profile for user {user_id}")
        return self.user_profiles[user_id]

    def update_user_profile(self, user_id, article_embedding, learning_rate=0.1):
        current_profile = self.get_user_profile(user_id)
        # Simple weighted average update
        self.user_profiles[user_id] = (1 - learning_rate) * current_profile + learning_rate * article_embedding
        self.user_profiles[user_id] = self.user_profiles[user_id] / np.linalg.norm(self.user_profiles[user_id]) # Normalize

# 4. Recommendation Engine
class VectorDatabase:
    def __init__(self, embedding_dim=EMBEDDING_DIM):
        self.index = faiss.IndexFlatIP(embedding_dim) # Inner Product for similarity
        self.article_ids = []

    def add_articles(self, article_ids, embeddings):
        self.index.add(embeddings.astype('float32'))
        self.article_ids.extend(article_ids)

    def search(self, query_embedding, k=5):
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        distances, indices = self.index.search(query_embedding, k)
        recommended_article_ids = [self.article_ids[idx] for idx in indices[0] if idx < len(self.article_ids)]
        return recommended_article_ids

class RecommendationEngine:
    def __init__(self, llm_model, user_profile_manager, vector_db):
        self.llm_model = llm_model
        self.user_profile_manager = user_profile_manager
        self.vector_db = vector_db
        self.articles_df = pd.DataFrame(columns=['id', 'title', 'content', 'preprocessed_content', 'embedding'])

    def ingest_and_embed_articles(self, new_articles_df, preprocessor):
        new_articles_df['preprocessed_content'] = new_articles_df['content'].apply(preprocessor.preprocess)
        new_embeddings = self.llm_model.get_embeddings(new_articles_df['preprocessed_content'].tolist())
        new_articles_df['embedding'] = list(new_embeddings)
        
        self.articles_df = pd.concat([self.articles_df, new_articles_df], ignore_index=True)
        self.vector_db.add_articles(new_articles_df['id'].tolist(), new_embeddings)
        print(f"Ingested and embedded {len(new_articles_df)} new articles.")

    def get_recommendations(self, user_id, k=5):
        user_embedding = self.user_profile_manager.get_user_profile(user_id)
        recommended_article_ids = self.vector_db.search(user_embedding, k=k)
        
        # Retrieve full article details for recommendations
        recommended_articles = self.articles_df[self.articles_df['id'].isin(recommended_article_ids)]
        # Ensure order based on similarity search results
        recommended_articles = recommended_articles.set_index('id').loc[recommended_article_ids].reset_index()
        return recommended_articles[['id', 'title', 'content']].to_dict(orient='records')

    def record_user_interaction(self, user_id, article_id):
        article_data = self.articles_df[self.articles_df['id'] == article_id]
        if not article_data.empty:
            article_embedding = article_data['embedding'].iloc[0]
            self.user_profile_manager.update_user_profile(user_id, article_embedding)
            print(f"User {user_id} interacted with article {article_id}. Profile updated.")
        else:
            print(f"Article {article_id} not found.")

# 5. Model Optimization & Serving Layer (FastAPI)
app = FastAPI()

# Global instances (simplified for this example, in a real app use dependency injection)
news_preprocessor = TextPreprocessor()
llm_embedding_model = LLMEmbeddingModel()
user_profile_manager = UserProfileManager()
vector_db = VectorDatabase()
recommender = RecommendationEngine(llm_embedding_model, user_profile_manager, vector_db)

# Simulate initial data ingestion
scraper = NewsScraper()
initial_articles = scraper.fetch_articles(num_articles=20)
recommender.ingest_and_embed_articles(initial_articles, news_preprocessor)


@app.get("/recommend/{user_id}")
async def get_user_recommendations(user_id: int, k: int = 5):
    return recommender.get_recommendations(user_id, k)

@app.post("/interact/{user_id}/{article_id}")
async def record_interaction(user_id: int, article_id: int):
    recommender.record_user_interaction(user_id, article_id)
    return {"message": f"Interaction recorded for user {user_id} with article {article_id}"}

@app.post("/ingest_news/")
async def ingest_new_articles():
    # Simulate fetching new articles periodically
    new_articles = scraper.fetch_articles(num_articles=5)
    if not new_articles.empty:
        # Assign new unique IDs
        max_id = recommender.articles_df['id'].max() if not recommender.articles_df.empty else -1
        new_articles['id'] = range(max_id + 1, max_id + 1 + len(new_articles))
        recommender.ingest_and_embed_articles(new_articles, news_preprocessor)
        return {"message": f"Ingested {len(new_articles)} new articles.", "new_articles": new_articles['id'].tolist()}
    return {"message": "No new articles to ingest."}

# Knowledge Distillation (Conceptual Placeholder)
# In a real scenario, this would involve a training pipeline
# where a smaller model learns from the LLMEmbeddingModel.
class KnowledgeDistillation:
    def __init__(self, teacher_model, student_model):
        self.teacher_model = teacher_model # The larger LLMEmbeddingModel
        self.student_model = student_model # A smaller, faster model (e.g., smaller SBERT, or a custom tiny NN)

    def distill_knowledge(self, dataset):
        print("Simulating knowledge distillation...")
        # This would involve generating teacher logits/embeddings and training the student
        # to mimic them. For demonstration, we just return the student model.
        time.sleep(1) # Simulate training time
        print("Knowledge distillation complete. Student model is ready for deployment.")
        return self.student_model

# Example of how you might initialize and use distillation (not integrated into FastAPI for simplicity)
# For online inference, a 'distilled_ranking_model' would likely replace or augment the similarity search ranking.
# student_embedding_model = SentenceTransformer("all-MiniLM-L6-v2") # Or a custom smaller model
# kd_process = KnowledgeDistillation(llm_embedding_model.model, student_embedding_model)
# distilled_model_for_ranking = kd_process.distill_knowledge(initial_articles['preprocessed_content'].tolist())


if __name__ == "__main__":
    # To run the FastAPI app, use: uvicorn news_recommender:app --reload --port 8000
    # For demonstration, we'll start it programmatically.
    print("\nSmart News Recommender API is starting...")
    print("Access recommendations at http://127.0.0.1:8000/recommend/{user_id}")
    print("Record interactions at http://127.0.0.1:8000/interact/{user_id}/{article_id}")
    print("Ingest new articles at http://127.0.0.1:8000/ingest_news/")
    uvicorn.run(app, host="127.0.0.1", port=8000)
