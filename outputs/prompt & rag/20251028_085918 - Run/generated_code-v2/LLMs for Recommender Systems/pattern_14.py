import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# For API and UI
from fastapi import FastAPI
import uvicorn
import streamlit as st
import json
import nltk

# Download NLTK resources
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

stop_words = set(stopwords.words('english'))

class NewsArticleFetcher:
    def __init__(self, dummy_data_path=None):
        self.articles = []
        if dummy_data_path:
            with open(dummy_data_path, 'r') as f:
                self.articles = json.load(f)
        else:
            self.articles = self._generate_dummy_articles()

    def _generate_dummy_articles(self):
        return [
            {"article_id": "1", "title": "Breaking News: AI Advances Rapidly", "description": "Recent breakthroughs in artificial intelligence are transforming industries globally, with new models achieving human-like performance in various tasks.", "category": "Technology"},
            {"article_id": "2", "title": "Global Economy Faces Inflationary Pressures", "description": "Economists warn of persistent inflation as supply chain issues and geopolitical tensions continue to disrupt global markets.", "category": "Economy"},
            {"article_id": "3", "title": "New Study on Climate Change Impacts", "description": "A comprehensive study reveals accelerating effects of climate change, emphasizing the urgent need for sustainable practices.", "category": "Environment"},
            {"article_id": "4", "title": "Tech Giants Announce Q3 Earnings", "description": "Major technology companies report strong third-quarter earnings, driven by cloud computing and advertising revenues.", "category": "Technology"},
            {"article_id": "5", "title": "Sustainable Energy Solutions Gain Traction", "description": "Innovations in renewable energy sources are making sustainable solutions more viable and accessible worldwide.", "category": "Environment"},
            {"article_id": "6", "title": "Market Volatility Continues Amidst Interest Rate Hikes", "description": "Investors are bracing for further market fluctuations as central banks signal continued interest rate increases to combat inflation.", "category": "Economy"},
            {"article_id": "7", "title": "Health Tech Innovations Revolutionize Patient Care", "description": "Advancements in health technology, including AI diagnostics and telemedicine, are significantly improving patient outcomes and access to care.", "category": "Healthcare"},
            {"article_id": "8", "title": "Discovery of Ancient Artifacts in Egypt", "description": "Archaeologists have unearthed a trove of ancient artifacts, providing new insights into a lost civilization.", "category": "Culture"},
            {"article_id": "9", "title": "Future of Work: Remote vs. Office Debate Continues", "description": "Companies and employees are still navigating the optimal balance between remote and in-office work models, with hybrid approaches becoming common.", "category": "Business"},
            {"article_id": "10", "title": "New Space Telescope Captures Stunning Images", "description": "A recently launched space telescope has delivered breathtaking images of distant galaxies, expanding our understanding of the universe.", "category": "Science"}
        ]

    def fetch_articles(self):
        return pd.DataFrame(self.articles)

class TextCleaner:
    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text) # Remove special characters
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stop_words]
        return " ".join(tokens)

class LLMContentEncoder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode_articles(self, articles_df):
        article_texts = (articles_df['title'] + " " + articles_df['description']).tolist()
        embeddings = self.model.encode(article_texts, convert_to_tensor=False)
        return dict(zip(articles_df['article_id'], embeddings))

class EmbeddingStorage:
    def __init__(self):
        self.article_embeddings = {}

    def store_embeddings(self, embeddings_dict):
        self.article_embeddings.update(embeddings_dict)

    def get_embedding(self, article_id):
        return self.article_embeddings.get(article_id)

    def get_all_embeddings(self):
        return self.article_embeddings

class UserProfileManager:
    def __init__(self, embedding_dim=384):
        self.user_profiles = defaultdict(lambda: {'read_articles': set(), 'profile_embedding': np.zeros(embedding_dim)})
        self.embedding_dim = embedding_dim

    def update_user_profile(self, user_id, article_id, article_embedding):
        user_profile = self.user_profiles[user_id]
        user_profile['read_articles'].add(article_id)
        
        # Simple aggregation: average of read article embeddings
        read_article_embeddings = [
            self.embedding_storage.get_embedding(aid) 
            for aid in user_profile['read_articles'] 
            if self.embedding_storage.get_embedding(aid) is not None
        ]
        if read_article_embeddings:
            user_profile['profile_embedding'] = np.mean(read_article_embeddings, axis=0)
        else:
            user_profile['profile_embedding'] = np.zeros(self.embedding_dim)
        self.user_profiles[user_id] = user_profile

    def get_user_profile(self, user_id):
        return self.user_profiles[user_id]
    
    def set_embedding_storage(self, storage):
        self.embedding_storage = storage

class InteractionLogger:
    def __init__(self, user_profile_manager):
        self.user_profile_manager = user_profile_manager

    def log_read_article(self, user_id, article_id, article_embedding):
        self.user_profile_manager.update_user_profile(user_id, article_id, article_embedding)

class SimilarityCalculator:
    def calculate_similarity(self, query_embedding, item_embeddings_dict):
        if query_embedding is None or not item_embeddings_dict:
            return {}
        
        item_ids = list(item_embeddings_dict.keys())
        embeddings = np.array(list(item_embeddings_dict.values()))
        
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1) # Ensure 2D for single item
        
        query_embedding_reshaped = query_embedding.reshape(1, -1)
        
        similarities = cosine_similarity(query_embedding_reshaped, embeddings)[0]
        return dict(zip(item_ids, similarities))

class RecommendationGenerator:
    def __init__(self, articles_df):
        self.articles_df = articles_df

    def generate_recommendations(self, user_id, user_profile, item_similarities, top_n=5):
        read_articles = user_profile['read_articles']
        
        # Filter out already read articles and sort by similarity
        recommended_article_ids = sorted(
            [item_id for item_id, sim in item_similarities.items() if item_id not in read_articles],
            key=item_similarities.get, 
            reverse=True
        )[:top_n]
        
        # Retrieve full article details
        recommendations = self.articles_df[self.articles_df['article_id'].isin(recommended_article_ids)].to_dict(orient='records')
        return recommendations

# --- Application Setup --- #

# Initialize components
article_fetcher = NewsArticleFetcher()
text_cleaner = TextCleaner()
llm_encoder = LLMContentEncoder()
embedding_storage = EmbeddingStorage()
user_profile_manager = UserProfileManager(embedding_dim=llm_encoder.model.get_sentence_embedding_dimension())
interaction_logger = InteractionLogger(user_profile_manager)
similarity_calculator = SimilarityCalculator()

# Set embedding storage for user profile manager
user_profile_manager.set_embedding_storage(embedding_storage)

# 1. Initialization: Fetch articles and generate embeddings
articles_df = article_fetcher.fetch_articles()
cleaned_article_texts = articles_df.apply(lambda row: text_cleaner.clean_text(row['title'] + " " + row['description']), axis=1)
articles_df['cleaned_text'] = cleaned_article_texts

article_embeddings = llm_encoder.encode_articles(articles_df)
embedding_storage.store_embeddings(article_embeddings)

recommendation_generator = RecommendationGenerator(articles_df)

# --- FastAPI Application --- #

app = FastAPI()

@app.get("/recommend/{user_id}")
async def get_recommendations(user_id: str):
    user_profile = user_profile_manager.get_user_profile(user_id)
    user_embedding = user_profile['profile_embedding']
    
    all_article_embeddings = embedding_storage.get_all_embeddings()
    
    # Ensure user_embedding is not all zeros if user hasn't read anything yet
    if np.all(user_embedding == 0) and all_article_embeddings:
        # For new users, recommend top articles by some default (e.g., popularity, or just all)
        # For this demo, if no reads, recommend articles not yet read (effectively all if new user)
        item_similarities = {aid: 0.5 for aid in all_article_embeddings.keys()} # Assign a default relevance
    else:
        item_similarities = similarity_calculator.calculate_similarity(user_embedding, all_article_embeddings)
    
    recommendations = recommendation_generator.generate_recommendations(user_id, user_profile, item_similarities)
    
    return {"user_id": user_id, "recommendations": recommendations}

@app.post("/log_read/{user_id}/{article_id}")
async def log_read(user_id: str, article_id: str):
    article_embedding = embedding_storage.get_embedding(article_id)
    if article_embedding is None:
        return {"status": "error", "message": f"Article ID {article_id} not found."}
    
    interaction_logger.log_read_article(user_id, article_id, article_embedding)
    return {"status": "success", "message": f"User {user_id} read article {article_id} logged."}

# --- Streamlit Application --- #

st.set_page_config(layout="wide")
st.title("Intelligent News Recommender")

st.sidebar.header("User Control")
user_id_input = st.sidebar.text_input("Enter User ID", "user_a")

# Simulate initial reads for demonstration
if st.sidebar.button("Simulate Initial Reads for User"):
    if user_id_input == "user_a":
        initial_reads = ["1", "4", "7"]
    elif user_id_input == "user_b":
        initial_reads = ["2", "6"]
    else:
        initial_reads = ["3", "5"]

    for article_id in initial_reads:
        article_embedding = embedding_storage.get_embedding(article_id)
        if article_embedding is not None:
            interaction_logger.log_read_article(user_id_input, article_id, article_embedding)
    st.sidebar.success(f"Simulated initial reads for {user_id_input}: {initial_reads}")

st.header(f"Recommendations for User: {user_id_input}")

if st.button("Get Recommendations"):
    # Call FastAPI endpoint to get recommendations
    # For this combined script, we directly call the function
    recommendations_data = await get_recommendations(user_id_input)
    
    if recommendations_data['recommendations']:
        for rec in recommendations_data['recommendations']:
            st.subheader(f"Article ID: {rec['article_id']} - {rec['title']}")
            st.write(f"**Category**: {rec['category']}")
            st.write(rec['description'])
            if st.button(f"Mark as Read: {rec['article_id']}", key=f"read_{user_id_input}_{rec['article_id']}"):
                article_embedding = embedding_storage.get_embedding(rec['article_id'])
                if article_embedding is not None:
                    interaction_logger.log_read_article(user_id_input, rec['article_id'], article_embedding)
                    st.success(f"Marked article {rec['article_id']} as read for {user_id_input}")
                    st.experimental_rerun()
            st.markdown("---")
    else:
        st.info("No new recommendations at this time or all articles read.")

st.sidebar.header("All Available Articles")
all_articles_display = articles_df[['article_id', 'title', 'category']]
st.sidebar.dataframe(all_articles_display)

# To run the FastAPI server alongside Streamlit, you would typically run them separately
# For a single file demo, you can't directly run uvicorn.run(app) here and also streamlit run.
# The `await` in `get_recommendations` means this Streamlit app expects an async context
# which is not directly available when running `streamlit run`. 
# To make this runnable as a single file and demonstrate, we'll adapt the Streamlit part
# to call the *internal* functions directly, simulating the API calls.

# To run this: save as news_recommender.py
# 1. Install dependencies: pip install pandas numpy nltk transformers sentence-transformers scikit-learn fastapi uvicorn "python-multipart" streamlit
# 2. Run streamlit: streamlit run news_recommender.py
#    (The FastAPI server part won't run as a separate process from `streamlit run`)

# For a true separation, you'd run:
# In terminal 1: uvicorn news_recommender:app --reload
# In terminal 2: streamlit run news_recommender.py
