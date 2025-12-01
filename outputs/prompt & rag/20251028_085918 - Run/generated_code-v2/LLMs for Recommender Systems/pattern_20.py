import pandas as pd
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import random

# 1. Data Ingestion & Preprocessing (Mock Data)
def load_mock_news_articles():
    articles_data = {
        "id": [f"article_{i}" for i in range(1, 21)],
        "title": [
            "Breaking News: Global Economy Shifts", "Local Elections See High Turnout",
            "New Scientific Discovery in AI", "Health Tips for a Better Lifestyle",
            "Technology Trends 2024", "Sports Highlights: Championship Finals",
            "Art and Culture: New Museum Exhibit", "Environmental Policies Debate",
            "Travel Destinations: Explore Hidden Gems", "Food and Recipe Ideas",
            "Understanding Quantum Computing", "The Future of Remote Work",
            "Investment Strategies for Beginners", "Mental Wellness: A Holistic Approach",
            "Space Exploration Milestones", "Digital Privacy Concerns",
            "Innovations in Renewable Energy", "The Rise of Indie Games",
            "Fashion Week: Latest Trends", "Historical Events Retold"
        ],
        "content": [
            "Experts discuss the latest economic indicators and their impact on global markets.",
            "Voters across the nation cast their ballots in a closely watched election.",
            "Researchers announce a breakthrough in artificial intelligence algorithms.",
            "Simple habits to improve your physical and mental well-being daily.",
            "An in-depth look at blockchain, AI, and VR technologies shaping the future.",
            "Recap of the most thrilling moments from this year's sports championship.",
            "A new collection of contemporary art opens to critical acclaim.",
            "Politicians and activists discuss the urgency of climate action.",
            "Discover serene beaches and vibrant cities for your next vacation.",
            "Easy and delicious recipes for your weeknight dinners.",
            "Demystifying the complex principles behind quantum mechanics.",
            "How companies are adapting to hybrid work models post-pandemic.",
            "A guide to smart investing for those new to the stock market.",
            "Techniques for stress reduction and mindfulness practices.",
            "NASA's latest mission to Mars yields unprecedented data.",
            "Examining the challenges of data security in the digital age.",
            "Advancements in solar and wind power generation technologies.",
            "The independent gaming scene is flourishing with creative titles.",
            "Runway trends and designer collections from the latest fashion event.",
            "Revisiting pivotal moments in history with new perspectives."
        ]
    }
    return pd.DataFrame(articles_data)

# 2. Content Interpretation (LLM as Content Interpreter)
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

class ArticleEmbedder:
    def __init__(self, model):
        self.model = model

    def generate_embeddings(self, texts):
        return self.model.encode(texts, show_progress_bar=False)

# 3. User Profile Generation & 4. Recommendation Engine
class NewsRecommender:
    def __init__(self, articles_df, embedder):
        self.articles_df = articles_df
        self.embedder = embedder
        self.article_embeddings = self.embedder.generate_embeddings(articles_df['content'].tolist())
        self.embedding_dim = self.article_embeddings.shape[1]

        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(self.embedding_dim) # IP for inner product (cosine similarity)
        self.index.add(np.array(self.article_embeddings).astype('float32'))

    def get_article_by_id(self, article_id):
        return self.articles_df[self.articles_df['id'] == article_id].iloc[0]

    def create_user_profile(self, read_article_ids):
        if not read_article_ids:
            return np.zeros(self.embedding_dim) # Return a zero vector if no articles read
        
        read_article_indices = [self.articles_df.index[self.articles_df['id'] == aid].tolist()[0] for aid in read_article_ids if aid in self.articles_df['id'].values]
        if not read_article_indices:
            return np.zeros(self.embedding_dim)

        user_read_embeddings = self.article_embeddings[read_article_indices]
        user_profile_embedding = np.mean(user_read_embeddings, axis=0)
        return user_profile_embedding

    def recommend_articles(self, user_profile_embedding, num_recommendations=5, exclude_article_ids=None):
        if np.all(user_profile_embedding == 0):
            # If user profile is empty, recommend random popular articles (for cold start)
            return self.articles_df.sample(num_recommendations).to_dict('records')

        user_profile_embedding_normalized = user_profile_embedding / np.linalg.norm(user_profile_embedding)
        
        D, I = self.index.search(np.array([user_profile_embedding_normalized]).astype('float32'), self.articles_df.shape[0])
        
        # Get original article indices and similarity scores
        recommended_indices = I[0]
        similarity_scores = D[0]

        # Filter out already read articles
        if exclude_article_ids:
            exclude_indices = [self.articles_df.index[self.articles_df['id'] == aid].tolist()[0] for aid in exclude_article_ids if aid in self.articles_df['id'].values]
            
            filtered_recommendations = []
            for idx, score in zip(recommended_indices, similarity_scores):
                if idx not in exclude_indices:
                    article_data = self.articles_df.iloc[idx].to_dict()
                    article_data['similarity_score'] = score
                    filtered_recommendations.append(article_data)
            
            # Sort by similarity score (descending) and take top N
            filtered_recommendations.sort(key=lambda x: x['similarity_score'], reverse=True)
            return filtered_recommendations[:num_recommendations]
        else:
            recommendations = []
            for idx, score in zip(recommended_indices, similarity_scores):
                article_data = self.articles_df.iloc[idx].to_dict()
                article_data['similarity_score'] = score
                recommendations.append(article_data)
            
            recommendations.sort(key=lambda x: x['similarity_score'], reverse=True)
            return recommendations[:num_recommendations]


# 6. User Interface (Streamlit app.py logic)
st.set_page_config(layout="wide")
st.title("📰 Smart News Recommender")
st.markdown("--- Choose articles you've read to get personalized recommendations --- ")

# Load data and model
articles_df = load_mock_news_articles()
embedding_model = load_embedding_model()
embedder = ArticleEmbedder(embedding_model)

recommender = NewsRecommender(articles_df, embedder)

# Sidebar for user input
st.sidebar.header("Your Reading History")

available_articles = articles_df.set_index('id')['title'].to_dict()
selected_read_article_titles = st.sidebar.multiselect(
    "Select articles you have read:",
    options=list(available_articles.values()),
    key="read_articles"
)

# Convert selected titles back to article IDs
read_article_ids = []
for title in selected_read_article_titles:
    for article_id, article_title in available_articles.items():
        if article_title == title:
            read_article_ids.append(article_id)
            break

st.sidebar.markdown("## How it works:")
st.sidebar.info(
    "This app uses a pre-trained language model (Sentence-BERT) to understand the semantic content of news articles. "
    "Based on the articles you mark as 'read', it builds a 'user profile' embedding. "
    "Then, it finds other articles semantically similar to your profile using FAISS for efficient search, "
    "providing you with personalized recommendations." 
)

# Main content area
st.header("Your Personalized Recommendations")

if read_article_ids:
    user_profile_embedding = recommender.create_user_profile(read_article_ids)
    recommendations = recommender.recommend_articles(
        user_profile_embedding,
        num_recommendations=5,
        exclude_article_ids=read_article_ids
    )

    if recommendations:
        for i, rec_article in enumerate(recommendations):
            st.subheader(f"{i+1}. {rec_article['title']}")
            st.write(f"*Similarity Score: {rec_article['similarity_score']:.4f}*")
            st.write(rec_article['content'])
            st.markdown("--- ")
    else:
        st.write("No new recommendations found based on your reading history.")
else:
    st.info("Please select some articles from the sidebar to get personalized recommendations.")
    st.header("Explore Popular Articles (Cold Start)")
    # For cold start, display some random articles
    cold_start_articles = articles_df.sample(5).to_dict('records')
    for i, article in enumerate(cold_start_articles):
        st.subheader(f"{i+1}. {article['title']}")
        st.write(article['content'])
        st.markdown("--- ")



