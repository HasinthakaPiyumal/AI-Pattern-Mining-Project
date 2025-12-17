import pandas as pd
import numpy as np
import re
import string
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI
import uvicorn
import joblib
import nltk

# Download NLTK resources (run once)
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except nltk.downloader.DownloadError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

app = FastAPI()

# --- 1. Simulate Data Ingestion ---

# Dummy data for demonstration
def get_dummy_data():
    reviews_data = {
        'review_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'product_id': [101, 102, 101, 103, 102, 104, 101, 105, 103, 104],
        'user_id': [1, 2, 1, 3, 2, 4, 1, 5, 3, 4],
        'review_text': [
            "This product is amazing! I love it so much.",
            "Disappointed with the quality. Not worth the price.",
            "Good value for money, works as expected.",
            "Absolutely terrible, don't buy this.",
            "Decent product, but could be better.",
            "Fantastic purchase, highly recommend.",
            "It's okay, nothing special.",
            "Best thing ever! So happy with it.",
            "Waste of money, completely broken.",
            "Pretty good, a solid 4/5 stars."
        ],
        'rating': [5, 2, 4, 1, 3, 5, 3, 5, 1, 4]
    }
    products_data = {
        'product_id': [101, 102, 103, 104, 105],
        'category': ['Electronics', 'Clothing', 'Electronics', 'Home & Kitchen', 'Books'],
        'price': [1200, 50, 800, 150, 25],
        'brand': ['BrandA', 'BrandB', 'BrandA', 'BrandC', 'BrandD'],
        'weight_kg': [1.5, 0.2, 0.8, 2.0, 0.1],
        'num_reviews': [150, 200, 80, 50, 120]
    }
    users_data = {
        'user_id': [1, 2, 3, 4, 5],
        'age': [30, 24, 45, 35, 29],
        'gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
        'membership_tier': ['Gold', 'Silver', 'Gold', 'Bronze', 'Silver']
    }

    df_reviews = pd.DataFrame(reviews_data)
    df_products = pd.DataFrame(products_data)
    df_users = pd.DataFrame(users_data)

    # Merge product data into reviews for training sentiment model
    df_reviews = pd.merge(df_reviews, df_products[['product_id', 'category', 'brand']], on='product_id', how='left')
    
    return df_reviews, df_products, df_users

reviews_df, products_df, users_df = get_dummy_data()

# --- 2. Data Preprocessing Layer ---

# 2.1 Text Preprocessing (for Customer Reviews)
class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000)

    def clean_text(self, text):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\d+', '', text) # Remove numbers
        text = text.strip()
        return text

    def tokenize_and_lemmatize(self, text):
        tokens = word_tokenize(text)
        lemmas = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        return " ".join(lemmas)

    def fit_transform_tfidf(self, texts):
        processed_texts = [self.tokenize_and_lemmatize(self.clean_text(text)) for text in texts]
        return self.tfidf_vectorizer.fit_transform(processed_texts)

    def transform_tfidf(self, texts):
        processed_texts = [self.tokenize_and_lemmatize(self.clean_text(text)) for text in texts]
        return self.tfidf_vectorizer.transform(processed_texts)

# 2.2 Numerical Data Preprocessing (for Product & User Data)
class NumericalPreprocessor:
    def __init__(self, numerical_cols, categorical_cols):
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols

        self.numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        self.categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', self.numeric_transformer, self.numerical_cols),
                ('cat', self.categorical_transformer, self.categorical_cols)
            ], remainder='passthrough'
        )

    def fit_transform(self, df):
        return self.preprocessor.fit_transform(df)

    def transform(self, df):
        return self.preprocessor.transform(df)

# --- Instantiate and Train Preprocessors ---

# Text Preprocessor for reviews
text_preprocessor = TextPreprocessor()
X_text_sentiment = text_preprocessor.fit_transform_tfidf(reviews_df['review_text'])
y_sentiment = (reviews_df['rating'] >= 4).astype(int) # Simple sentiment: 4-5 stars = positive (1), 1-3 stars = negative (0)

# Numerical Preprocessor for product data (for recommendations)
product_numerical_cols = ['price', 'weight_kg', 'num_reviews']
product_categorical_cols = ['category', 'brand']
product_numerical_preprocessor = NumericalPreprocessor(product_numerical_cols, product_categorical_cols)
X_numerical_products = product_numerical_preprocessor.fit_transform(products_df)

# --- 3. Machine Learning Models Layer ---

# 3.1 Sentiment Analysis Model (Logistic Regression)
sentiment_model = LogisticRegression(max_iter=1000)
sentiment_model.fit(X_text_sentiment, y_sentiment)

# 3.2 Product Recommendation Engine (Content-Based)
class ContentBasedRecommender:
    def __init__(self, product_features, product_ids):
        self.product_features = product_features # Preprocessed numerical features
        self.product_ids = product_ids
        self.cosine_sim = None

    def fit(self):
        self.cosine_sim = cosine_similarity(self.product_features, self.product_features)

    def recommend(self, product_id, n_recommendations=5):
        try:
            idx = self.product_ids[self.product_ids == product_id].index[0]
        except IndexError:
            return [] # Product not found

        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:n_recommendations+1] # Exclude itself

        product_indices = [i[0] for i in sim_scores]
        recommended_product_ids = self.product_ids.iloc[product_indices].tolist()
        return recommended_product_ids

product_recommender = ContentBasedRecommender(X_numerical_products, products_df['product_id'])
product_recommender.fit()

# --- Save Models and Preprocessors (for deployment simulation) ---
joblib.dump(text_preprocessor, 'text_preprocessor.joblib')
joblib.dump(sentiment_model, 'sentiment_model.joblib')
joblib.dump(product_numerical_preprocessor, 'product_numerical_preprocessor.joblib')
joblib.dump(product_recommender, 'product_recommender.joblib')

# --- Load Models and Preprocessors (for actual API use) ---
# In a real BentoML scenario, these would be loaded within a service or runner
# For this single file example, we just use the trained objects directly after saving them.

# --- 4. API & Deployment Layer (FastAPI) ---

# FastAPI Pydantic Models for request/response
from pydantic import BaseModel

class SentimentRequest(BaseModel):
    review_text: str

class SentimentResponse(BaseModel):
    sentiment: str
    probability_positive: float

class RecommendRequest(BaseModel):
    product_id: int
    n_recommendations: int = 5

class RecommendResponse(BaseModel):
    recommended_product_ids: list[int]

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/sentiment", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    processed_text_features = text_preprocessor.transform_tfidf([request.review_text])
    prediction = sentiment_model.predict(processed_text_features)[0]
    probability = sentiment_model.predict_proba(processed_text_features)[0][1] # Probability of positive class
    sentiment_label = "positive" if prediction == 1 else "negative"
    return SentimentResponse(sentiment=sentiment_label, probability_positive=round(probability, 4))

@app.post("/recommend", response_model=RecommendResponse)
async def get_recommendations(request: RecommendRequest):
    recommended_ids = product_recommender.recommend(request.product_id, request.n_recommendations)
    return RecommendResponse(recommended_product_ids=recommended_ids)

# To run the FastAPI application, save this file as e.g., 'main.py' and run:
# uvicorn main:app --reload
# For this example, it's just a placeholder for the API layer.

