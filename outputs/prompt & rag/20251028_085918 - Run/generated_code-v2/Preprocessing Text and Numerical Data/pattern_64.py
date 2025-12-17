import pandas as pd
import numpy as np
import re
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, SimpleImputer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'. This may take a while...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

app = FastAPI()

# --- 1. Simulate Data Ingestion ---
# Dummy data for customer reviews
review_data = {
    "product_id": [1, 1, 2, 2, 3, 3, 1, 4, 4, 5],
    "review_text": [
        "This product is amazing! Highly recommend.",
        "Not bad, but could be better. The quality is okay.",
        "Terrible experience, completely broken after a week.",
        "Love it! Works perfectly as described.",
        "It's alright, nothing special.",
        "Excellent value for money, very durable.",
        "Good product, fast delivery.",
        "Worst purchase ever, completely regret it.",
        "Decent product for the price.",
        "Fantastic! Exceeded my expectations."
    ],
    "sentiment": ["positive", "neutral", "negative", "positive", "neutral", "positive", "positive", "negative", "neutral", "positive"]
}
reviews_df = pd.DataFrame(review_data)

# Dummy data for product metadata
product_metadata = {
    "product_id": [1, 2, 3, 4, 5],
    "price": [29.99, 15.50, 50.00, 5.99, 100.00],
    "rating": [4.5, 3.0, 1.5, 2.0, 4.8],
    "sales_history": [120, 50, 10, 200, 30],
    "category": ["Electronics", "Home Goods", "Electronics", "Books", "Fashion"]
}
products_df = pd.DataFrame(product_metadata)

# Merge dataframes for a complete view for recommendation system training
merged_df = pd.merge(reviews_df, products_df, on="product_id", how="left")

# --- 2. Data Preprocessing Module ---

# Text Preprocessing for Sentiment Analysis
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Remove punctuation and numbers
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    doc = nlp(" ".join(tokens))
    lemmas = [token.lemma_ for token in doc]
    return " ".join(lemmas)

reviews_df["processed_review"] = reviews_df["review_text"].apply(preprocess_text)

# TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=1000) # Limiting features for this example
X_text = tfidf_vectorizer.fit_transform(reviews_df["processed_review"])
y_sentiment = reviews_df["sentiment"]

# Numerical and Categorical Preprocessing for Product Recommendation
numerical_features = ["price", "rating", "sales_history"]
categorical_features = ["category"]

numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ], 
    remainder="passthrough"
)

# Apply preprocessing to product metadata for recommendation
X_numerical = preprocessor.fit_transform(products_df)

# --- 3. Model Training Module ---

# Sentiment Analysis Model
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_text, y_sentiment, test_size=0.2, random_state=42)
sentiment_model = LogisticRegression(max_iter=1000)
sentiment_model.fit(X_train_s, y_train_s)

# Product Recommendation Model
# For simplicity, let's just use the preprocessed numerical product features directly
# In a real system, you might aggregate sentiment from reviews and incorporate it.
recommendation_model = NearestNeighbors(n_neighbors=2, metric="cosine") # Recommend 2 similar products
recommendation_model.fit(X_numerical)

# --- Save Models and Preprocessors ---
joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(preprocessor, "preprocessor.pkl")
joblib.dump(sentiment_model, "sentiment_model.pkl")
joblib.dump(recommendation_model, "recommendation_model.pkl")
joblib.dump(products_df, "products_df.pkl") # To retrieve product info for recommendations

# --- 4. API and Deployment ---

# Load models and preprocessors for API
loaded_tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
loaded_preprocessor = joblib.load("preprocessor.pkl")
loaded_sentiment_model = joblib.load("sentiment_model.pkl")
loaded_recommendation_model = joblib.load("recommendation_model.pkl")
loaded_products_df = joblib.load("products_df.pkl")

class ReviewInput(BaseModel):
    review_text: str

class ProductIdInput(BaseModel):
    product_id: int

@app.post("/analyze_sentiment")
async def analyze_sentiment(review_input: ReviewInput):
    processed_text = preprocess_text(review_input.review_text)
    text_vector = loaded_tfidf_vectorizer.transform([processed_text])
    sentiment = loaded_sentiment_model.predict(text_vector)[0]
    return {"review_text": review_input.review_text, "sentiment": sentiment}

@app.post("/recommend_products")
async def recommend_products(product_id_input: ProductIdInput):
    product_id = product_id_input.product_id
    if product_id not in loaded_products_df["product_id"].values:
        return {"error": "Product ID not found"}

    # Get the features of the target product
    target_product_features = loaded_products_df[loaded_products_df["product_id"] == product_id][numerical_features + categorical_features]
    transformed_features = loaded_preprocessor.transform(target_product_features)

    distances, indices = loaded_recommendation_model.kneighbors(transformed_features)

    # Exclude the product itself from recommendations if it's in the results
    recommended_product_indices = [idx for idx in indices.flatten() if loaded_products_df.iloc[idx]["product_id"] != product_id]

    recommended_products = loaded_products_df.iloc[recommended_product_indices][["product_id", "price", "rating", "category"]].to_dict(orient="records")

    return {"product_id": product_id, "recommended_products": recommended_products}