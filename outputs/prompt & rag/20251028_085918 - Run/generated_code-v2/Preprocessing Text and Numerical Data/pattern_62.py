import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# Ensure NLTK data is downloaded
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except nltk.downloader.DownloadError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/omw-1.4')
except nltk.downloader.DownloadError:
    nltk.download('omw-1.4')

# --- 1. Data Ingestion Layer (Simulated) ---
def load_data():
    # Simulate loading product reviews and metadata
    product_reviews_data = {
        'product_id': [1, 1, 2, 2, 3, 3, 4, 4],
        'review_text': [
            "This product is amazing, I love it! Highly recommend.",
            "It's okay, but could be better. The quality is a bit low.",
            "Terrible product, completely broken after a week. Don't buy!",
            "Fantastic value for money. Very satisfied with my purchase.",
            "Neutral review, nothing special, nothing bad.",
            "Good product, does what it says. Fast shipping.",
            "Awful experience, customer service was rude and product arrived damaged.",
            "Best gadget ever! Exceeded my expectations."
        ],
        'sentiment_label': ['positive', 'neutral', 'negative', 'positive', 'neutral', 'positive', 'negative', 'positive'] # Dummy labels for training
    }
    products_metadata_data = {
        'product_id': [1, 2, 3, 4],
        'product_name': ['Laptop Pro', 'Wireless Mouse', 'USB Hub', 'Smart Speaker'],
        'price': [1200.00, 25.50, 15.00, 99.99],
        'category': ['Electronics', 'Electronics', 'Accessories', 'Electronics'],
        'brand': ['TechCorp', 'Generic', 'ConnectAll', 'AudioMax'],
        'rating': [4.5, 3.0, 3.5, 4.8]
    }
    
    reviews_df = pd.DataFrame(product_reviews_data)
    products_df = pd.DataFrame(products_metadata_data)
    
    return reviews_df, products_df

# --- 2. Data Preprocessing Layer ---

class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = set(stopwords.words('english'))

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation and numbers
        return text

    def tokenize_and_lemmatize(self, text):
        tokens = nltk.word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stopwords]
        return " ".join(tokens)

    def fit_transform(self, series):
        cleaned = series.apply(self.clean_text)
        processed = cleaned.apply(self.tokenize_and_lemmatize)
        return processed

# --- Pipeline for Numerical and Categorical Features ---
def create_numerical_categorical_pipeline(numerical_cols, categorical_cols):
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ], 
        remainder='passthrough'
    )
    return preprocessor

# --- 3. Sentiment Analysis Model Layer ---

def train_sentiment_model(reviews_df):
    text_preprocessor_obj = TextPreprocessor()
    processed_reviews = text_preprocessor_obj.fit_transform(reviews_df['review_text'])
    
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    X_text = tfidf_vectorizer.fit_transform(processed_reviews)
    y_sentiment = reviews_df['sentiment_label']
    
    # Map sentiment labels to numerical values for classification
    sentiment_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
    y_sentiment_encoded = y_sentiment.map(sentiment_mapping)
    
    sentiment_model = LogisticRegression(max_iter=1000, random_state=42)
    sentiment_model.fit(X_text, y_sentiment_encoded)
    
    return sentiment_model, tfidf_vectorizer, text_preprocessor_obj, sentiment_mapping

def predict_sentiment(text_preprocessor_obj, tfidf_vectorizer, sentiment_model, sentiment_mapping, review_text):
    cleaned_text = text_preprocessor_obj.clean_text(review_text)
    processed_text = text_preprocessor_obj.tokenize_and_lemmatize(cleaned_text)
    text_vector = tfidf_vectorizer.transform([processed_text])
    sentiment_prediction_encoded = sentiment_model.predict(text_vector)[0]
    
    # Inverse map to get sentiment label
    inverse_sentiment_mapping = {v: k for k, v in sentiment_mapping.items()}
    return inverse_sentiment_mapping[sentiment_prediction_encoded]

# --- 4. Recommendation Engine Layer ---

def build_recommendation_features(products_df, reviews_df, sentiment_model, tfidf_vectorizer, text_preprocessor_obj, sentiment_mapping, numerical_cols, categorical_cols):
    # Calculate sentiment for all reviews and aggregate per product
    processed_reviews_for_sentiment = text_preprocessor_obj.fit_transform(reviews_df['review_text'])
    review_vectors = tfidf_vectorizer.transform(processed_reviews_for_sentiment)
    sentiment_predictions_encoded = sentiment_model.predict(review_vectors)
    
    inverse_sentiment_mapping = {v: k for k, v in sentiment_mapping.items()}
    sentiment_predictions_labels = [inverse_sentiment_mapping[s] for s in sentiment_predictions_encoded]
    reviews_df['predicted_sentiment'] = sentiment_predictions_labels
    
    # Convert sentiment labels to numerical for aggregation (e.g., negative=-1, neutral=0, positive=1)
    sentiment_scores = reviews_df['predicted_sentiment'].map({'negative': -1, 'neutral': 0, 'positive': 1})
    reviews_df['sentiment_score'] = sentiment_scores

    # Aggregate sentiment per product
    product_sentiment_agg = reviews_df.groupby('product_id')['sentiment_score'].mean().reset_index()
    product_sentiment_agg.rename(columns={'sentiment_score': 'avg_sentiment_score'}, inplace=True)
    
    # Merge with products_df
    products_with_sentiment_df = pd.merge(products_df, product_sentiment_agg, on='product_id', how='left')
    products_with_sentiment_df['avg_sentiment_score'] = products_with_sentiment_df['avg_sentiment_score'].fillna(0) # Fill NaN for products without reviews
    
    # Prepare features for content-based recommendation
    # Use price, rating, avg_sentiment_score, and one-hot encoded category/brand
    
    # Define numerical and categorical columns for the preprocessor
    rec_numerical_cols = numerical_cols + ['rating', 'avg_sentiment_score']
    rec_categorical_cols = categorical_cols
    
    rec_preprocessor = create_numerical_categorical_pipeline(rec_numerical_cols, rec_categorical_cols)
    
    # Fit and transform the product features
    # Ensure products_with_sentiment_df has all the required columns for rec_preprocessor
    # For simplicity, we'll align columns based on the original products_df and the added sentiment
    features_df = products_with_sentiment_df[['product_id'] + rec_numerical_cols + rec_categorical_cols]
    
    # Drop product_id before fitting/transforming as it's not a feature
    X_rec_features = rec_preprocessor.fit_transform(features_df.drop(columns=['product_id']))
    
    return X_rec_features, products_with_sentiment_df, rec_preprocessor

def get_recommendations(product_id, X_rec_features, products_with_sentiment_df, top_n=5):
    product_index = products_with_sentiment_df[products_with_sentiment_df['product_id'] == product_id].index[0]
    product_vector = X_rec_features[product_index].reshape(1, -1)
    
    similarities = cosine_similarity(product_vector, X_rec_features)[0]
    
    # Get indices of top_n most similar products (excluding itself)
    similar_product_indices = similarities.argsort()[-top_n-1:-1][::-1]
    
    recommended_products = products_with_sentiment_df.iloc[similar_product_indices]
    return recommended_products[['product_id', 'product_name', 'category', 'brand', 'price', 'rating', 'avg_sentiment_score']]


# --- 5. API and Deployment Layer (Conceptual) ---

app = FastAPI()

# Global variables to hold models and preprocessors
sentiment_model_global = None
tfidf_vectorizer_global = None
text_preprocessor_global = None
sentiment_mapping_global = None

X_rec_features_global = None
products_with_sentiment_df_global = None
rec_preprocessor_global = None


class ReviewInput(BaseModel):
    review_text: str

class RecommendationInput(BaseModel):
    product_id: int

@app.on_event("startup")
async def load_models():
    global sentiment_model_global, tfidf_vectorizer_global, text_preprocessor_global, sentiment_mapping_global
    global X_rec_features_global, products_with_sentiment_df_global, rec_preprocessor_global
    
    print("Loading models and data...")
    reviews_df, products_df = load_data()
    
    # Train/load sentiment components
    sentiment_model_global, tfidf_vectorizer_global, text_preprocessor_global, sentiment_mapping_global = train_sentiment_model(reviews_df)
    
    # Define numerical and categorical columns for general product metadata (excluding sentiment for now)
    numerical_cols = ['price'] 
    categorical_cols = ['category', 'brand']

    # Build recommendation features
    X_rec_features_global, products_with_sentiment_df_global, rec_preprocessor_global = build_recommendation_features(
        products_df, reviews_df, sentiment_model_global, tfidf_vectorizer_global, 
        text_preprocessor_global, sentiment_mapping_global, numerical_cols, categorical_cols
    )

    # For persistence, uncomment to save/load (requires appropriate directories)
    # joblib.dump(sentiment_model_global, 'sentiment_model.joblib')
    # joblib.dump(tfidf_vectorizer_global, 'tfidf_vectorizer.joblib')
    # joblib.dump(text_preprocessor_global, 'text_preprocessor.joblib')
    # joblib.dump(sentiment_mapping_global, 'sentiment_mapping.joblib')
    # joblib.dump(X_rec_features_global, 'X_rec_features.joblib')
    # joblib.dump(products_with_sentiment_df_global, 'products_with_sentiment_df.joblib')
    # joblib.dump(rec_preprocessor_global, 'rec_preprocessor.joblib')

    print("Models and data loaded.")

@app.post("/analyze_sentiment/")
async def analyze_sentiment(review_input: ReviewInput):
    if sentiment_model_global is None:
        return {"error": "Model not loaded yet. Please try again in a moment."}
    
    sentiment = predict_sentiment(
        text_preprocessor_global, 
        tfidf_vectorizer_global, 
        sentiment_model_global, 
        sentiment_mapping_global, 
        review_input.review_text
    )
    return {"review_text": review_input.review_text, "predicted_sentiment": sentiment}

@app.post("/recommend_products/")
async def recommend_products(rec_input: RecommendationInput):
    if X_rec_features_global is None:
        return {"error": "Recommendation engine not ready yet. Please try again in a moment."}

    # Check if the product_id exists in our dataset
    if rec_input.product_id not in products_with_sentiment_df_global['product_id'].values:
        return {"error": f"Product ID {rec_input.product_id} not found."}

    recommendations = get_recommendations(
        rec_input.product_id, 
        X_rec_features_global, 
        products_with_sentiment_df_global
    )
    return recommendations.to_dict(orient="records")

@app.get("/")
async def read_root():
    return {"message": "E-commerce ML System: Sentiment Analysis and Recommendations"}

# To run the FastAPI app:
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
