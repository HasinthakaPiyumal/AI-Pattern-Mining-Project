import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK data (if not already downloaded)
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

class DataPreprocessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stopwords = set(stopwords.words('english'))
        self.preprocessor = None
        self.tfidf_vectorizer_desc = None
        self.tfidf_vectorizer_review = None

    def _clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9\s]', '', text) # Remove punctuation and special characters
        text = re.sub(r'\d+', '', text) # Remove numbers
        tokens = word_tokenize(text)
        tokens = [self.stemmer.stem(word) for word in tokens if word not in self.stopwords]
        return ' '.join(tokens)

    def fit(self, df, numerical_cols, categorical_cols, text_description_col, text_review_col):
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Fit TF-IDF for product descriptions
        self.tfidf_vectorizer_desc = TfidfVectorizer(max_features=5000)
        df[text_description_col] = df[text_description_col].apply(self._clean_text)
        self.tfidf_vectorizer_desc.fit(df[text_description_col])

        # Fit TF-IDF for customer reviews
        self.tfidf_vectorizer_review = TfidfVectorizer(max_features=5000)
        df[text_review_col] = df[text_review_col].apply(self._clean_text)
        self.tfidf_vectorizer_review.fit(df[text_review_col])

        transformers = [
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
        
        self.preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')
        self.preprocessor.fit(df)

    def transform(self, df, numerical_cols, categorical_cols, text_description_col, text_review_col):
        # Apply text cleaning for descriptions and reviews first
        df_processed = df.copy()
        df_processed[text_description_col] = df_processed[text_description_col].apply(self._clean_text)
        df_processed[text_review_col] = df_processed[text_review_col].apply(self._clean_text)
        
        # Transform numerical and categorical features
        transformed_features = self.preprocessor.transform(df_processed)

        # Transform text features using fitted TF-IDF vectorizers
        desc_vectors = self.tfidf_vectorizer_desc.transform(df_processed[text_description_col]).toarray()
        review_vectors = self.tfidf_vectorizer_review.transform(df_processed[text_review_col]).toarray()
        
        # Combine all features
        return np.hstack((transformed_features[:, :-2], desc_vectors, review_vectors)) # Exclude original text columns from ColumnTransformer output

class SentimentAnalyzer:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)
        self.tfidf_vectorizer = None

    def train(self, X_text_reviews, y_sentiment):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        X_vectors = self.tfidf_vectorizer.fit_transform(X_text_reviews)
        self.model.fit(X_vectors, y_sentiment)

    def predict_sentiment(self, text_reviews):
        if self.tfidf_vectorizer is None:
            raise ValueError("Sentiment model not trained. Call 'train' first.")
        X_vectors = self.tfidf_vectorizer.transform(text_reviews)
        return self.model.predict(X_vectors)
    
    def predict_proba_sentiment(self, text_reviews):
        if self.tfidf_vectorizer is None:
            raise ValueError("Sentiment model not trained. Call 'train' first.")
        X_vectors = self.tfidf_vectorizer.transform(text_reviews)
        return self.model.predict_proba(X_vectors)

class RecommendationSystem:
    def __init__(self):
        self.product_features = None
        self.product_ids = None

    def fit(self, processed_features, product_ids):
        self.product_features = processed_features
        self.product_ids = product_ids

    def get_recommendations(self, target_product_id, top_n=5):
        if self.product_features is None or self.product_ids is None:
            raise ValueError("Recommendation system not fitted. Call 'fit' first.")

        if target_product_id not in self.product_ids:
            return [] # Product not found

        target_idx = np.where(self.product_ids == target_product_id)[0][0]
        target_vector = self.product_features[target_idx].reshape(1, -1)

        similarities = cosine_similarity(target_vector, self.product_features)
        # Get indices of most similar products, excluding itself
        similar_indices = similarities.argsort()[0][::-1]
        similar_indices = [idx for idx in similar_indices if idx != target_idx][:top_n]

        return [self.product_ids[idx] for idx in similar_indices]


if __name__ == "__main__":
    # 1. Simulate Data Loading
    data = {
        'product_id': ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7'],
        'price': [10.99, 25.50, 5.00, 120.00, 15.75, 8.99, 30.00],
        'rating': [4.5, 3.8, 4.9, 4.2, 3.5, 4.1, 4.6],
        'stock': [100, 50, 200, 10, 75, 120, 40],
        'category': ['Electronics', 'Books', 'Electronics', 'Clothing', 'Books', 'Home', 'Electronics'],
        'brand': ['BrandA', 'BrandB', 'BrandA', 'BrandC', 'BrandB', 'BrandD', 'BrandA'],
        'product_description': [
            'High-quality smartphone with advanced features.',
            'Bestselling novel by a renowned author, a must-read.',
            'Compact and powerful earbuds for on-the-go music.',
            'Stylish summer dress made from breathable fabric.',
            'Fantasy adventure book, part of a popular series.',
            'Durable kitchen blender for smoothies and more.',
            '4K Ultra HD Smart TV with stunning visuals.'
        ],
        'customer_review': [
            'Great phone, very fast!',
            'Story was okay, a bit slow.',
            'Amazing sound quality and comfortable fit.',
            'Dress fits perfectly, love the material.',
            'Good read but ending felt rushed.',
            'Blender works well, a bit noisy.',
            'Picture is incredible, easy to set up.'
        ],
        'review_sentiment_label': ['positive', 'neutral', 'positive', 'positive', 'neutral', 'negative', 'positive'] # For sentiment training
    }
    df = pd.DataFrame(data)

    numerical_cols = ['price', 'rating', 'stock']
    categorical_cols = ['category', 'brand']
    text_description_col = 'product_description'
    text_review_col = 'customer_review'
    product_id_col = 'product_id'

    # 2. Data Preprocessing
    print("\n--- Data Preprocessing ---")
    preprocessor = DataPreprocessor()
    
    # Fit the preprocessor (includes fitting TF-IDF for text cols)
    preprocessor.fit(
        df,
        numerical_cols,
        categorical_cols,
        text_description_col,
        text_review_col
    )

    # Transform the data
    processed_features = preprocessor.transform(
        df,
        numerical_cols,
        categorical_cols,
        text_description_col,
        text_review_col
    )
    print(f"Shape of processed features: {processed_features.shape}")
    print("Sample of processed features (first 2 rows):\n", processed_features[:2, :5]) # Display first 5 columns of first 2 rows

    # 3. Sentiment Analysis
    print("\n--- Sentiment Analysis ---")
    sentiment_analyzer = SentimentAnalyzer()
    
    # Prepare data for sentiment training
    X_reviews_train = df[text_review_col].apply(preprocessor._clean_text) # Use preprocessor's text cleaning
    y_sentiment_train = df['review_sentiment_label']

    # Train the sentiment model
    sentiment_analyzer.train(X_reviews_train, y_sentiment_train)
    print("Sentiment model trained successfully.")

    # Predict sentiment for new reviews (example)
    new_reviews = ["This product is amazing, highly recommend!", "Terrible quality, very disappointed.", "It's okay, nothing special."]
    cleaned_new_reviews = [preprocessor._clean_text(r) for r in new_reviews]
    predicted_sentiments = sentiment_analyzer.predict_sentiment(cleaned_new_reviews)
    predicted_probas = sentiment_analyzer.predict_proba_sentiment(cleaned_new_reviews)
    print(f"New Reviews: {new_reviews}")
    print(f"Predicted Sentiments: {predicted_sentiments}")
    print(f"Predicted Probabilities:\n{predicted_probas}")

    # 4. Recommendation System
    print("\n--- Recommendation System ---")
    recommender = RecommendationSystem()
    recommender.fit(processed_features, df[product_id_col].values)
    print("Recommendation system fitted.")

    # Get recommendations for a specific product (e.g., 'P1')
    target_product = 'P1'
    recommendations = recommender.get_recommendations(target_product, top_n=3)
    print(f"Top 3 recommendations for product '{target_product}': {recommendations}")

    target_product = 'P4'
    recommendations = recommender.get_recommendations(target_product, top_n=2)
    print(f"Top 2 recommendations for product '{target_product}': {recommendations}")