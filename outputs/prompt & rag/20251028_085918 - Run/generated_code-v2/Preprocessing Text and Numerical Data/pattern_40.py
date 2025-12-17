import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except nltk.downloader.DownloadError:
    nltk.download('wordnet')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')


class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000)

    def preprocess_text(self, text):
        tokens = word_tokenize(text.lower())
        lemmas = [self.lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in self.stop_words]
        return ' '.join(lemmas)

    def fit_transform(self, texts):
        processed_texts = [self.preprocess_text(text) for text in texts]
        return self.tfidf_vectorizer.fit_transform(processed_texts)

    def transform(self, texts):
        processed_texts = [self.preprocess_text(text) for text in texts]
        return self.tfidf_vectorizer.transform(processed_texts)


class NumericalPreprocessor:
    def __init__(self):
        self.imputer = SimpleImputer(strategy='mean')
        self.scaler = StandardScaler()
        self.one_hot_encoder = None  # Initialize only when categorical features are identified

    def fit_transform(self, df, numerical_cols, categorical_cols=None):
        df_processed = df.copy()

        # Impute numerical columns
        df_processed[numerical_cols] = self.imputer.fit_transform(df_processed[numerical_cols])

        # Scale numerical columns
        df_processed[numerical_cols] = self.scaler.fit_transform(df_processed[numerical_cols])

        if categorical_cols:
            self.one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            encoded_features = self.one_hot_encoder.fit_transform(df_processed[categorical_cols])
            encoded_df = pd.DataFrame(encoded_features, columns=self.one_hot_encoder.get_feature_names_out(categorical_cols), index=df_processed.index)
            df_processed = pd.concat([df_processed.drop(columns=categorical_cols), encoded_df], axis=1)
        
        return df_processed

    def transform(self, df, numerical_cols, categorical_cols=None):
        df_processed = df.copy()

        # Impute numerical columns
        df_processed[numerical_cols] = self.imputer.transform(df_processed[numerical_cols])

        # Scale numerical columns
        df_processed[numerical_cols] = self.scaler.transform(df_processed[numerical_cols])

        if categorical_cols and self.one_hot_encoder:
            encoded_features = self.one_hot_encoder.transform(df_processed[categorical_cols])
            encoded_df = pd.DataFrame(encoded_features, columns=self.one_hot_encoder.get_feature_names_out(categorical_cols), index=df_processed.index)
            df_processed = pd.concat([df_processed.drop(columns=categorical_cols), encoded_df], axis=1)
        elif categorical_cols and not self.one_hot_encoder:
            raise ValueError("OneHotEncoder was not fitted. Call fit_transform first.")

        return df_processed


class SentimentAnalyzer:
    def __init__(self, model=LogisticRegression(max_iter=1000, random_state=42)):
        self.model = model

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        return accuracy_score(y_test, predictions)


class ProductRecommender:
    def __init__(self):
        self.product_features = None
        self.product_ids = None

    def fit(self, product_df_processed, product_id_col='product_id'):
        self.product_ids = product_df_processed[product_id_col]
        self.product_features = product_df_processed.drop(columns=[product_id_col]).values

    def recommend_products(self, user_preferred_product_id, top_n=5):
        if user_preferred_product_id not in self.product_ids.values:
            return []

        idx = self.product_ids[self.product_ids == user_preferred_product_id].index[0]
        target_product_features = self.product_features[idx].reshape(1, -1)

        similarities = cosine_similarity(target_product_features, self.product_features).flatten()
        
        # Exclude the product itself from recommendations
        similar_indices = similarities.argsort()[-top_n-1:-1][::-1]
        
        recommended_product_ids = [self.product_ids.iloc[i] for i in similar_indices]
        return recommended_product_ids


if __name__ == "__main__":
    # Simulate raw data
    print("Simulating raw data...")
    customer_reviews_data = {
        'review_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'product_id': [101, 102, 101, 103, 102, 104, 101, 105, 103, 104],
        'review_text': [
            "This product is amazing! I love it so much.",
            "Terrible quality, very disappointed.",
            "Good value for money, highly recommend.",
            "It's okay, nothing special.",
            "Worst purchase ever, completely broken.",
            "Fantastic item, arrived quickly.",
            "Mediocre, could be better.",
            "Absolutely perfect! Will buy again.",
            "Not bad, but expected more.",
            "Very durable and useful."
        ],
        'sentiment': ['positive', 'negative', 'positive', 'neutral', 'negative', 'positive', 'neutral', 'positive', 'neutral', 'positive']
    }
    product_metadata_data = {
        'product_id': [101, 102, 103, 104, 105],
        'price': [25.99, 12.50, 49.99, 7.80, 150.00],
        'rating': [4.5, 2.0, 4.0, 3.5, 5.0],
        'category': ['Electronics', 'Books', 'Electronics', 'Home Goods', 'Apparel'],
        'stock_quantity': [100, 500, 75, 200, 30]
    }

    customer_reviews_df = pd.DataFrame(customer_reviews_data)
    product_metadata_df = pd.DataFrame(product_metadata_data)

    # 1. Data Preprocessing
    print("\n--- Data Preprocessing ---")
    text_preprocessor = TextPreprocessor()
    numerical_preprocessor = NumericalPreprocessor()

    # Preprocess text reviews
    print("Preprocessing text reviews...")
    X_reviews_processed = text_preprocessor.fit_transform(customer_reviews_df['review_text'])
    y_sentiment = customer_reviews_df['sentiment']
    
    # Preprocess numerical product metadata
    print("Preprocessing numerical product metadata...")
    numerical_cols = ['price', 'rating', 'stock_quantity']
    categorical_cols = ['category']
    product_metadata_processed_df = numerical_preprocessor.fit_transform(
        product_metadata_df.copy(), numerical_cols, categorical_cols
    )
    product_metadata_processed_df['product_id'] = product_metadata_df['product_id'] # Add product_id back
    
    # 2. Sentiment Analysis
    print("\n--- Sentiment Analysis ---")
    sentiment_analyzer = SentimentAnalyzer()

    # Split data for sentiment analysis
    X_train_sentiment, X_test_sentiment, y_train_sentiment, y_test_sentiment = train_test_split(
        X_reviews_processed, y_sentiment, test_size=0.2, random_state=42, stratify=y_sentiment
    )

    print("Training sentiment model...")
    sentiment_analyzer.train(X_train_sentiment, y_train_sentiment)

    accuracy = sentiment_analyzer.evaluate(X_test_sentiment, y_test_sentiment)
    print(f"Sentiment Model Accuracy: {accuracy:.2f}")

    # Predict sentiment for a new review
    new_review = "This is an amazing product, really happy with it!"
    processed_new_review = text_preprocessor.transform([new_review])
    predicted_sentiment = sentiment_analyzer.predict(processed_new_review)[0]
    print(f"Predicted sentiment for \"{new_review}\": {predicted_sentiment}")

    # 3. Product Recommendation
    print("\n--- Product Recommendation ---")
    product_recommender = ProductRecommender()
    product_recommender.fit(product_metadata_processed_df.copy(), product_id_col='product_id')

    # Get recommendations for a product (e.g., product_id 101)
    target_product_id = 101
    recommended_products = product_recommender.recommend_products(target_product_id, top_n=2)
    print(f"Recommended products for product {target_product_id}: {recommended_products}")

    target_product_id = 102
    recommended_products = product_recommender.recommend_products(target_product_id, top_n=3)
    print(f"Recommended products for product {target_product_id}: {recommended_products}")
