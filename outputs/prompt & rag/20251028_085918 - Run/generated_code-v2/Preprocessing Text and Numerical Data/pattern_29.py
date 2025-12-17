import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity

# Download necessary NLTK data
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

class DataPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.tfidf_vectorizer = None
        self.numerical_imputer = None
        self.numerical_scaler = None

    def preprocess_text(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        tokens = nltk.word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in self.stop_words]
        return " ".join(tokens)

    def fit_transform_text_vectorizer(self, texts):
        processed_texts = [self.preprocess_text(text) for text in texts]
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000) # Limit features for simplicity
        return self.tfidf_vectorizer.fit_transform(processed_texts)

    def transform_text_vectorizer(self, texts):
        if self.tfidf_vectorizer is None:
            raise RuntimeError("TF-IDF vectorizer not fitted. Call fit_transform_text_vectorizer first.")
        processed_texts = [self.preprocess_text(text) for text in texts]
        return self.tfidf_vectorizer.transform(processed_texts)

    def fit_transform_numerical(self, df, numerical_cols):
        self.numerical_imputer = SimpleImputer(strategy='median')
        df_imputed = pd.DataFrame(self.numerical_imputer.fit_transform(df[numerical_cols]), columns=numerical_cols, index=df.index)
        
        self.numerical_scaler = StandardScaler()
        df_scaled = pd.DataFrame(self.numerical_scaler.fit_transform(df_imputed), columns=numerical_cols, index=df.index)
        return df_scaled

    def transform_numerical(self, df, numerical_cols):
        if self.numerical_imputer is None or self.numerical_scaler is None:
            raise RuntimeError("Numerical imputer/scaler not fitted. Call fit_transform_numerical first.")
        df_imputed = pd.DataFrame(self.numerical_imputer.transform(df[numerical_cols]), columns=numerical_cols, index=df.index)
        df_scaled = pd.DataFrame(self.numerical_scaler.transform(df_imputed), columns=numerical_cols, index=df.index)
        return df_scaled

class SentimentAnalyzer:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        print("Sentiment Analysis Classification Report:")
        print(classification_report(y_test, y_pred))

class RecommendationEngine:
    def __init__(self, preprocessed_product_features, product_ids):
        self.product_features = preprocessed_product_features
        self.product_ids = product_ids
        self.product_id_to_index = {pid: i for i, pid in enumerate(product_ids)}

    def recommend(self, user_history_product_ids, num_recommendations=5):
        if not user_history_product_ids:
            print("No purchase history provided for recommendations.")
            return []

        user_product_indices = [self.product_id_to_index[pid] for pid in user_history_product_ids if pid in self.product_id_to_index]
        if not user_product_indices:
            print("None of the user's historical products are in the current product catalog.")
            return []

        # Aggregate user's past product features (e.g., average)
        user_profile = np.mean([self.product_features[idx] for idx in user_product_indices], axis=0).reshape(1, -1)

        # Calculate similarity with all products
        similarities = cosine_similarity(user_profile, self.product_features).flatten()

        # Exclude already purchased products from recommendations
        for idx in user_product_indices:
            similarities[idx] = -1  # Set similarity to a low value

        # Get top recommendations
        recommended_product_indices = similarities.argsort()[-num_recommendations:][::-1]
        recommended_products = [self.product_ids[idx] for idx in recommended_product_indices if similarities[idx] > 0]
        
        return recommended_products

# --- Main Application Workflow ---
if __name__ == "__main__":
    # 1. Simulate Data Ingestion
    print("\n--- Simulating Data Ingestion ---")
    reviews_data = {
        'review_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'product_id': [101, 102, 101, 103, 102, 104, 101, 105, 103, 104],
        'review_text': [
            "This product is amazing! Highly recommend.",
            "It's okay, nothing special. Could be better.",
            "Absolutely love it. Great quality and fast shipping.",
            "Very disappointing, broke after a week.",
            "Decent for the price.",
            "Fantastic value! So happy with this purchase.",
            "Good product, but the color is not what I expected.",
            "Worst purchase ever. Complete waste of money.",
            "Works as described, satisfied.",
            "Not bad, but I've seen better options."
        ],
        'sentiment_label': ['positive', 'neutral', 'positive', 'negative', 'neutral', 'positive', 'positive', 'negative', 'positive', 'neutral'] # For training
    }
    products_data = {
        'product_id': [101, 102, 103, 104, 105],
        'price': [50.0, 30.0, 120.0, 75.0, 200.0],
        'rating': [4.5, 3.0, 2.0, 4.8, 1.5],
        'num_reviews': [150, 80, 25, 200, 10],
        'category': ['Electronics', 'Home', 'Electronics', 'Books', 'Home']
    }
    purchase_history_data = {
        'user_id': [1, 1, 2, 2, 3, 3, 4, 4, 5],
        'product_id': [101, 102, 103, 101, 104, 102, 105, 101, 103]
    }

    reviews_df = pd.DataFrame(reviews_data)
    products_df = pd.DataFrame(products_data)
    purchase_history_df = pd.DataFrame(purchase_history_data)

    print("Raw Reviews (first 2 rows):\n", reviews_df.head(2))
    print("Raw Products (first 2 rows):\n", products_df.head(2))

    # 2. Data Preprocessing Module
    print("\n--- Preprocessing Data ---")
    preprocessor = DataPreprocessor()

    # Preprocess text data (reviews)
    X_text = preprocessor.fit_transform_text_vectorizer(reviews_df['review_text'])
    y_sentiment = reviews_df['sentiment_label']

    # Preprocess numerical data (products)
    numerical_cols = ['price', 'rating', 'num_reviews']
    preprocessed_products_numerical = preprocessor.fit_transform_numerical(products_df, numerical_cols)
    
    # For simplicity, combine product numerical features with product_id for recommendation engine
    # In a real scenario, you might also one-hot encode categorical features like 'category' and concatenate
    product_features_for_reco = preprocessed_products_numerical.values
    product_ids_for_reco = products_df['product_id'].tolist()

    print(f"Shape of preprocessed text features: {X_text.shape}")
    print(f"Shape of preprocessed numerical product features: {preprocessed_products_numerical.shape}")

    # 3. Sentiment Analysis Module
    print("\n--- Training Sentiment Analysis Model ---")
    X_train_sentiment, X_test_sentiment, y_train_sentiment, y_test_sentiment = train_test_split(
        X_text, y_sentiment, test_size=0.2, random_state=42, stratify=y_sentiment
    )

    sentiment_analyzer = SentimentAnalyzer()
    sentiment_analyzer.train(X_train_sentiment, y_train_sentiment)
    sentiment_analyzer.evaluate(X_test_sentiment, y_test_sentiment)

    # Example: Predict sentiment for a new review
    new_review = "This item is absolutely fantastic, very sturdy and useful!"
    processed_new_review = preprocessor.transform_text_vectorizer([new_review])
    predicted_sentiment = sentiment_analyzer.predict(processed_new_review)[0]
    print(f"\nNew review: \"{new_review}\" -> Predicted Sentiment: {predicted_sentiment}")

    # 4. Recommendation Engine Module
    print("\n--- Generating Product Recommendations ---")
    # In a real system, you'd feed in more sophisticated product features (e.g., product embeddings, aggregated sentiment)
    # Here, we use only the numerical features we processed.

    recommendation_engine = RecommendationEngine(product_features_for_reco, product_ids_for_reco)

    # Example: Get recommendations for user_id = 1 (purchased products: 101, 102)
    user_1_purchases = purchase_history_df[purchase_history_df['user_id'] == 1]['product_id'].tolist()
    print(f"\nUser 1's purchase history: {user_1_purchases}")
    recommended_for_user_1 = recommendation_engine.recommend(user_1_purchases)
    print(f"Recommended products for User 1: {recommended_for_user_1}")

    # Example: Get recommendations for user_id = 4 (purchased products: 105, 101)
    user_4_purchases = purchase_history_df[purchase_history_df['user_id'] == 4]['product_id'].tolist()
    print(f"\nUser 4's purchase history: {user_4_purchases}")
    recommended_for_user_4 = recommendation_engine.recommend(user_4_purchases)
    print(f"Recommended products for User 4: {recommended_for_user_4}")

    # Example: Get recommendations for a new user with no history (or just based on popular items)
    print("\nRecommendations for a user with no purchase history (will return empty as it's content-based): ")
    new_user_recommendations = recommendation_engine.recommend([])
    print(new_user_recommendations)

