import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib
import re

# Download NLTK data if not already present
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

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def simulate_sentiment_data():
    data = {
        'review': [
            "This product is amazing! I love it so much.",
            "Terrible experience, utterly disappointed.",
            "It's okay, nothing special.",
            "Best purchase ever, highly recommend!",
            "Very bad quality, will return.",
            "Decent for the price, quite satisfied.",
            "Absolutely fantastic, exceeded expectations.",
            "Worst thing I've bought in years.",
            "Could be better, but it does the job.",
            "Highly effective and user-friendly."
        ],
        'sentiment': [
            'positive', 'negative', 'neutral', 'positive', 'negative',
            'positive', 'positive', 'negative', 'neutral', 'positive'
        ]
    }
    return pd.DataFrame(data)

def simulate_price_data():
    data = {
        'feature_1': [10, 20, 15, 25, 12, 18, 22, 11, 16, 23],
        'feature_2': [100, 150, 120, 180, 110, 140, 160, 105, 130, 170],
        'feature_3': [1, 0, 1, 1, 0, 1, 0, 0, 1, 1],
        'missing_feature': [5, None, 7, 8, None, 6, 9, 5, None, 7],
        'price': [110, 160, 130, 190, 120, 150, 170, 115, 140, 180]
    }
    return pd.DataFrame(data)

class SentimentAnalysis:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.model = LogisticRegression(max_iter=1000)

    def train(self, reviews, sentiments):
        processed_reviews = [preprocess_text(review) for review in reviews]
        X_vectorized = self.vectorizer.fit_transform(processed_reviews)
        self.model.fit(X_vectorized, sentiments)

    def predict(self, new_reviews):
        processed_new_reviews = [preprocess_text(review) for review in new_reviews]
        X_new_vectorized = self.vectorizer.transform(processed_new_reviews)
        return self.model.predict(X_new_vectorized)

    def save_models(self, vectorizer_path="tfidf_vectorizer.joblib", model_path="sentiment_model.joblib"):
        joblib.dump(self.vectorizer, vectorizer_path)
        joblib.dump(self.model, model_path)

    def load_models(self, vectorizer_path="tfidf_vectorizer.joblib", model_path="sentiment_model.joblib"):
        self.vectorizer = joblib.load(vectorizer_path)
        self.model = joblib.load(model_path)

class PricePrediction:
    def __init__(self):
        self.imputer = SimpleImputer(strategy='mean')
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(random_state=42)

    def train(self, features, prices):
        X_imputed = self.imputer.fit_transform(features)
        X_scaled = self.scaler.fit_transform(X_imputed)
        self.model.fit(X_scaled, prices)

    def predict(self, new_features):
        X_new_imputed = self.imputer.transform(new_features)
        X_new_scaled = self.scaler.transform(X_new_imputed)
        return self.model.predict(X_new_scaled)

    def save_models(self, imputer_path="imputer.joblib", scaler_path="scaler.joblib", model_path="price_model.joblib"):
        joblib.dump(self.imputer, imputer_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.model, model_path)

    def load_models(self, imputer_path="imputer.joblib", scaler_path="scaler.joblib", model_path="price_model.joblib"):
        self.imputer = joblib.load(imputer_path)
        self.scaler = joblib.load(scaler_path)
        self.model = joblib.load(model_path)

if __name__ == "__main__":
    # --- Sentiment Analysis Workflow ---
    print("\n--- Sentiment Analysis Workflow ---")
    sentiment_df = simulate_sentiment_data()
    X_sentiment_train, X_sentiment_test, y_sentiment_train, y_sentiment_test = train_test_split(
        sentiment_df['review'], sentiment_df['sentiment'], test_size=0.3, random_state=42
    )

    sa_system = SentimentAnalysis()
    sa_system.train(X_sentiment_train, y_sentiment_train)
    sa_system.save_models()
    print("Sentiment Analysis models trained and saved.")

    # Load and predict
    loaded_sa_system = SentimentAnalysis()
    loaded_sa_system.load_models()
    sentiment_predictions = loaded_sa_system.predict(X_sentiment_test)
    print(f"Test reviews: {list(X_sentiment_test)}")
    print(f"Actual sentiments: {list(y_sentiment_test)}")
    print(f"Predicted sentiments: {list(sentiment_predictions)}")
    print(f"Sentiment Accuracy: {accuracy_score(y_sentiment_test, sentiment_predictions):.2f}")

    # --- Price Prediction Workflow ---
    print("\n--- Price Prediction Workflow ---")
    price_df = simulate_price_data()
    features = price_df.drop('price', axis=1)
    prices = price_df['price']

    X_price_train, X_price_test, y_price_train, y_price_test = train_test_split(
        features, prices, test_size=0.3, random_state=42
    )

    pp_system = PricePrediction()
    pp_system.train(X_price_train, y_price_train)
    pp_system.save_models()
    print("Price Prediction models trained and saved.")

    # Load and predict
    loaded_pp_system = PricePrediction()
    loaded_pp_system.load_models()
    price_predictions = loaded_pp_system.predict(X_price_test)
    print(f"Test features:\n{X_price_test}")
    print(f"Actual prices: {list(y_price_test)}")
    print(f"Predicted prices: {[f'{p:.2f}' for p in price_predictions]}")
    print(f"Price Prediction MSE: {mean_squared_error(y_price_test, price_predictions):.2f}")
