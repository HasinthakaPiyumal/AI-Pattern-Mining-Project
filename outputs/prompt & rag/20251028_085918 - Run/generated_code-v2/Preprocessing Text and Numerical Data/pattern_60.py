
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import re

# Download necessary NLTK data (if not already downloaded)
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except nltk.downloader.DownloadError:
    nltk.download('wordnet')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except nltk.downloader.DownloadError:
    nltk.download('averaged_perceptron_tagger')

# --- 1. Data Ingestion (Simulated) ---
def simulate_data():
    np.random.seed(42)
    data_size = 1000
    products = ['Laptop', 'Smartphone', 'Headphones', 'Smartwatch', 'Tablet']
    categories = ['Electronics', 'Gadgets']

    df = pd.DataFrame({
        'product_id': range(data_size),
        'product_name': np.random.choice(products, data_size),
        'category': np.random.choice(categories, data_size),
        'price': np.random.normal(loc=500, scale=200, size=data_size).round(2),
        'rating': np.random.randint(1, 6, data_size),
        'num_reviews': np.random.randint(10, 500, data_size),
        'review_text': ["This product is amazing! I love it.",
                        "Terrible quality, broke after a week.",
                        "It's okay, nothing special.",
                        "Good value for money.",
                        "Very disappointed, false advertising."] * (data_size // 5),
        'brand': np.random.choice(['BrandA', 'BrandB', 'BrandC'], data_size)
    })

    # Introduce some missing values and outliers for demonstration
    df.loc[np.random.choice(df.index, 50, replace=False), 'price'] = np.nan
    df.loc[np.random.choice(df.index, 20, replace=False), 'rating'] = np.nan
    df.loc[np.random.choice(df.index, 10, replace=False), 'review_text'] = np.nan
    df.loc[np.random.choice(df.index, 5, replace=False), 'category'] = np.nan
    df.loc[np.random.choice(df.index, 5, replace=False), 'price'] = 5000 # outlier

    # Create a target for sentiment (0=negative, 1=neutral, 2=positive)
    df['sentiment_target'] = df['review_text'].apply(lambda x: 2 if 'amazing' in str(x) or 'love' in str(x) else (0 if 'terrible' in str(x) or 'disappointed' in str(x) else 1))

    print("Simulated Data Head:")
    print(df.head())
    print("\nSimulated Data Info:")
    print(df.info())
    return df

# --- 2. Data Preprocessing - Numerical ---
# --- 3. Data Preprocessing - Text ---

class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = text.split()
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        return " ".join(tokens)

# --- Main processing and model training ---
def run_pipeline(df):
    # Separate features and targets
    X = df.drop(columns=['price', 'sentiment_target'])
    y_price = df['price']
    y_sentiment = df['sentiment_target']

    # Split data for sentiment analysis training
    X_train_sentiment, X_test_sentiment, y_train_sentiment, y_test_sentiment = train_test_split(
        X[['review_text']], y_sentiment, test_size=0.2, random_state=42, stratify=y_sentiment
    )

    # Text preprocessing and TF-IDF for Sentiment Analysis
    text_preprocessor = TextPreprocessor()
    text_transformer_sentiment = Pipeline([
        ('text_clean', TextPreprocessor()),
        ('tfidf', TfidfVectorizer(max_features=1000))
    ])

    # Sentiment Analysis Model Pipeline
    sentiment_pipeline = Pipeline([
        ('text_features', text_transformer_sentiment),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])

    print("\nTraining Sentiment Analysis Model...")
    sentiment_pipeline.fit(X_train_sentiment['review_text'], y_train_sentiment)
    sentiment_accuracy = sentiment_pipeline.score(X_test_sentiment['review_text'], y_test_sentiment)
    print(f"Sentiment Model Accuracy: {sentiment_accuracy:.4f}")
    joblib.dump(sentiment_pipeline, 'sentiment_model.pkl')
    print("Sentiment model saved as sentiment_model.pkl")

    # Generate sentiment predictions for the entire dataset to use as a feature
    df['predicted_sentiment'] = sentiment_pipeline.predict(df['review_text'].fillna(''))

    # Numerical and Categorical features for Price Prediction
    numerical_features = ['rating', 'num_reviews', 'predicted_sentiment']
    categorical_features = ['product_name', 'category', 'brand']

    # Preprocessing for numerical and categorical features
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
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Price Prediction Model Pipeline
    # Ensure y_price has no NaNs for training the regressor
    df_price_train = df.dropna(subset=['price'])
    X_price_train = df_price_train.drop(columns=['price', 'sentiment_target', 'review_text'])
    y_price_train = df_price_train['price']

    price_prediction_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])

    print("\nTraining Price Prediction Model...")
    price_prediction_pipeline.fit(X_price_train, y_price_train)
    price_r2 = price_prediction_pipeline.score(X_price_train, y_price_train) # Using training R2 for simplicity
    print(f"Price Prediction Model R2 Score (on training data): {price_r2:.4f}")
    joblib.dump(price_prediction_pipeline, 'price_prediction_model.pkl')
    print("Price prediction model saved as price_prediction_model.pkl")

    print("\n--- Demonstrating Prediction on New Data ---")
    # Simulate a new product entry
    new_data = pd.DataFrame({
        'product_id': [9999],
        'product_name': ['New Smartphone X'],
        'category': ['Electronics'],
        'price': [np.nan], # Price to be predicted
        'rating': [4],
        'num_reviews': [150],
        'review_text': ["This new phone is fantastic! Great features and camera."],
        'brand': ['BrandD'],
        'sentiment_target': [np.nan] # Not used for new prediction input
    })

    # Preprocess the new review text for sentiment prediction
    new_data['predicted_sentiment'] = sentiment_pipeline.predict(new_data['review_text'].fillna(''))

    # Select features for price prediction from new_data
    X_new_price = new_data.drop(columns=['price', 'sentiment_target', 'review_text', 'product_id'])

    # Make price prediction
    predicted_price = price_prediction_pipeline.predict(X_new_price)[0]
    print(f"Predicted Sentiment for new review: {new_data['predicted_sentiment'].iloc[0]} (0: Negative, 1: Neutral, 2: Positive)")
    print(f"Predicted Price for 'New Smartphone X': ${predicted_price:.2f}")


if __name__ == "__main__":
    data = simulate_data()
    run_pipeline(data)
