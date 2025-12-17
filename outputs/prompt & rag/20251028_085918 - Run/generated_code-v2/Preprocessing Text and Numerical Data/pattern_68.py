import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Download NLTK resources
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

# 1. Data Simulation
def simulate_data(num_samples=1000):
    np.random.seed(42)
    data = {
        'review_id': range(1, num_samples + 1),
        'customer_id': np.random.randint(100, 500, num_samples),
        'product_id': np.random.randint(1000, 2000, num_samples),
        'review_text': [
            "This product is amazing! I love it so much.",
            "It's okay, not the best, but decent for the price.",
            "Terrible quality, completely broken after a week.",
            "Highly recommend, excellent value and fast delivery.",
            "Disappointed with the purchase, will not buy again.",
            "Good product, meets expectations.",
            "The worst experience ever. Avoid at all costs.",
            "Fantastic! Exceeded all my expectations.",
            "Mediocre, could be better.",
            "Absolutely brilliant, a game changer!"
        ] * (num_samples // 10),
        'rating': np.random.randint(1, 6, num_samples),
        'price': np.random.uniform(10, 500, num_samples),
        'product_category_id': np.random.choice([1, 2, 3, 4, 5], num_samples)
    }
    df = pd.DataFrame(data)

    # Introduce some missing values and noise
    missing_indices_text = np.random.choice(df.index, int(num_samples * 0.02), replace=False)
    df.loc[missing_indices_text, 'review_text'] = np.nan
    missing_indices_price = np.random.choice(df.index, int(num_samples * 0.03), replace=False)
    df.loc[missing_indices_price, 'price'] = np.nan

    # Simulate review sentiments for sentiment analysis (for demonstration)
    sentiment_map = {
        'amazing': 'positive', 'love': 'positive', 'excellent': 'positive', 'highly recommend': 'positive', 'fantastic': 'positive', 'brilliant': 'positive',
        'okay': 'neutral', 'decent': 'neutral', 'good': 'neutral', 'meets expectations': 'neutral', 'mediocre': 'neutral',
        'terrible': 'negative', 'broken': 'negative', 'disappointed': 'negative', 'worst': 'negative', 'avoid': 'negative'
    }
    df['sentiment'] = df['review_text'].astype(str).apply(lambda x: 'positive' if any(word in x.lower() for word in ['amazing', 'love', 'excellent', 'highly recommend', 'fantastic', 'brilliant']) else 
                                                    ('negative' if any(word in x.lower() for word in ['terrible', 'broken', 'disappointed', 'worst', 'avoid']) else 'neutral'))
    
    return df

# 2. Preprocessing Layer

# Text Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if isinstance(text, float) and np.isnan(text): # Handle NaN values
        return ""
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(tokens)

# Define the full preprocessing pipeline

# Text pipeline
text_transformer = Pipeline([
    ('tfidf', TfidfVectorizer(preprocessor=preprocess_text, max_features=5000)) # Adjust max_features as needed
])

# Numerical pipeline
numerical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Categorical pipeline (for product_category_id if treated as categorical)
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessors using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('text_pipeline', text_transformer, 'review_text'),
        ('num_pipeline', numerical_transformer, ['price', 'rating']),
        ('cat_pipeline', categorical_transformer, ['product_category_id'])
    ],
    remainder='passthrough' # Keep other columns if any, or 'drop'
)

# Main execution
if __name__ == "__main__":
    print("Simulating raw data...")
    raw_df = simulate_data(num_samples=1000)
    print("Raw Data Sample:\n", raw_df.head())
    print("Raw Data Info:\n")
    raw_df.info()

    # Separate features and target for sentiment analysis
    # For sentiment analysis, we only need the review_text and its sentiment label
    sentiment_df = raw_df.dropna(subset=['review_text'])
    X_text = sentiment_df['review_text']
    y_sentiment = sentiment_df['sentiment']

    # For recommendation, we will use all numerical/categorical features after preprocessing
    X_recommendation_features = raw_df[['review_text', 'rating', 'price', 'product_category_id']]

    print("\nFitting and transforming data with preprocessing pipeline...")
    # Fit and transform the full dataset for combined features
    preprocessed_data_combined = preprocessor.fit_transform(X_recommendation_features)

    # For sentiment analysis, we specifically need the TF-IDF output
    # Create a separate pipeline for just text preprocessing for sentiment model
    text_only_pipeline = Pipeline([
        ('text_prep', text_transformer)
    ])
    X_sentiment_preprocessed = text_only_pipeline.fit_transform(X_text.to_frame())

    print("\nShape of preprocessed data (combined features for recommendation):", preprocessed_data_combined.shape)
    print("Shape of preprocessed text data (for sentiment analysis):", X_sentiment_preprocessed.shape)

    # 3. Model Application (Illustrative)

    # Sentiment Analysis
    print("\nPerforming illustrative Sentiment Analysis...")
    # We need to ensure y_sentiment aligns with X_sentiment_preprocessed rows
    # Since we dropped NaNs for sentiment_df, y_sentiment is already aligned.

    # Filter out 'neutral' sentiments for a clearer binary classification example if desired,
    # or keep it as multi-class. For simplicity, we'll keep as multi-class now.

    # Ensure X_sentiment_preprocessed is not empty after NaN handling
    if X_sentiment_preprocessed.shape[0] > 0 and len(y_sentiment) > 0:
        X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
            X_sentiment_preprocessed, y_sentiment, test_size=0.2, random_state=42, stratify=y_sentiment
        )

        sentiment_model = LogisticRegression(max_iter=1000)
        sentiment_model.fit(X_train_s, y_train_s)
        y_pred_s = sentiment_model.predict(X_test_s)

        print(f"Sentiment Analysis Accuracy: {accuracy_score(y_test_s, y_pred_s):.4f}")
    else:
        print("Not enough data for sentiment analysis after preprocessing.")

    # Product Recommendation Preparation
    print("\nShowcasing preprocessed numerical/categorical data for Product Recommendation...")
    # 'preprocessed_data_combined' contains features ready for a recommendation engine.
    # The exact form depends on the recommendation algorithm (e.g., collaborative filtering, content-based).
    # For content-based, these numerical features would directly describe products.
    
    # Example: Display a few rows of the preprocessed numerical/categorical features
    # Note: Column names are lost after ColumnTransformer, you'd typically handle this if needed for inspection
    # or pass through feature names if using custom transformers.

    # To convert back to DataFrame with meaningful columns for inspection is complex
    # due to TF-IDF sparse output and OneHotEncoder dynamic columns. 
    # For demonstration, we'll just show the raw array shape and a snippet.
    
    print("Preprocessed data snippet (first 5 rows, showing numerical/categorical parts):\n", preprocessed_data_combined[:5])
    print("\nThis preprocessed data is ready to be fed into a recommendation model (e.g., a clustering algorithm, matrix factorization, or a neural network for embeddings).")
    print("For example, you could calculate similarity between product vectors derived from this data.")