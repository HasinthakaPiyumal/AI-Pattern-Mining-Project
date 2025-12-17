import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from transformers import pipeline

# Download necessary NLTK data
try:
    nltk.data.find("corpora/stopwords")
except nltk.downloader.DownloadError:
    nltk.download("stopwords")
try:
    nltk.data.find("tokenizers/punkt")
except nltk.downloader.DownloadError:
    nltk.download("punkt")

class DataPreprocessor:
    def __init__(self):
        self.numerical_imputer = SimpleImputer(strategy="mean")
        self.numerical_scaler = StandardScaler()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        self.stop_words = set(stopwords.words("english"))
        self.stemmer = PorterStemmer()

    def preprocess_numerical_data(self, df, numerical_cols):
        """Applies imputation and scaling to numerical columns."""
        df_processed = df.copy()
        # Check if numerical_cols exist and are not empty
        if numerical_cols and not df_processed[numerical_cols].empty:
            df_processed[numerical_cols] = self.numerical_imputer.fit_transform(df_processed[numerical_cols])
            df_processed[numerical_cols] = self.numerical_scaler.fit_transform(df_processed[numerical_cols])
        return df_processed

    def _clean_text(self, text):
        """Helper function to clean and stem text."""
        if not isinstance(text, str): 
            return ""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text) 
        tokens = nltk.word_tokenize(text)
        tokens = [self.stemmer.stem(word) for word in tokens if word not in self.stop_words]
        return " ".join(tokens)

    def preprocess_text_data(self, df, text_col):
        """Applies cleaning, tokenization, stop-word removal, stemming, and TF-IDF vectorization to text data."""
        df_processed = df.copy()
        df_processed["cleaned_text"] = df_processed[text_col].apply(self._clean_text)
        
        # Fit and transform TF-IDF
        # Only proceed if there is cleaned text to vectorize
        if not df_processed["cleaned_text"].empty and any(df_processed["cleaned_text"].astype(bool)):
            tfidf_features = self.tfidf_vectorizer.fit_transform(df_processed["cleaned_text"])
            tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=self.tfidf_vectorizer.get_feature_names_out())
        else:
            # Return an empty DataFrame with expected columns if no text to process
            tfidf_df = pd.DataFrame()

        return tfidf_df

class SentimentAnalyzer:
    def __init__(self):
        try:
            self.sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        except Exception as e:
            print(f"Could not load sentiment model, using a dummy analyzer: {e}")
            self.sentiment_pipeline = None

    def analyze_sentiment(self, texts):
        """Analyzes the sentiment of a list or Series of texts."""
        if self.sentiment_pipeline is None:
            print("Warning: Using dummy sentiment analysis. Install transformers and ensure internet connectivity for full functionality.")
            return pd.DataFrame({"sentiment_label": ["NEUTRAL"] * len(texts), "sentiment_score": [0.5] * len(texts)})

        valid_texts = [text for text in texts if isinstance(text, str) and text.strip() != ""]
        
        if not valid_texts:
            return pd.DataFrame({"sentiment_label": ["NEUTRAL"] * len(texts), "sentiment_score": [0.5] * len(texts)})

        results = self.sentiment_pipeline(valid_texts)

        # Map results back to original text list, filling in for invalid entries
        full_results = [None] * len(texts)
        valid_results_iter = iter(results)
        for i, text_original in enumerate(texts):
            if not isinstance(text_original, str) or text_original.strip() == "":
                full_results[i] = {"label": "NEUTRAL", "score": 0.5}
            else:
                full_results[i] = next(valid_results_iter)

        sentiments = [res["label"] for res in full_results]
        scores = [res["score"] if res["label"] == "POSITIVE" else (1 - res["score"] if res["label"] == "NEGATIVE" else 0.5) for res in full_results]
        
        return pd.DataFrame({"sentiment_label": sentiments, "sentiment_score": scores})

if __name__ == "__main__":
    # Example Usage:
    # 1. Create a dummy dataset resembling e-commerce product data and reviews
    data = {
        'product_id': [1, 2, 3, 4, 5],
        'product_name': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Webcam'],
        'price': [1200.00, 25.50, 75.00, 300.00, 50.00],
        'ratings': [4.5, 3.8, 4.2, 4.7, 3.5],
        'sales_volume': [150, 500, 200, 80, 120],
        'customer_review': [
            'This laptop is amazing, super fast and lightweight! Highly recommend.',
            'Mouse is okay for the price, but sometimes disconnects. Average.',
            'Great mechanical keyboard, typing experience is fantastic. Love it.',
            'Monitor has stunning display quality, perfect for gaming and work.',
            'Webcam quality is poor, very grainy. Disappointed with this purchase.'
        ]
    }
    df = pd.DataFrame(data)

    print("Original DataFrame:")
    print(df)
    print("\n" + "-" * 50 + "\n")

    # 2. Initialize preprocessor and sentiment analyzer
    preprocessor = DataPreprocessor()
    sentiment_analyzer = SentimentAnalyzer()

    # 3. Preprocess numerical data
    numerical_cols = ['price', 'ratings', 'sales_volume']
    df_processed_numerical = preprocessor.preprocess_numerical_data(df, numerical_cols)
    
    print("DataFrame after Numerical Preprocessing (scaled & imputed):")
    print(df_processed_numerical[numerical_cols].head())
    print("\n" + "-" * 50 + "\n")

    # 4. Preprocess text data (customer reviews)
    tfidf_features_df = preprocessor.preprocess_text_data(df, 'customer_review')
    
    print("TF-IDF Features from Customer Reviews (first 5 features):")
    if not tfidf_features_df.empty:
        print(tfidf_features_df.iloc[:, :5].head())
    else:
        print("No TF-IDF features generated.")
    print("\n" + "-" * 50 + "\n")

    # 5. Analyze sentiment of customer reviews
    sentiment_results_df = sentiment_analyzer.analyze_sentiment(df['customer_review'])

    print("Sentiment Analysis Results:")
    print(sentiment_results_df.head())
    print("\n" + "-" * 50 + "\n")

    # 6. Combine all processed features (for demonstration, showing how they'd be merged)
    # For a real ML model, you'd combine these appropriately
    # Ensure indices align for concatenation
    df_final = pd.concat([
        df_processed_numerical.drop(columns=['customer_review', 'cleaned_text'], errors='ignore'), 
        tfidf_features_df,
        sentiment_results_df
    ], axis=1)
    
    print("Combined Final DataFrame (numerical, TF-IDF, sentiment - partial view):")
    print(df_final.head())
    print("\nThis combined DataFrame would then be used for product recommendation or other ML tasks.")
