import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import nltk

# Ensure NLTK resources are downloaded
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

class EcomDataProcessor:
    def __init__(self):
        self.numerical_imputer = SimpleImputer(strategy='mean')
        self.scaler = MinMaxScaler()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.sentiment_model = None

    def _preprocess_text(self, text):
        if not isinstance(text, str): # Handle non-string inputs
            return ""
        text = text.lower() # Lowercasing
        text = re.sub(r'[^a-zA-Z\s]', '', text) # Remove punctuation and numbers
        tokens = word_tokenize(text) # Tokenization
        tokens = [word for word in tokens if word not in self.stop_words] # Stop word removal
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens] # Lemmatization
        return ' '.join(tokens)

    def preprocess_data(self, df):
        # 1. Numerical Data Preprocessing
        df['rating_processed'] = self.numerical_imputer.fit_transform(df[['rating']])
        df['rating_processed'] = self.scaler.fit_transform(df[['rating_processed']])

        # 2. Text Data Preprocessing
        df['cleaned_review'] = df['review_text'].apply(self._preprocess_text)
        self.tfidf_features = self.tfidf_vectorizer.fit_transform(df['cleaned_review'])

        return df

    def train_sentiment_model(self, df, sentiment_column='sentiment'):
        if self.tfidf_features is None:
            raise ValueError("TF-IDF features not generated. Run preprocess_data first.")

        X = self.tfidf_features
        y = df[sentiment_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.sentiment_model = LogisticRegression(max_iter=1000) # Increased max_iter for convergence
        self.sentiment_model.fit(X_train, y_train)

        y_pred = self.sentiment_model.predict(X_test)
        print("Sentiment Model Performance:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(classification_report(y_test, y_pred))

    def predict_sentiment(self, new_reviews):
        cleaned_new_reviews = [self._preprocess_text(review) for review in new_reviews]
        new_tfidf_features = self.tfidf_vectorizer.transform(cleaned_new_reviews)
        if self.sentiment_model:
            return self.sentiment_model.predict(new_tfidf_features)
        else:
            raise ValueError("Sentiment model not trained. Call train_sentiment_model first.")

    def recommend_products(self, product_id, df, top_n=5):
        if self.tfidf_features is None:
            raise ValueError("TF-IDF features not generated. Run preprocess_data first.")

        # Get features for the target product
        product_index = df[df['product_id'] == product_id].index
        if product_index.empty:
            return f"Product ID {product_id} not found."

        product_index = product_index[0]
        target_product_features = self.tfidf_features[product_index]

        # Calculate cosine similarity with all other products
        similarities = cosine_similarity(target_product_features, self.tfidf_features).flatten()

        # Exclude the target product itself and get top_n similar products
        similar_product_indices = similarities.argsort()[-top_n-1:][::-1]
        similar_product_indices = [idx for idx in similar_product_indices if idx != product_index]

        recommended_products = df.iloc[similar_product_indices[:top_n]][['product_id', 'product_name', 'rating']]
        return recommended_products

# --- Example Usage ---
if __name__ == "__main__":
    # 1. Simulate Data
    data = {
        'product_id': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3],
        'product_name': ['Laptop A', 'Mouse B', 'Keyboard C', 'Monitor D', 'Webcam E', 
                         'Laptop A', 'Mouse B', 'Keyboard C', 'Monitor D', 'Webcam E',
                         'Laptop A', 'Mouse B', 'Keyboard C'],
        'user_id': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113],
        'rating': [5, 4, 3, 5, np.nan, 4, 5, 2, 4, 3, 5, 4, 3],
        'review_text': [
            'Excellent laptop, very fast and sleek design.',
            'Good mouse for the price, ergonomic.',
            'Keyboard is okay, keys are a bit mushy.',
            'Amazing monitor, great colors and refresh rate.',
            'Webcam stopped working after a week, very disappointed.',
            'Love this laptop, perfect for my work and gaming needs.',
            'Mouse is responsive, but a little small for my hand.',
            'Keys are too loud, not suitable for office.',
            'Crisp display, highly recommend for professionals.',
            'Decent webcam for basic video calls, nothing special.',
            'Best laptop I ever had, super reliable.',
            'Comfortable mouse, long battery life.',
            'This keyboard feels cheap, not worth the money.'
        ],
        'sentiment': ['positive', 'positive', 'neutral', 'positive', 'negative',
                      'positive', 'neutral', 'negative', 'positive', 'neutral',
                      'positive', 'positive', 'negative'] # Assuming a pre-labeled sentiment for training
    }
    df = pd.DataFrame(data)

    # 2. Initialize and Preprocess Data
    processor = EcomDataProcessor()
    processed_df = processor.preprocess_data(df.copy())
    print("\n--- Processed DataFrame Sample ---")
    print(processed_df[['product_id', 'rating', 'rating_processed', 'cleaned_review', 'sentiment']].head())

    # 3. Train Sentiment Model
    print("\n--- Training Sentiment Model ---")
    processor.train_sentiment_model(processed_df, sentiment_column='sentiment')

    # 4. Predict Sentiment for New Reviews
    new_reviews = [
        "This product is fantastic! Love it.",
        "Absolutely terrible, a complete waste of money.",
        "It's an average product, does the job."
    ]
    predicted_sentiments = processor.predict_sentiment(new_reviews)
    print("\n--- Predicted Sentiments for New Reviews ---")
    for review, sentiment in zip(new_reviews, predicted_sentiments):
        print(f"Review: '{review}' -> Sentiment: {sentiment}")

    # 5. Get Product Recommendations
    print("\n--- Recommendations for Product ID 1 (Laptop A) ---")
    recommendations = processor.recommend_products(product_id=1, df=processed_df, top_n=3)
    print(recommendations)

    print("\n--- Recommendations for Product ID 5 (Webcam E) ---")
    recommendations = processor.recommend_products(product_id=5, df=processed_df, top_n=3)
    print(recommendations)