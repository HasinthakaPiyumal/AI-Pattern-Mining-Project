import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import random

# Download NLTK data (run once)
try:
    nltk.data.find("corpora/stopwords")
except nltk.downloader.DownloadError:
    nltk.download("stopwords")
try:
    nltk.data.find("tokenizers/punkt")
except nltk.downloader.DownloadError:
    nltk.download("punkt")
try:
    nltk.data.find("corpora/wordnet")
except nltk.downloader.DownloadError:
    nltk.download("wordnet")

class DataSimulator:
    def generate_numerical_data(self, num_users=100, num_products=50, num_interactions=500):
        user_ids = np.arange(1, num_users + 1)
        product_ids = np.arange(1, num_products + 1)
        categories = ["Electronics", "Clothing", "Books", "Home", "Sports"]

        data = {
            "user_id": np.random.choice(user_ids, num_interactions),
            "product_id": np.random.choice(product_ids, num_interactions),
            "views": np.random.randint(1, 100, num_interactions),
            "purchases": np.random.randint(0, 5, num_interactions),
            "rating": np.random.randint(1, 6, num_interactions),
            "category": np.random.choice(categories, num_interactions)
        }

        df = pd.DataFrame(data)
        df.loc[np.random.choice(df.index, int(num_interactions * 0.05)), "rating"] = np.nan
        df.loc[np.random.choice(df.index, int(num_interactions * 0.02)), "purchases"] = np.nan
        return df.drop_duplicates(subset=["user_id", "product_id"])

    def generate_review_data(self, num_products=50, num_reviews=300):
        product_ids = np.arange(1, num_products + 1)
        positive_reviews = [
            "Absolutely love this product! Highly recommend.",
            "Great quality and works perfectly. Very happy with my purchase.",
            "Exceeded my expectations, fantastic value for money.",
            "This is a game changer, so useful and well made.",
            "Couldn\'t be happier, five stars all the way!"
        ]
        negative_reviews = [
            "Very disappointed, product broke after a week.",
            "Poor quality, not worth the price.",
            "Didn\'t work as described, complete waste of money.",
            "Customer service was unhelpful, and the item arrived damaged.",
            "Wish I hadn\'t bought this, total regret."
        ]
        neutral_reviews = [
            "It\'s an okay product, does what it\'s supposed to.",
            "Nothing special, just average.",
            "Works fine, no major complaints but no praises either.",
            "Received it on time, seems to be alright so far.",
            "Standard item, nothing to write home about."
        ]

        all_reviews = positive_reviews + negative_reviews + neutral_reviews
        sentiments = ([1] * len(positive_reviews)) + ([0] * len(negative_reviews)) + ([2] * len(neutral_reviews))
        sentiment_map = {1: "positive", 0: "negative", 2: "neutral"}

        data = {
            "product_id": np.random.choice(product_ids, num_reviews),
            "review_text": np.random.choice(all_reviews, num_reviews),
            "sentiment_label": [random.choice(sentiments) for _ in range(num_reviews)]
        }

        df = pd.DataFrame(data)
        df["sentiment_text"] = df["sentiment_label"].map(sentiment_map)
        return df

class Preprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))
        self.numerical_transformer = None
        self.text_vectorizer = None

    def preprocess_numerical_data(self, df):
        numerical_features = ["views", "purchases", "rating"]
        categorical_features = ["category"]

        numerical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_pipeline, numerical_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
            ])

        self.numerical_transformer = preprocessor.fit(df)
        preprocessed_data = self.numerical_transformer.transform(df)

        new_columns = numerical_features + \
                      list(self.numerical_transformer.named_transformers_["cat"].get_feature_names_out(categorical_features))

        return pd.DataFrame(preprocessed_data, columns=new_columns, index=df.index)

    def preprocess_text_data(self, texts):
        processed_texts = []
        for text in texts:
            tokens = word_tokenize(text.lower())
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in self.stop_words]
            processed_texts.append(" ".join(tokens))

        if not self.text_vectorizer:
            self.text_vectorizer = TfidfVectorizer(max_features=1000)
            text_vectors = self.text_vectorizer.fit_transform(processed_texts)
        else:
            text_vectors = self.text_vectorizer.transform(processed_texts)

        return text_vectors, processed_texts

class SentimentAnalyzer:
    def __init__(self):
        self.model = None

    def train_model(self, X_tfidf, y_labels):
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X_tfidf, y_labels)

    def predict_sentiment(self, X_tfidf):
        if self.model:
            return self.model.predict(X_tfidf)
        else:
            raise RuntimeError("Sentiment model not trained.")

class RecommendationEngine:
    def __init__(self):
        self.nn_model = None
        self.product_features = None
        self.product_id_map = None

    def build_product_features(self, numerical_df, review_df, preprocessor, sentiment_analyzer):
        # Aggregate numerical data by product
        agg_numerical = numerical_df.groupby("product_id").agg({
            "views": "mean",
            "purchases": "sum",
            "rating": "mean"
        }).reset_index()

        # Preprocess review text for sentiment analysis
        review_texts = review_df["review_text"]
        tfidf_vectors, _ = preprocessor.preprocess_text_data(review_texts)

        # Predict sentiment
        review_df["predicted_sentiment"] = sentiment_analyzer.predict_sentiment(tfidf_vectors)

        # Aggregate sentiment by product (e.g., mean sentiment score, or counts of sentiment labels)
        sentiment_dummies = pd.get_dummies(review_df["predicted_sentiment"], prefix="sentiment")
        product_sentiment = pd.concat([review_df["product_id"], sentiment_dummies], axis=1)
        product_sentiment = product_sentiment.groupby("product_id").mean().reset_index()

        # Merge numerical and sentiment features
        combined_features = pd.merge(agg_numerical, product_sentiment, on="product_id", how="left")
        combined_features = combined_features.fillna(0) # Fill NaN from left join if a product has no reviews

        # Create a mapping for product_id to index
        self.product_id_map = {pid: i for i, pid in enumerate(combined_features["product_id"]) }
        self.reverse_product_id_map = {i: pid for i, pid in enumerate(combined_features["product_id"]) }

        # Store features for NN model
        self.product_features = combined_features.drop("product_id", axis=1).values
        self.nn_model = NearestNeighbors(n_neighbors=5, algorithm="brute", metric="cosine")
        self.nn_model.fit(self.product_features)

    def get_recommendations(self, product_id, n_recommendations=5):
        if product_id not in self.product_id_map:
            return []

        product_idx = self.product_id_map[product_id]
        distances, indices = self.nn_model.kneighbors(self.product_features[product_idx].reshape(1, -1), n_neighbors=n_recommendations + 1)

        recommended_product_indices = indices.flatten()[1:] # Exclude the product itself
        recommendations = [self.reverse_product_id_map[idx] for idx in recommended_product_indices]

        return recommendations

def main():
    simulator = DataSimulator()
    numerical_data = simulator.generate_numerical_data()
    review_data = simulator.generate_review_data()

    preprocessor = Preprocessor()

    # Preprocess Numerical Data
    preprocessed_numerical = preprocessor.preprocess_numerical_data(numerical_data)

    # Preprocess Text Data for Sentiment Analysis
    # We need to train the TF-IDF vectorizer on the full review texts first
    # And then transform them for training the sentiment model
    all_review_texts = review_data["review_text"]
    tfidf_vectors_for_sentiment, _ = preprocessor.preprocess_text_data(all_review_texts)

    # Train Sentiment Model
    sentiment_analyzer = SentimentAnalyzer()
    sentiment_analyzer.train_model(tfidf_vectors_for_sentiment, review_data["sentiment_label"])

    # Recommendation Engine
    recommender = RecommendationEngine()
    recommender.build_product_features(numerical_data, review_data, preprocessor, sentiment_analyzer)

    print("\n--- Sample Data ---")
    print("Numerical Data Head:\n", numerical_data.head())
    print("\nReview Data Head:\n", review_data.head())
    print("\nPreprocessed Numerical Features Head (example):\n", preprocessed_numerical.head())

    print("\n--- Recommendations ---")
    # Get recommendations for a sample product
    sample_product_id = random.choice(numerical_data["product_id"].unique())
    print(f"Recommendations for product {sample_product_id}:")
    recommendations = recommender.get_recommendations(sample_product_id, n_recommendations=5)
    if recommendations:
        print(f"Recommended products: {recommendations}")
    else:
        print("Could not find recommendations for this product or product not in system.")

    # Another sample
    sample_product_id_2 = random.choice(numerical_data["product_id"].unique())
    print(f"\nRecommendations for product {sample_product_id_2}:")
    recommendations_2 = recommender.get_recommendations(sample_product_id_2, n_recommendations=5)
    if recommendations_2:
        print(f"Recommended products: {recommendations_2}")
    else:
        print("Could not find recommendations for this product or product not in system.")

if __name__ == "__main__":
    main()