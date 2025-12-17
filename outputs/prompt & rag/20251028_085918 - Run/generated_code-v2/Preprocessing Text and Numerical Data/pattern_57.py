import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import hstack

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

class DataLoader:
    def load_data(self, num_samples=1000):
        np.random.seed(42)
        data = {
            "product_id": np.arange(1, num_samples + 1),
            "review_text": [
                "This product is amazing! I love it so much.",
                "Terrible quality, completely disappointed.",
                "It's okay, not great but not bad either.",
                "Highly recommend, excellent value for money.",
                "Broke after a week, very frustrating.",
                "Good product, does what it says on the tin.",
                "Not what I expected, a bit flimsy.",
                "Best purchase this year!",
                "Worst experience ever, avoid at all costs.",
                "Decent for the price, no complaints.",
            ] * (num_samples // 10),
            "rating": np.random.randint(1, 6, size=num_samples),
            "price": np.random.uniform(10, 500, size=num_samples),
            "category": np.random.choice(["Electronics", "Clothing", "Home", "Books", "Sports"], size=num_samples)
        }
        df = pd.DataFrame(data)

        # Introduce some missing values
        for col in ["price", "rating"]:
            missing_indices = np.random.choice(df.index, size=int(num_samples * 0.05), replace=False)
            df.loc[missing_indices, col] = np.nan
        
        df["review_text"] = df["review_text"].apply(lambda x: x if np.random.rand() > 0.1 else np.nan)

        return df

class NumericalPreprocessor:
    def __init__(self):
        self.imputer = SimpleImputer(strategy="mean")
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, X):
        self.imputer.fit(X)
        imputed_X = self.imputer.transform(X)
        self.scaler.fit(imputed_X)
        self.fitted = True
        return self

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("NumericalPreprocessor not fitted. Call fit() first.")
        imputed_X = self.imputer.transform(X)
        scaled_X = self.scaler.transform(imputed_X)
        return scaled_X
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def preprocess_text(self, text):
        if pd.isna(text):
            return ""
        text = text.lower()
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word.isalpha() and word not in self.stop_words]
        lemmatized_tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        return " ".join(lemmatized_tokens)

class TextVectorizer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.fitted = False

    def fit(self, text_data):
        self.vectorizer.fit(text_data)
        self.fitted = True
        return self

    def transform(self, text_data):
        if not self.fitted:
            raise RuntimeError("TextVectorizer not fitted. Call fit() first.")
        return self.vectorizer.transform(text_data)
    
    def fit_transform(self, text_data):
        self.fit(text_data)
        return self.transform(text_data)

class SentimentModelTrainer:
    def __init__(self, model=LogisticRegression(max_iter=1000, random_state=42)):
        self.model = model
        self.fitted = False

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.fitted = True
        print("Sentiment model trained successfully.")

    def predict(self, X):
        if not self.fitted:
            raise RuntimeError("SentimentModelTrainer not fitted. Call train() first.")
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        if not self.fitted:
            raise RuntimeError("SentimentModelTrainer not fitted. Call train() first.")
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        print(f"\nSentiment Model Accuracy: {accuracy:.4f}")
        print("Sentiment Model Classification Report:\n", report)
        return accuracy, report

class ProductRecommender:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.model = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
        self.product_data = None
        self.fitted = False

    def fit(self, features, product_data):
        self.model.fit(features)
        self.product_data = product_data.reset_index(drop=True)
        self.fitted = True
        print("Product recommender fitted successfully.")

    def recommend(self, product_id, features):
        if not self.fitted:
            raise RuntimeError("ProductRecommender not fitted. Call fit() first.")
        
        if product_id not in self.product_data["product_id"].values:
            print(f"Product ID {product_id} not found in the dataset.")
            return pd.DataFrame()

        product_idx = self.product_data[self.product_data["product_id"] == product_id].index[0]
        distances, indices = self.model.kneighbors(features[product_idx])
        
        # Exclude the product itself from recommendations
        recommended_indices = indices.flatten()[1:]
        recommended_distances = distances.flatten()[1:]

        recommendations = self.product_data.iloc[recommended_indices].copy()
        recommendations["similarity_score"] = 1 - recommended_distances
        return recommendations.sort_values(by="similarity_score", ascending=False)

if __name__ == "__main__":
    print("--- E-commerce Product Review Sentiment Analysis and Feature Recommendation System ---")

    # 1. Data Simulation/Loading
    print("\n1. Loading and Simulating Data...")
    data_loader = DataLoader()
    df = data_loader.load_data(num_samples=1000)
    print(f"Original DataFrame shape: {df.shape}")
    print("Sample of raw data:\n", df.head())
    print("Missing values before preprocessing:\n", df.isnull().sum())

    # Map ratings to sentiment (binary: positive/negative)
    df["sentiment"] = df["rating"].apply(lambda x: 1 if x >= 3 else 0)
    # Handle NaN ratings for sentiment generation by mapping them to a neutral or specific category first if needed for analysis,
    # but for model training, we'll impute them.

    # Separate features for preprocessing
    numerical_cols = ["price"]
    text_col = "review_text"
    target_col = "sentiment"
    
    numerical_data = df[numerical_cols]
    text_data = df[text_col].fillna("") # Fill NaN text with empty string for text processing

    # 2. Preprocessing
    print("\n2. Preprocessing Numerical and Text Data...")

    # Numerical Preprocessing
    numerical_preprocessor = NumericalPreprocessor()
    processed_numerical_features = numerical_preprocessor.fit_transform(numerical_data)
    print(f"Processed numerical features shape: {processed_numerical_features.shape}")

    # Text Preprocessing
    text_preprocessor = TextPreprocessor()
    preprocessed_texts = text_data.apply(text_preprocessor.preprocess_text)
    print("Sample of preprocessed text:\n", preprocessed_texts.head())

    # Text Vectorization
    text_vectorizer = TextVectorizer()
    text_features = text_vectorizer.fit_transform(preprocessed_texts)
    print(f"Text features (TF-IDF) shape: {text_features.shape}")

    # Combine all features for sentiment analysis and recommendation
    # Ensure 'rating' (which was used for sentiment) is imputed before combining if it's considered a numerical feature for recommendation
    # For sentiment training, we use the sentiment labels directly.
    
    # For sentiment analysis, we'll use numerical and text features
    combined_features_sentiment = hstack([processed_numerical_features, text_features])
    
    # Prepare target for sentiment analysis (impute NaNs in original rating for this if target was directly rating)
    # Here, 'sentiment' column handles NaN ratings by the original logic.
    df_cleaned_for_sentiment_target = df.dropna(subset=["rating"])
    y = df_cleaned_for_sentiment_target["sentiment"]
    
    # Align X and y after dropping NaNs for sentiment target
    original_indices_with_sentiment = df_cleaned_for_sentiment_target.index
    X_sentiment = combined_features_sentiment[original_indices_with_sentiment]

    print(f"Combined features for sentiment analysis shape: {X_sentiment.shape}")
    print(f"Sentiment target shape: {y.shape}")

    # 3. Sentiment Analysis
    print("\n3. Performing Sentiment Analysis...")
    X_train, X_test, y_train, y_test = train_test_split(X_sentiment, y, test_size=0.2, random_state=42, stratify=y)
    
    sentiment_trainer = SentimentModelTrainer()
    sentiment_trainer.train(X_train, y_train)
    sentiment_trainer.evaluate(X_test, y_test)

    # 4. Recommendation System
    print("\n4. Building and Using Recommendation System...")

    # For recommendation, we might want to include the predicted sentiment as a feature
    # Or just use the preprocessed numerical and text features.
    # Let's use the full combined features (numerical + TF-IDF) for recommendations.
    
    # We need to ensure that the features for the recommender are aligned with the product_id for indexing.
    # For simplicity, we use the `df` dataframe with all rows, but ensure features are imputed/vectorized.
    # Note: If some original rows had NaN reviews, the TF-IDF vector for those will be zeros. 
    # This is handled by `fillna("")` for text processing.
    
    combined_features_recommender = hstack([processed_numerical_features, text_features])

    product_recommender = ProductRecommender(n_neighbors=5)
    product_recommender.fit(combined_features_recommender, df)

    # Get recommendations for a sample product
    sample_product_id = df["product_id"].sample(1).iloc[0]
    print(f"\nRecommending products similar to Product ID: {sample_product_id}")
    recommendations = product_recommender.recommend(sample_product_id, combined_features_recommender)
    if not recommendations.empty:
        print("Top recommendations:\n", recommendations[["product_id", "review_text", "rating", "price", "similarity_score"]])
    else:
        print("No recommendations found.")

    print("\n--- End of Demonstration ---")
