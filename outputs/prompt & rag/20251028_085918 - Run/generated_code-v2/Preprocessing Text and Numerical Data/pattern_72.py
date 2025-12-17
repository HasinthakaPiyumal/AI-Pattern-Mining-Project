import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

import nltk
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

class DataIngestion:
    def load_data(self, review_filepath, numerical_filepath):
        reviews_df = pd.read_csv(review_filepath) if review_filepath else pd.DataFrame(columns=["product_id", "review", "sentiment"])
        numerical_df = pd.read_csv(numerical_filepath) if numerical_filepath else pd.DataFrame(columns=["product_id", "current_price", "historical_sales", "ratings", "stock_levels"])
        return reviews_df, numerical_df

class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000)

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r"[^a-z\s]", "", text)
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        return " ".join(tokens)

    def preprocess_reviews(self, reviews_df, text_column="review", fit_vectorizer=True):
        reviews_df["cleaned_review"] = reviews_df[text_column].apply(self.clean_text)
        if fit_vectorizer:
            text_features = self.tfidf_vectorizer.fit_transform(reviews_df["cleaned_review"])
        else:
            text_features = self.tfidf_vectorizer.transform(reviews_df["cleaned_review"])
        return text_features, self.tfidf_vectorizer

class NumericalDataPreprocessor:
    def __init__(self):
        self.imputer = SimpleImputer(strategy="mean")
        self.scaler = StandardScaler()

    def preprocess_numerical_data(self, numerical_df, feature_columns, fit_transformers=True):
        if fit_transformers:
            imputed_data = self.imputer.fit_transform(numerical_df[feature_columns])
            scaled_data = self.scaler.fit_transform(imputed_data)
        else:
            imputed_data = self.imputer.transform(numerical_df[feature_columns])
            scaled_data = self.scaler.transform(imputed_data)
        
        processed_numerical_df = pd.DataFrame(scaled_data, columns=feature_columns, index=numerical_df.index)
        return processed_numerical_df, self.imputer, self.scaler

class SentimentAnalysisModule:
    def __init__(self, model=LogisticRegression(max_iter=1000, random_state=42)):
        self.model = model

    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict_sentiment(self, X_test):
        return self.model.predict(X_test)

    def evaluate_model(self, X_test, y_test):
        y_pred = self.predict_sentiment(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))

class RecommendationLogicModule:
    def generate_recommendations(self, product_data):
        recommendations = []
        for index, row in product_data.iterrows():
            product_id = row["product_id"]
            sentiment = row["predicted_sentiment"]
            current_price = row["current_price"]
            historical_sales = row["historical_sales"]
            ratings = row["ratings"]
            stock_levels = row["stock_levels"]

            recommendation = "Maintain current price."
            action = "Monitor"
            reason = "-"

            if sentiment == "negative" and historical_sales < np.percentile(product_data["historical_sales"].dropna(), 30):
                recommendation = "Suggest price reduction."
                action = "Reduce Price"
                reason = "Predominantly negative sentiment and declining sales."
            elif sentiment == "positive" and ratings > np.percentile(product_data["ratings"].dropna(), 70) and stock_levels < np.percentile(product_data["stock_levels"].dropna(), 30):
                recommendation = "Consider price increase or focused marketing."
                action = "Increase Price / Marketing Push"
                reason = "Positive sentiment, high ratings, and low stock levels."
            elif sentiment == "neutral" and historical_sales > np.percentile(product_data["historical_sales"].dropna(), 70):
                recommendation = "Maintain current price with emphasis on marketing current value."
                action = "Maintain Price"
                reason = "Neutral sentiment but strong sales, focus on value proposition."
            elif sentiment == "positive" and historical_sales < np.percentile(product_data["historical_sales"].dropna(), 30):
                 recommendation = "Investigate why positive sentiment isn't translating to sales. Consider promotions."
                 action = "Investigate Sales / Promote"
                 reason = "Positive sentiment but low sales."
            elif sentiment == "negative" and historical_sales > np.percentile(product_data["historical_sales"].dropna(), 70):
                 recommendation = "Address product issues despite good sales. Risk of future decline."
                 action = "Address Issues"
                 reason = "Negative sentiment despite good sales. Potential future risk."
            

            recommendations.append({
                "product_id": product_id,
                "current_price": current_price,
                "sentiment": sentiment,
                "historical_sales": historical_sales,
                "ratings": ratings,
                "stock_levels": stock_levels,
                "recommendation": recommendation,
                "action": action,
                "reason": reason
            })
        return pd.DataFrame(recommendations)

class OutputReporter:
    def report_recommendations(self, recommendations_df):
        print("\n--- Price Recommendations ---")
        print(recommendations_df.to_string())

if __name__ == "__main__":
    # Example Data (replace with actual file paths for real use)
    # Create dummy dataframes for demonstration
    review_data = {
        "product_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        "review": [
            "This product is amazing, absolutely love it!",
            "Great quality for the price, highly recommend.",
            "It was okay, nothing special, a bit flimsy.",
            "I regret buying this, completely broken after a week.",
            "Decent product, good value for money.",
            "Works as expected, no complaints.",
            "Worst purchase ever, waste of money.",
            "Terrible design, very inconvenient to use.",
            "Fantastic product, exceeded my expectations!",
            "Pretty good, would buy again if needed."
        ],
        "sentiment": ["positive", "positive", "neutral", "negative", "positive", "neutral", "negative", "negative", "positive", "positive"]
    }
    numerical_data = {
        "product_id": [1, 2, 3, 4, 5],
        "current_price": [25.99, 12.50, 45.00, 5.99, 110.00],
        "historical_sales": [1200, 300, 800, 50, 2000],
        "ratings": [4.8, 2.1, 3.9, 1.5, 4.9],
        "stock_levels": [500, 150, 300, 20, 100]
    }

    reviews_df = pd.DataFrame(review_data)
    numerical_df = pd.DataFrame(numerical_data)

    # Save dummy data to CSV for loading simulation
    reviews_df.to_csv("product_reviews.csv", index=False)
    numerical_df.to_csv("product_numerical_data.csv", index=False)

    print("--- Starting Product Recommendation System ---")

    # 1. Data Ingestion
    data_ingestor = DataIngestion()
    reviews_df, numerical_df = data_ingestor.load_data("product_reviews.csv", "product_numerical_data.csv")
    print("\n--- Data Ingested ---")
    print("Reviews head:\n", reviews_df.head())
    print("Numerical data head:\n", numerical_df.head())

    # 2. Text Preprocessing
    text_preprocessor = TextPreprocessor()
    text_features, tfidf_vectorizer = text_preprocessor.preprocess_reviews(reviews_df, text_column="review", fit_vectorizer=True)
    print("\n--- Text Preprocessing Complete ---")
    print("TF-IDF features shape:", text_features.shape)

    # 3. Numerical Data Preprocessing
    numerical_feature_columns = ["current_price", "historical_sales", "ratings", "stock_levels"]
    numerical_preprocessor = NumericalDataPreprocessor()
    processed_numerical_df, imputer, scaler = numerical_preprocessor.preprocess_numerical_data(numerical_df, numerical_feature_columns, fit_transformers=True)
    
    # Combine processed numerical data with product_id for later merging
    processed_numerical_df["product_id"] = numerical_df["product_id"]
    print("\n--- Numerical Data Preprocessing Complete ---")
    print("Processed numerical data head:\n", processed_numerical_df.head())

    # 4. Sentiment Analysis
    sentiment_analyzer = SentimentAnalysisModule()

    # Prepare data for sentiment model training (using the sentiment column from reviews_df)
    X = text_features
    y = reviews_df["sentiment"]

    # Handle class imbalance if necessary (not implemented for simplicity here)
    # Map sentiment labels to numerical values for Logistic Regression if needed
    sentiment_mapping = {"positive": 2, "neutral": 1, "negative": 0}
    y_encoded = y.map(sentiment_mapping)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)

    print("\n--- Training Sentiment Analysis Model ---")
    sentiment_analyzer.train_model(X_train, y_train)
    print("Model Evaluation on Test Set:")
    sentiment_analyzer.evaluate_model(X_test, y_test)

    # Predict sentiment for ALL reviews (both training and testing part, for full product analysis)
    predicted_sentiment_encoded = sentiment_analyzer.predict_sentiment(text_features)
    # Map back to original sentiment labels
    reverse_sentiment_mapping = {v: k for k, v in sentiment_mapping.items()}
    reviews_df["predicted_sentiment"] = pd.Series(predicted_sentiment_encoded).map(reverse_sentiment_mapping)

    # 5. Combine data for Recommendation Logic
    # Aggregate sentiment per product_id. For simplicity, we'll take the mode or most frequent sentiment.
    # In a real-world scenario, you might want more sophisticated aggregation (e.g., weighted average of sentiment scores)
    product_sentiment_agg = reviews_df.groupby("product_id")["predicted_sentiment"].agg(lambda x: x.mode()[0] if not x.mode().empty else "neutral").reset_index()
    product_sentiment_agg.rename(columns={"predicted_sentiment": "aggregated_sentiment"}, inplace=True)

    # Merge aggregated sentiment with numerical data
    combined_product_data = pd.merge(numerical_df, product_sentiment_agg, on="product_id", how="left")
    combined_product_data.rename(columns={"aggregated_sentiment": "predicted_sentiment"}, inplace=True)

    print("\n--- Combined Product Data for Recommendations ---")
    print(combined_product_data.head())

    # 6. Recommendation Logic
    recommendation_engine = RecommendationLogicModule()
    recommendations_df = recommendation_engine.generate_recommendations(combined_product_data)

    # 7. Output and Reporting
    reporter = OutputReporter()
    reporter.report_recommendations(recommendations_df)

    print("\n--- Product Recommendation System Finished ---")