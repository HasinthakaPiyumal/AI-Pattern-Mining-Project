
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, mean_squared_error

# Ensure NLTK data is downloaded
try:
    stopwords.words("english")
except LookupError:
    import nltk
    nltk.download("stopwords")
    nltk.download("punkt")
    nltk.download("wordnet")
    nltk.download("omw-1.4")


def generate_dummy_data():
    """Generates synthetic e-commerce product review and price data."""
    # Simulate Product Review Data
    num_reviews = 1000
    reviews_data = {
        "product_id": np.random.randint(100, 200, num_reviews),
        "review_text": [
            "This product is amazing! I absolutely love it. Highly recommend.",
            "It's okay, not great, not terrible. Met expectations.",
            "Terrible quality, broke after a week. Very disappointed.",
            "Fantastic purchase, worth every penny.",
            "Could be better, a bit flimsy.",
            "Excellent value for money, very happy.",
            "Worst product ever, complete waste of money.",
            "Good product, works as described.",
            "Neutral feelings, nothing special.",
            "A must-buy, changed my life!",
            "Had high hopes, but it failed.",
            "Solid performance for the price point.",
            "Below average, wouldn't buy again.",
            "Highly satisfied with the performance.",
            "Just received it, looks good so far.",
            "Definitely recommend this item.",
            "The quality is poor, don't bother.",
            "Works perfectly, no complaints.",
            "Indifferent about this product.",
            "This is a game-changer, amazing!",
            "The color is vibrant and beautiful, really stands out.",
            "Battery life is terrible, dies too quickly.",
            "Easy to assemble and use, very user-friendly.",
            "Customer service was unhelpful when I had an issue.",
            "Great for daily use, very practical.",
            "Overpriced for what it offers, not good value.",
            "The packaging was damaged when it arrived.",
            "Smooth performance, no lag whatsoever.",
            "Too bulky to carry around easily.",
            "Perfect fit and comfortable.",
            "Durability is questionable, feels cheap.",
            "Fast shipping and well-packaged.",
            "Some features are missing that I expected.",
            "I'm pleasantly surprised by its quality.",
            "Wouldn't recommend for heavy-duty tasks.",
            "Sleek design and modern look.",
            "Software is buggy and crashes often.",
            "An essential tool for my work.",
            "It's an average product at best.",
            "Absolutely thrilled with this purchase!",
        ] * (num_reviews // 40 + 1)
    }
    reviews_data["review_text"] = reviews_data["review_text"][:num_reviews]
    reviews_df = pd.DataFrame(reviews_data)

    # Simulate Historical Sales and Price Data
    num_products = len(reviews_df["product_id"].unique())
    product_ids = reviews_df["product_id"].unique()
    price_data = []
    for product_id in product_ids:
        for _ in range(np.random.randint(5, 15)):  # Multiple historical entries per product
            price_data.append({
                "product_id": product_id,
                "date": pd.to_datetime("2022-01-01") + pd.to_timedelta(np.random.randint(0, 730), unit='D'),
                "price": np.random.uniform(10.0, 500.0),
                "units_sold": np.random.randint(10, 500),
                "category": np.random.choice(["Electronics", "Apparel", "Home & Kitchen", "Books", "Sports"]),
                "brand": np.random.choice(["BrandA", "BrandB", "BrandC", "BrandD"])
            })
    price_df = pd.DataFrame(price_data)

    # Add some missing values for demonstration
    reviews_df.loc[np.random.choice(reviews_df.index, 50, replace=False), "review_text"] = np.nan
    price_df.loc[np.random.choice(price_df.index, 30, replace=False), "price"] = np.nan
    price_df.loc[np.random.choice(price_df.index, 20, replace=False), "units_sold"] = np.nan
    price_df.loc[np.random.choice(price_df.index, 10, replace=False), "category"] = np.nan

    return reviews_df, price_df


def preprocess_text(text):
    """Applies a series of preprocessing steps to a given text."""
    if pd.isna(text):
        return ""
    text = text.lower()  # Lowercasing
    text = re.sub(r"[^a-zA-Z]", " ", text)  # Remove punctuation and numbers
    tokens = word_tokenize(text)  # Tokenization
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]  # Stop-word removal
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
    return " ".join(tokens)


def setup_numerical_preprocessing_pipeline(numerical_cols, categorical_cols):
    """
    Sets up a ColumnTransformer for numerical and categorical feature preprocessing.
    Includes imputation and scaling for numerical, and imputation and one-hot encoding for categorical.
    """
    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols)
        ],
        remainder="passthrough"  # Keep other columns (e.g., product_id, date) if not specified
    )
    return preprocessor


if __name__ == "__main__":
    print("--- E-commerce Product Review Sentiment Analysis and Price Prediction System ---")

    print("\n1. Generating Dummy Data...")
    reviews_df, price_df = generate_dummy_data()
    print("Reviews Data Head:")
    print(reviews_df.head())
    print("\nPrice Data Head:")
    print(price_df.head())

    print("\n2. Preprocessing Text Data for Sentiment Analysis...")
    reviews_df["processed_review_text"] = reviews_df["review_text"].apply(preprocess_text)

    # Basic sentiment labeling for demonstration (in a real scenario, this would be trained or come from labeled data)
    def simple_sentiment_labeler(text):
        if "amazing" in text or "love" in text or "text" in text or "fantastic" in text or "excellent" in text or "happy" in text or "recommend" in text or "thrilled" in text or "positive" in text:
            return "positive"
        elif "terrible" in text or "disappointed" in text or "worst" in text or "waste" in text or "poor" in text or "broken" in text or "negative" in text:
            return "negative"
        else:
            return "neutral"

    reviews_df["sentiment"] = reviews_df["processed_review_text"].apply(simple_sentiment_labeler)

    # Convert sentiment to numerical labels for modeling
    sentiment_mapping = {"negative": 0, "neutral": 1, "positive": 2}
    reviews_df["sentiment_label"] = reviews_df["sentiment"].map(sentiment_mapping)

    # TF-IDF Vectorization
    # Drop rows where processed_review_text is empty after preprocessing NaN original reviews
    reviews_df_clean = reviews_df.dropna(subset=["sentiment_label"])
    X_text = reviews_df_clean["processed_review_text"]
    y_sentiment = reviews_df_clean["sentiment_label"]

    if not X_text.empty:
        tfidf_vectorizer = TfidfVectorizer(max_features=1000)
        X_text_vectorized = tfidf_vectorizer.fit_transform(X_text)
        print(f"Shape of vectorized text data: {X_text_vectorized.shape}")
        print("Text Preprocessing and Vectorization Complete.")

        # --- Train Sentiment Analysis Model ---
        print("\n3. Training Sentiment Analysis Model...")
        X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
            X_text_vectorized, y_sentiment, test_size=0.2, random_state=42, stratify=y_sentiment
        )
        sentiment_model = LogisticRegression(max_iter=1000)
        sentiment_model.fit(X_train_s, y_train_s)
        y_pred_s = sentiment_model.predict(X_test_s)
        print(f"Sentiment Model Training Complete. Classification Report:\n{classification_report(y_test_s, y_pred_s)}")

        # Predict sentiment for all reviews for price prediction integration
        all_review_vectors = tfidf_vectorizer.transform(reviews_df["processed_review_text"].fillna(""))
        reviews_df["predicted_sentiment"] = sentiment_model.predict(all_review_vectors)
        # Convert numerical sentiment back to string for grouping, or directly use numerical for aggregation
        reviews_df["predicted_sentiment_str"] = reviews_df["predicted_sentiment"].map({0: "negative", 1: "neutral", 2: "positive"})

    else:
        print("No valid text data after preprocessing. Skipping sentiment analysis.")
        reviews_df["predicted_sentiment"] = 1  # Default to neutral if no text
        reviews_df["predicted_sentiment_str"] = "neutral"

    print("\n4. Preprocessing Numerical Data for Price Prediction...")
    # Merge sentiment data into price_df for price prediction features
    # Calculate average sentiment per product
    product_sentiment = reviews_df.groupby("product_id")["predicted_sentiment"].mean().reset_index()
    product_sentiment.rename(columns={"predicted_sentiment": "avg_sentiment_score"}, inplace=True)
    price_df_merged = pd.merge(price_df, product_sentiment, on="product_id", how="left")

    # Define numerical and categorical features
    # "price" is the target variable for price prediction
    numerical_features_for_preprocessing = ["units_sold", "avg_sentiment_score"]
    categorical_features_for_preprocessing = ["category", "brand"]

    # Target variable for price prediction (e.g., predict 'price' itself for simplicity)
    # Drop rows where 'price' is NaN before splitting target
    price_df_clean = price_df_merged.dropna(subset=["price"])

    X_numerical = price_df_clean.drop(columns=["price", "date"])  # 'date' is not used as feature directly for now
    y_price = price_df_clean["price"]

    # Filter features to only include those present in X_numerical
    actual_numerical_features = [f for f in numerical_features_for_preprocessing if f in X_numerical.columns]
    actual_categorical_features = [f for f in categorical_features_for_preprocessing if f in X_numerical.columns]

    if not X_numerical.empty and not y_price.empty and (actual_numerical_features or actual_categorical_features):
        numerical_preprocessor = setup_numerical_preprocessing_pipeline(actual_numerical_features, actual_categorical_features)

        # Split data for price prediction
        X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
            X_numerical, y_price, test_size=0.2, random_state=42
        )

        # Apply preprocessing
        X_train_p_processed = numerical_preprocessor.fit_transform(X_train_p)
        X_test_p_processed = numerical_preprocessor.transform(X_test_p)

        print(f"Shape of processed numerical data (training): {X_train_p_processed.shape}")
        print("Numerical Preprocessing Complete.")

        # --- Train Price Prediction Model ---
        print("\n5. Training Price Prediction Model...")
        price_model = RandomForestRegressor(n_estimators=100, random_state=42)
        price_model.fit(X_train_p_processed, y_train_p)
        y_pred_p = price_model.predict(X_test_p_processed)
        print(f"Price Prediction Model Training Complete. Mean Squared Error: {mean_squared_error(y_test_p, y_pred_p):.2f}")

        # Example prediction
        print("\n6. Demonstrating a Price Prediction:")
        # Take a sample from test set
        sample_index = X_test_p.index[0]
        sample_data = X_test_p.loc[[sample_index]]
        actual_price = y_test_p.loc[sample_index]

        processed_sample = numerical_preprocessor.transform(sample_data)
        predicted_price = price_model.predict(processed_sample)[0]

        print(f"Sample Product ID: {sample_data['product_id'].iloc[0]}")
        print(f"Actual Price: ${actual_price:.2f}")
        print(f"Predicted Price: ${predicted_price:.2f}")

    else:
        print("No valid numerical data after preprocessing. Skipping price prediction.")

    print("\n--- System Execution Complete ---")
