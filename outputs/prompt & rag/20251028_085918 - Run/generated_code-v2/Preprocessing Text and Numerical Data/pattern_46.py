import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Download necessary NLTK data
try:
    nltk.data.find("corpora/stopwords")
except nltk.downloader.DownloadError:
    nltk.download("stopwords")
try:
    nltk.data.find("corpora/wordnet")
except nltk.downloader.DownloadError:
    nltk.download("wordnet")
try:
    nltk.data.find("tokenizers/punkt")
except nltk.downloader.DownloadError:
    nltk.download("punkt")

# 1. Data Simulation Module
def simulate_data(num_samples=100):
    np.random.seed(42)
    data = {
        "customer_id": np.arange(1, num_samples + 1),
        "product_id": np.random.randint(1001, 1050, num_samples),
        "review_text": [
            "This product is amazing! I love it so much. Highly recommended.",
            "It's okay, but the quality could be better. A bit flimsy.",
            "Terrible experience. The item broke after one use. Waste of money.",
            "Good value for the price. Works as expected.",
            "I had some issues with delivery, but the product itself is fine.",
            "Absolutely fantastic! Exceeded my expectations.",
            "Not what I expected. Very disappointed with the features.",
            "Decent product, but there are better alternatives out there.",
            "The best purchase I've made this year. Super happy!",
            "Mediocre at best. Would not buy again."
        ] * (num_samples // 10) + [np.nan] * (num_samples % 10),
        "rating": np.random.randint(1, 6, num_samples),
        "price": np.random.uniform(10.0, 500.0, num_samples),
        "purchase_frequency": np.random.randint(1, 10, num_samples),
        "category": np.random.choice(["Electronics", "Clothing", "Home", "Books"], num_samples)
    }
    df = pd.DataFrame(data)
    # Introduce some missing values for numerical features
    for col in ["rating", "price", "purchase_frequency"]:
        missing_indices = np.random.choice(df.index, size=int(num_samples * 0.05), replace=False)
        df.loc[missing_indices, col] = np.nan
    return df

# 2. Text Preprocessing Module
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if pd.isna(text):
        return ""
    tokens = nltk.word_tokenize(text.lower())
    filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(filtered_tokens)

# 3. Numerical Data Preprocessing Module
def create_numerical_pipeline():
    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    return numerical_transformer

def create_categorical_pipeline():
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    return categorical_transformer


if __name__ == "__main__":
    print("\n--- Original Data --- ")
    raw_df = simulate_data(num_samples=20)
    print(raw_df.head())
    print("\nOriginal Data Info:")
    raw_df.info()
    print("\nOriginal Data Missing Values:\n", raw_df.isnull().sum())

    # Apply Text Preprocessing
    print("\n--- Applying Text Preprocessing (Tokenization, Stop-word removal, Lemmatization) ---")
    raw_df["cleaned_review_text"] = raw_df["review_text"].apply(preprocess_text)
    print("Example of cleaned text (first 5 reviews):\n", raw_df["cleaned_review_text"].head().tolist())

    # TF-IDF Vectorization
    print("\n--- Applying TF-IDF Vectorization to Text Data ---")
    tfidf_vectorizer = TfidfVectorizer(max_features=100)
    tfidf_features = tfidf_vectorizer.fit_transform(raw_df["cleaned_review_text"])
    tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    print("Shape of TF-IDF features:", tfidf_df.shape)
    print("First 5 TF-IDF features for the first review:\n", tfidf_df.head(1).values)

    # Numerical and Categorical Feature Preprocessing
    numerical_features = ["rating", "price", "purchase_frequency"]
    categorical_features = ["category"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", create_numerical_pipeline(), numerical_features),
            ("cat", create_categorical_pipeline(), categorical_features)
        ])

    print("\n--- Applying Numerical (Imputation, Scaling) and Categorical (Imputation, One-Hot Encoding) Preprocessing ---")
    transformed_data = preprocessor.fit_transform(raw_df[numerical_features + categorical_features])

    # Get feature names for preprocessed numerical and categorical data
    # This part can be tricky with ColumnTransformer, a more robust way might be to fit and transform separately for explanation
    # For simplicity, we'll just show the combined array and its shape
    
    print("Shape of preprocessed numerical and categorical features:", transformed_data.shape)
    print("First 5 rows of preprocessed numerical and categorical features:\n", transformed_data[:5])

    # 4. Integration Module
    # Convert transformed_data to DataFrame
    # A more robust way to get feature names after ColumnTransformer is needed for proper DataFrame conversion.
    # For this example, we'll combine the TF-IDF features with the preprocessed numerical/categorical array.
    # This assumes the indices align after initial data creation.
    
    # For a simple integration, we'll convert transformed_data to a DataFrame with generic column names
    # and concatenate with tfidf_df.
    numerical_cat_df = pd.DataFrame(transformed_data, index=raw_df.index)
    
    # Ensure both dataframes have the same number of rows before concatenating
    if len(tfidf_df) != len(numerical_cat_df):
        # Handle potential length mismatch if some rows were dropped or reordered, though not expected here
        min_len = min(len(tfidf_df), len(numerical_cat_df))
        tfidf_df = tfidf_df.iloc[:min_len]
        numerical_cat_df = numerical_cat_df.iloc[:min_len]

    final_preprocessed_df = pd.concat([tfidf_df, numerical_cat_df], axis=1)
    
    print("\n--- Integrated Preprocessed Data (Text + Numerical/Categorical) ---")
    print("Shape of integrated preprocessed data:", final_preprocessed_df.shape)
    print(final_preprocessed_df.head())

    print("\n--- Explanation of Transformations ---")
    print("1. Text Data: Customer review text was tokenized, stop words were removed, and words were lemmatized. "
          "Then, TF-IDF vectorization was applied to convert the processed text into numerical features, "
          "representing the importance of words in reviews.")
    print("2. Numerical Data: Missing values in 'rating', 'price', and 'purchase_frequency' were imputed with the mean. "
          "These features were then scaled using StandardScaler to ensure they have a mean of 0 and standard deviation of 1.")
    print("3. Categorical Data: Missing values in 'category' were imputed with the most frequent category. "
          "'Category' was then One-Hot Encoded, converting categorical values into a binary numerical format, "
          "avoiding ordinal assumptions.")
    print("4. Integration: The TF-IDF features from text and the scaled/encoded numerical/categorical features "
          "were concatenated to form a unified, clean dataset ready for machine learning models.")
