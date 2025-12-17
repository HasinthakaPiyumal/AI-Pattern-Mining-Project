
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Download NLTK data (if not already downloaded)
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except nltk.downloader.DownloadError:
    nltk.download('wordnet')

class PreprocessingPipeline:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        self.numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        self.categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preprocessor = None
        self.tfidf_vectorizer = TfidfVectorizer()

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)  # Remove punctuation and numbers
        tokens = text.split()
        tokens = [word for word in tokens if word not in self.stop_words]
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens] # Using lemmatization
        return ' '.join(tokens)

    def fit_numerical_and_categorical_preprocessors(self, df, numerical_features, categorical_features):
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', self.numerical_transformer, numerical_features),
                ('cat', self.categorical_transformer, categorical_features)
            ],
            remainder='passthrough'
        )
        self.preprocessor.fit(df)

    def transform_numerical_and_categorical(self, df):
        if self.preprocessor is None:
            raise ValueError("Call fit_numerical_and_categorical_preprocessors first.")
        return self.preprocessor.transform(df)
    
    def fit_tfidf(self, text_data):
        self.tfidf_vectorizer.fit(text_data)

    def transform_tfidf(self, text_data):
        return self.tfidf_vectorizer.transform(text_data)


def simulate_data():
    # Simulate customer review data
    review_data = {
        'product_id': [1, 1, 2, 2, 3, 3, 1],
        'user_id': [101, 102, 103, 101, 104, 102, 105],
        'rating': [5, 4, 2, 5, 3, None, 4],
        'review_text': [
            "This product is amazing! I love it so much. Highly recommend.",
            "Good product, but the delivery was a bit slow.",
            "Very disappointed with this purchase. Poor quality.",
            "Excellent! Works perfectly, great value for money.",
            "It's okay, nothing special. Could be better.",
            "Worst product ever. Don't buy.",
            "Absolutely fantastic! A must-have for everyone."
        ]
    }
    df_reviews = pd.DataFrame(review_data)

    # Simulate product metadata
    product_data = {
        'product_id': [1, 2, 3, 4],
        'category': ['Electronics', 'Home & Kitchen', 'Books', 'Electronics'],
        'price': [120.50, 45.00, 15.99, 299.99],
        'description': [
            "High-performance wireless headphones with noise cancellation.",
            "Durable non-stick pan set for everyday cooking.",
            "Bestselling novel by a renowned author.",
            "Smartwatch with advanced fitness tracking and long battery life."
        ]
    }
    df_products = pd.DataFrame(product_data)

    return df_reviews, df_products

if __name__ == "__main__":
    df_reviews, df_products = simulate_data()

    print("--- Original Review Data ---")
    print(df_reviews.head())
    print("\n--- Original Product Data ---")
    print(df_products.head())

    # Initialize preprocessing pipeline
    pipeline = PreprocessingPipeline()

    # --- Text Preprocessing (Reviews) ---
    df_reviews['cleaned_review_text'] = df_reviews['review_text'].apply(pipeline.preprocess_text)
    print("\n--- Reviews after Text Preprocessing ---")
    print(df_reviews[['review_text', 'cleaned_review_text']].head())

    # --- Numerical and Categorical Preprocessing (Products) ---
    # Merge dataframes for combined preprocessing if necessary, or process separately
    # For demonstration, let's process numerical/categorical from product data
    numerical_features = ['price']
    categorical_features = ['category']

    # Fit and transform numerical and categorical features from product data
    pipeline.fit_numerical_and_categorical_preprocessors(df_products, numerical_features, categorical_features)
    transformed_product_features = pipeline.transform_numerical_and_categorical(df_products)

    print("\n--- Transformed Numerical and Categorical Product Features (Shape) ---")
    print(transformed_product_features.shape)
    # You would typically convert this sparse matrix or array back to a DataFrame with meaningful column names
    # For demonstration, we just show the shape.

    # --- TF-IDF Vectorization (Cleaned Reviews) ---
    pipeline.fit_tfidf(df_reviews['cleaned_review_text'])
    tfidf_matrix = pipeline.transform_tfidf(df_reviews['cleaned_review_text'])
    print("\n--- TF-IDF Vectorization of Cleaned Reviews (Shape) ---")
    print(tfidf_matrix.shape)
    print("Sample TF-IDF vector for the first review:\n", tfidf_matrix[0].toarray())

    # --- Example of combining features (conceptual) ---
    # In a real scenario, you'd combine tfidf_matrix with transformed_product_features
    # and potentially other numerical features from df_reviews (like 'rating' after imputation/scaling)
    # to create a final feature set for a machine learning model.
    # For instance, if you were building a sentiment model, you'd combine tfidf_matrix with numerical features
    # from df_reviews. If building a recommendation model, you'd combine product features and user behavior.
    print("\nPreprocessing and feature engineering complete. The output 'tfidf_matrix' and 'transformed_product_features' can now be used for downstream ML models.")
