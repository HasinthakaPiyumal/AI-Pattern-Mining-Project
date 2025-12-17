import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import nltk

try:
    nltk.data.find('corpora/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except nltk.downloader.DownloadError:
    nltk.download('wordnet')

class TextPreprocessor:
    def __init__(self, use_stemming=False):
        self.stop_words = set(stopwords.words('english'))
        self.use_stemming = use_stemming
        if self.use_stemming:
            self.stemmer = PorterStemmer()
        else:
            self.lemmatizer = WordNetLemmatizer()

    def _tokenize(self, text):
        return word_tokenize(text.lower())

    def _remove_stopwords(self, tokens):
        return [word for word in tokens if word.isalpha() and word not in self.stop_words]

    def _lemmatize_or_stem(self, tokens):
        if self.use_stemming:
            return [self.stemmer.stem(word) for word in tokens]
        else:
            return [self.lemmatizer.lemmatize(word) for word in tokens]

    def preprocess_text(self, text):
        tokens = self._tokenize(text)
        filtered_tokens = self._remove_stopwords(tokens)
        processed_tokens = self._lemmatize_or_stem(filtered_tokens)
        return " ".join(processed_tokens)

class NumericalCategoricalPreprocessor:
    def __init__(self, numerical_features, categorical_features):
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.preprocessor = None

    def fit(self, X):
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])
        self.preprocessor.fit(X)

    def transform(self, X):
        if self.preprocessor is None:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")
        return self.preprocessor.transform(X)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

if __name__ == "__main__":
    # --- Simulate Data Ingestion ---
    review_data = {
        'review_id': [1, 2, 3, 4, 5],
        'review_text': [
            "This product is absolutely amazing! I love it so much. Highly recommend.",
            "It's okay, not great, not terrible. Just average.",
            "Terrible quality, broke after one week. Very disappointed.",
            "Good value for money. Works as expected.",
            "The item was damaged upon arrival, but customer service was helpful."
        ]
    }
    df_reviews = pd.DataFrame(review_data)

    product_data = {
        'product_id': [101, 102, 103, 104, 105],
        'price': [50.0, 25.0, 120.0, 30.0, 75.0],
        'rating': [4.5, 3.0, 1.5, 4.0, np.nan],
        'stock_quantity': [100, 500, 20, 300, 80],
        'category': ['Electronics', 'Home', 'Electronics', 'Books', 'Home'],
        'brand': ['BrandA', 'BrandB', 'BrandA', 'BrandC', 'BrandB']
    }
    df_products = pd.DataFrame(product_data)

    print("--- Original Review Data ---")
    print(df_reviews)
    print("\n--- Original Product Data ---")
    print(df_products)

    # --- Text Preprocessing --- 
    print("\n--- Text Preprocessing ---")
    text_preprocessor = TextPreprocessor(use_stemming=False)  # Using lemmatization
    df_reviews['processed_review_text'] = df_reviews['review_text'].apply(text_preprocessor.preprocess_text)
    
    print("Reviews after text preprocessing (lemmatization):")
    print(df_reviews[['review_text', 'processed_review_text']])

    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(max_features=100) 
    tfidf_matrix = tfidf_vectorizer.fit_transform(df_reviews['processed_review_text'])
    df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    print("\nTF-IDF Features for Reviews:")
    print(df_tfidf.head())

    # --- Numerical and Categorical Data Preprocessing ---
    print("\n--- Numerical and Categorical Data Preprocessing ---")
    numerical_features = ['price', 'rating', 'stock_quantity']
    categorical_features = ['category', 'brand']

    num_cat_preprocessor = NumericalCategoricalPreprocessor(numerical_features, categorical_features)
    processed_product_data = num_cat_preprocessor.fit_transform(df_products)
    
    # To reconstruct DataFrame with feature names (for demonstration)
    # This part can be more complex due to OneHotEncoder output, but for simple demo, get feature names.
    ohe_feature_names = num_cat_preprocessor.preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
    all_feature_names = numerical_features + list(ohe_feature_names)
    df_processed_products = pd.DataFrame(processed_product_data, columns=all_feature_names)

    print("\nProduct data after numerical and categorical preprocessing:")
    print(df_processed_products.head())
