import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np

# Download necessary NLTK data
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except nltk.downloader.DownloadError:
    nltk.download('wordnet')

class DataPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.tfidf_vectorizer = TfidfVectorizer()
        self.min_max_scaler = MinMaxScaler()
        self.standard_scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')

    def preprocess_text(self, text_series):
        processed_texts = []
        for text in text_series:
            tokens = word_tokenize(text.lower())
            filtered_tokens = [word for word in tokens if word.isalnum() and word not in self.stop_words]
            lemmatized_tokens = [self.lemmatizer.lemmatize(word) for word in filtered_tokens]
            processed_texts.append(' '.join(lemmatized_tokens))
        return self.tfidf_vectorizer.fit_transform(processed_texts)

    def preprocess_numerical(self, df_numerical, columns_to_scale=None, strategy='minmax'):
        df_processed = df_numerical.copy()
        
        # Impute missing values
        df_processed[df_processed.columns] = self.imputer.fit_transform(df_processed)

        if columns_to_scale is None:
            columns_to_scale = df_processed.columns

        if strategy == 'minmax':
            df_processed[columns_to_scale] = self.min_max_scaler.fit_transform(df_processed[columns_to_scale])
        elif strategy == 'standard':
            df_processed[columns_to_scale] = self.standard_scaler.fit_transform(df_processed[columns_to_scale])
        else:
            raise ValueError("Scaling strategy must be 'minmax' or 'standard'")
        return df_processed

# --- Example Usage ---
if __name__ == "__main__":
    # Dummy Data Ingestion
    review_data = {
        'product_id': [1, 2, 3, 4, 5],
        'review_text': [
            "This product is amazing! I love it so much.",
            "Terrible quality, very disappointed with my purchase.",
            "It's okay, not great but not bad either.",
            "Highly recommended, worth every penny.",
            "Mediocre at best. Expected more."
        ]
    }
    sales_data = {
        'product_id': [1, 2, 3, 4, 5],
        'sales_amount': [1200.50, 300.75, 750.00, 1500.25, 400.00],
        'units_sold': [100, 30, 75, 120, np.nan], # Added a NaN for imputation demo
        'average_rating': [4.8, 2.1, 3.5, 4.9, 2.8]
    }

    df_reviews = pd.DataFrame(review_data)
    df_sales = pd.DataFrame(sales_data)

    preprocessor = DataPreprocessor()

    # Preprocess Text Data
    print("\n--- Original Review Text ---")
    print(df_reviews['review_text'])
    tfidf_features = preprocessor.preprocess_text(df_reviews['review_text'])
    print("\n--- TF-IDF Features (Shape) ---")
    print(tfidf_features.shape)
    print(tfidf_features.toarray()[:2]) # Print first 2 rows of TF-IDF vectors

    # Preprocess Numerical Data
    print("\n--- Original Sales Data ---")
    print(df_sales)
    
    # Example with Min-Max Scaling
    scaled_sales_minmax = preprocessor.preprocess_numerical(df_sales[['sales_amount', 'units_sold', 'average_rating']], strategy='minmax')
    print("\n--- Min-Max Scaled Sales Data ---")
    print(scaled_sales_minmax)

    # Example with Standard Scaling
    scaled_sales_standard = preprocessor.preprocess_numerical(df_sales[['sales_amount', 'units_sold', 'average_rating']], strategy='standard')
    print("\n--- Standard Scaled Sales Data ---")
    print(scaled_sales_standard)
