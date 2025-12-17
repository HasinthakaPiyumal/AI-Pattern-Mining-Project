import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import nltk

# Download necessary NLTK data (run once)
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

class ECommerceDataPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000) # Limiting features for demonstration
        self.numerical_imputer = SimpleImputer(strategy='mean')
        self.numerical_scaler = MinMaxScaler()

    def _clean_text(self, text):
        text = str(text).lower() # Ensure text is string and lowercase
        text = re.sub(r'[^a-z\s]', '', text) # Remove punctuation and numbers
        tokens = word_tokenize(text)
        filtered_tokens = [self.stemmer.stem(word) for word in tokens if word not in self.stop_words]
        return " ".join(filtered_tokens)

    def preprocess_text_data(self, reviews: pd.Series) -> np.ndarray:
        cleaned_reviews = reviews.apply(self._clean_text)
        tfidf_features = self.tfidf_vectorizer.fit_transform(cleaned_reviews)
        return tfidf_features.toarray()

    def preprocess_numerical_data(self, df: pd.DataFrame, numerical_cols: list) -> pd.DataFrame:
        # Impute missing values
        df_imputed = pd.DataFrame(self.numerical_imputer.fit_transform(df[numerical_cols]),
                                  columns=numerical_cols,
                                  index=df.index)
        
        # Scale numerical values
        df_scaled = pd.DataFrame(self.numerical_scaler.fit_transform(df_imputed),
                                 columns=numerical_cols,
                                 index=df.index)
        return df_scaled

    def preprocess_all(self, df: pd.DataFrame, review_col: str, numerical_cols: list) -> pd.DataFrame:
        # Preprocess text data
        tfidf_features_array = self.preprocess_text_data(df[review_col])
        tfidf_features_df = pd.DataFrame(tfidf_features_array,
                                         columns=[f'tfidf_{i}' for i in range(tfidf_features_array.shape[1])],
                                         index=df.index)
        
        # Preprocess numerical data
        processed_numerical_df = self.preprocess_numerical_data(df.copy(), numerical_cols)
        
        # Combine all features
        combined_df = pd.concat([tfidf_features_df, processed_numerical_df], axis=1)
        return combined_df

if __name__ == "__main__":
    # Simulate E-commerce Data
    data = {
        'product_id': [1, 2, 3, 4, 5, 6, 7],
        'review_text': [
            "This product is absolutely amazing! Highly recommend it.",
            "It's okay, nothing special. The quality could be better.",
            "Terrible experience, broke after a week. Very disappointed.",
            "Good value for money, very happy with my purchase.       ",
            "Mediocre performance, expected more. Shipping was fast though.",
            np.nan, # Missing review
            "Fantastic! Love it, perfect for my needs. "
        ],
        'rating': [5, 3, 1, 4, 2, 4, 5],
        'price': [25.99, 12.50, 75.00, 30.00, 50.00, 15.00, 40.00],
        'sales_volume': [120, 50, 10, 80, 30, np.nan, 95] # Missing sales volume
    }
    df = pd.DataFrame(data)

    print("Original DataFrame:")
    print(df)
    print("\n" + "-"*50 + "\n")

    preprocessor = ECommerceDataPreprocessor()
    review_column = 'review_text'
    numerical_columns = ['rating', 'price', 'sales_volume']

    processed_df = preprocessor.preprocess_all(df, review_column, numerical_columns)

    print("Processed DataFrame (first 5 rows and columns):")
    print(processed_df.head())
    print("\nShape of processed DataFrame:", processed_df.shape)
    print("\n" + "-"*50 + "\n")

    # Example: Check if numerical columns are scaled between 0 and 1
    print("Min/Max of processed numerical columns:")
    for col in numerical_columns:
        if f'tfidf_{preprocessor.tfidf_vectorizer.max_features - 1}' in processed_df.columns:
            # Adjust column name based on actual structure, assuming numericals are at the end
            # This is a bit fragile; better to access by actual names or slice
            # For demonstration, let's just show the last few columns
            pass
    print(processed_df[numerical_columns].describe().loc[['min', 'max']])

    # Example: Check for NaNs (should be none after imputation)
    print("\nNaNs in processed DataFrame:")
    print(processed_df.isnull().sum().sum()) # Total NaNs
