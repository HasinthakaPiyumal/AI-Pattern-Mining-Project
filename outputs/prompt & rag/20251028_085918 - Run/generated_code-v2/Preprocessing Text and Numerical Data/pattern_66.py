import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.numerical_transformer = None
        self.categorical_transformer = None
        self.tfidf_vectorizer = None

    def _clean_text(self, text):
        if not isinstance(text, str): # Handle non-string values like NaN
            return ""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove URLs
        text = re.sub(r'<.*?>', '', text) # Remove HTML tags
        text = re.sub(r'[^a-z\s]', '', text) # Remove special characters and numbers
        return text

    def _tokenize_text(self, text):
        return nltk.word_tokenize(text)

    def _remove_stopwords(self, tokens):
        return [word for word in tokens if word not in self.stop_words]

    def _lemmatize_text(self, tokens):
        return [self.lemmatizer.lemmatize(word) for word in tokens]

    def preprocess_text(self, text_series):
        cleaned_text = text_series.apply(self._clean_text)
        tokenized_text = cleaned_text.apply(self._tokenize_text)
        filtered_text = tokenized_text.apply(self._remove_stopwords)
        lemmatized_text = filtered_text.apply(self._lemmatize_text)
        return lemmatized_text.apply(lambda x: ' '.join(x))

    def fit(self, df, numerical_cols, categorical_cols, text_col):
        # Fit numerical and categorical transformers
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_pipeline, numerical_cols),
                ('cat', categorical_pipeline, categorical_cols)
            ],
            remainder='passthrough' # Keep other columns if any
        )
        self.numerical_categorical_preprocessor = preprocessor.fit(df)

        # Fit TF-IDF Vectorizer
        processed_text = self.preprocess_text(df[text_col])
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000) # Limit features for demonstration
        self.tfidf_vectorizer.fit(processed_text)

    def transform(self, df, numerical_cols, categorical_cols, text_col):
        # Transform numerical and categorical data
        transformed_numerical_categorical = self.numerical_categorical_preprocessor.transform(df)
        
        # Get feature names for numerical and categorical data
        num_features = numerical_cols
        cat_features = self.numerical_categorical_preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_cols)
        all_nc_features = list(num_features) + list(cat_features)

        transformed_df_nc = pd.DataFrame(transformed_numerical_categorical, columns=all_nc_features, index=df.index)

        # Transform text data
        processed_text = self.preprocess_text(df[text_col])
        tfidf_features = self.tfidf_vectorizer.transform(processed_text)
        tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=self.tfidf_vectorizer.get_feature_names_out(), index=df.index)

        # Combine all features
        combined_features_df = pd.DataFrame(transformed_df_nc).join(tfidf_df)
        return combined_features_df


if __name__ == "__main__":
    # 1. Simulate Data Ingestion
    data = {
        'product_id': [1, 2, 3, 4, 5, 6],
        'price': [10.99, 25.50, np.nan, 5.00, 12.75, 8.20],
        'rating': [4.5, 3.8, 5.0, 2.5, 4.2, 3.0],
        'category': ['Electronics', 'Books', 'Electronics', 'Home', 'Books', 'Home'],
        'brand': ['A', 'B', 'C', 'A', 'B', 'D'],
        'review_count': [150, 80, 200, 30, 120, np.nan],
        'customer_review': [
            "This product is amazing! Fast shipping and great quality. Highly recommend. https://example.com/product1",
            "Disappointed with the battery life. It's okay, but expected more. <p>See details.</p>",
            "Absolutely love it! Best purchase this year. The screen is vibrant.",
            "Terrible product. Broke after a week. Don't buy. Returns are difficult!",
            "Good book, a bit slow at times but informative. Customer service was helpful.",
            np.nan # Missing review
        ]
    }
    df = pd.DataFrame(data)

    print("Original DataFrame:\n", df)
    print("\n--- Starting Preprocessing ---\n")

    numerical_features = ['price', 'rating', 'review_count']
    categorical_features = ['category', 'brand']
    text_feature = 'customer_review'

    preprocessor = DataPreprocessor()

    # Fit and transform the data
    preprocessor.fit(df, numerical_features, categorical_features, text_feature)
    processed_data = preprocessor.transform(df, numerical_features, categorical_features, text_feature)

    print("\nProcessed Numerical and Text Features (first 5 rows and columns):\n")
    print(processed_data.head())
    print(f"\nShape of processed data: {processed_data.shape}")

    print("\n--- Example of transformed text from customer_review ---")
    # To show the intermediate processed text:
    processed_raw_text = preprocessor.preprocess_text(df[text_feature])
    for i, original in enumerate(df[text_feature].fillna('')): # Fill NaN for display
        if i < 3: # Show first 3 examples
            print(f"Original: {original[:70]}...")
            print(f"Processed: {processed_raw_text.iloc[i][:70]}...\n")

    print("\nPreprocessing complete. The `processed_data` DataFrame now contains numerical, one-hot encoded categorical, and TF-IDF vectorized text features, ready for an ML model.")

    # Optional: Demonstrate Sentence Transformer (requires installing sentence-transformers)
    # from sentence_transformers import SentenceTransformer
    # try:
    #     model = SentenceTransformer('all-MiniLM-L6-v2')
    #     sentence_embeddings = model.encode(processed_raw_text.tolist(), show_progress_bar=False)
    #     print(f"\nSentence embeddings generated with shape: {sentence_embeddings.shape}")
    #     print(f"First embedding (first 5 values): {sentence_embeddings[0][:5]}\n")
    # except Exception as e:
    #     print(f"\nCould not load SentenceTransformer or encode: {e}. Please ensure the library is installed.")
