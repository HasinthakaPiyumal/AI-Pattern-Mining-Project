import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure NLTK data is downloaded
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

class TextPreprocessor:
    """Handles preprocessing for textual data, including tokenization, stop-word removal, lemmatization, and TF-IDF vectorization."""
    def __init__(self, stop_words_lang='english', max_features=5000):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words(stop_words_lang))
        self.vectorizer = TfidfVectorizer(max_features=max_features)

    def preprocess_text(self, text):
        """Tokenizes, removes stop words, and lemmatizes a single text string."""
        if not isinstance(text, str):
            return ""
        tokens = nltk.word_tokenize(text.lower())
        filtered_tokens = [token for token in tokens if token.isalpha() and token not in self.stop_words]
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in filtered_tokens]
        return " ".join(lemmatized_tokens)

    def fit_transform_vectorizer(self, texts):
        """Applies preprocessing to a list of texts and then fits and transforms them using TF-IDF vectorization."""
        processed_texts = [self.preprocess_text(text) for text in texts]
        return self.vectorizer.fit_transform(processed_texts)

    def transform_vectorizer(self, texts):
        """Applies preprocessing to a list of texts and then transforms them using the fitted TF-IDF vectorizer."""
        processed_texts = [self.preprocess_text(text) for text in texts]
        return self.vectorizer.transform(processed_texts)


class NumericalPreprocessor:
    """Handles preprocessing for numerical data, including imputation and scaling."""
    def __init__(self, strategy='mean', scaler_type='standard'):
        self.imputer = SimpleImputer(strategy=strategy)
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("scaler_type must be 'standard' or 'minmax'")

    def fit_transform_numerical(self, data):
        """Fits the imputer and scaler to the numerical data and then transforms it."""
        imputed_data = self.imputer.fit_transform(data)
        scaled_data = self.scaler.fit_transform(imputed_data)
        return scaled_data

    def transform_numerical(self, data):
        """Transforms numerical data using the fitted imputer and scaler."""
        imputed_data = self.imputer.transform(data)
        scaled_data = self.scaler.transform(imputed_data)
        return scaled_data
