import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re

# Download necessary NLTK data (run once)
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

class PatientDataPreprocessor:
    def __init__(self, numerical_features, categorical_features, text_features):
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.text_features = text_features

        self.text_transformer = Pipeline([
            ('tfidf', TfidfVectorizer(preprocessor=self._preprocess_text, stop_words='english', min_df=2))
        ])

        self.numerical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        self.categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', self.numerical_transformer, self.numerical_features),
                ('cat', self.categorical_transformer, self.categorical_features),
                ('txt', self.text_transformer, self.text_features)
            ], 
            remainder='drop'
        )

    def _preprocess_text(self, text):
        if not isinstance(text, str): # Handle non-string inputs gracefully
            return ""
        # Lowercasing
        text = text.lower()
        # Remove punctuation and numbers
        text = re.sub(r'[^a-z\s]', '', text)
        # Tokenization
        tokens = nltk.word_tokenize(text)
        # Stop word removal and stemming/lemmatization
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        processed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        return ' '.join(processed_tokens)

    def fit(self, X, y=None):
        self.preprocessor.fit(X)
        return self

    def transform(self, X):
        return self.preprocessor.transform(X)

    def fit_transform(self, X, y=None):
        return self.preprocessor.fit_transform(X, y)


if __name__ == '__main__':
    # --- Simulate Data Ingestion ---
    data = {
        'patient_id': [1, 2, 3, 4, 5, 6],
        'age': [45, 62, 30, np.nan, 70, 55],
        'weight': [70.5, 85.1, 60.0, 75.3, np.nan, 80.2],
        'blood_pressure_systolic': [120, 140, 110, 130, 150, 125],
        'gender': ['Male', 'Female', 'Female', 'Male', 'Female', 'Male'],
        'disease_history': ['Hypertension', 'Diabetes', 'None', 'Asthma', 'Hypertension', 'Diabetes'],
        'doctor_notes': [
            'Patient reports mild chest pain and fatigue. Regular checkup.',
            'Diabetic patient with stable blood sugar. Follow-up in 3 months.',
            'Healthy individual with no significant concerns. Annual physical.',
            'History of asthma, occasional shortness of breath. Prescribed inhaler.',
            'Elderly patient with high blood pressure. Medication adjusted.',
            'New onset of joint pain, suspected rheumatoid arthritis. Referred to specialist.'
        ]
    }
    df = pd.DataFrame(data)

    print("\nOriginal DataFrame head:")
    print(df.head())
    print("\nOriginal DataFrame info:")
    df.info()

    # Define features
    numerical_features = ['age', 'weight', 'blood_pressure_systolic']
    categorical_features = ['gender', 'disease_history']
    text_features = ['doctor_notes']

    # Initialize and use the preprocessor
    preprocessor = PatientDataPreprocessor(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        text_features=text_features
    )

    # Fit and transform the data
    preprocessed_data = preprocessor.fit_transform(df)

    print("\nShape of preprocessed data:", preprocessed_data.shape)
    print("\nType of preprocessed data:", type(preprocessed_data))

    # To see the output in a more readable format, especially for demonstration
    # This part can be complex due to OneHotEncoder and TfidfVectorizer outputting many columns
    # We'll just show a snippet or shape here.
    # For real applications, you'd likely feed this directly into an ML model.

    # Example of how to get feature names after transformation (complex for ColumnTransformer)
    # This part is just for explanation, not strictly part of the preprocessing pipeline output itself.
    # You would typically feed `preprocessed_data` directly to an ML model.

    # If you want to convert back to DataFrame for inspection (requires more effort to get column names)
    # num_cols = preprocessor.preprocessor.named_transformers_['num'].named_steps['scaler'].get_feature_names_out(numerical_features)
    # cat_cols = preprocessor.preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
    # text_cols = preprocessor.preprocessor.named_transformers_['txt'].named_steps['tfidf'].get_feature_names_out()
    # all_cols = list(num_cols) + list(cat_cols) + list(text_cols)
    # preprocessed_df = pd.DataFrame(preprocessed_data.toarray() if hasattr(preprocessed_data, 'toarray') else preprocessed_data, columns=all_cols)
    # print("\nPreprocessed DataFrame head (partial view, complex with many columns):\n", preprocessed_df.head())
