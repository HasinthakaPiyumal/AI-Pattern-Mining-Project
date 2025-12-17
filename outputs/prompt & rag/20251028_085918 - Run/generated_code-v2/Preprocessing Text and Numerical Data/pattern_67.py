import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

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
    def __init__(self):
        self.numerical_imputer = SimpleImputer(strategy='mean')
        self.numerical_scaler = StandardScaler()
        self.text_vectorizer = TfidfVectorizer(max_features=1000) # Limiting features for demonstration
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def _preprocess_numerical_data(self, df_numerical):
        imputed_data = self.numerical_imputer.fit_transform(df_numerical)
        scaled_data = self.numerical_scaler.fit_transform(imputed_data)
        return pd.DataFrame(scaled_data, columns=df_numerical.columns, index=df_numerical.index)

    def _preprocess_text_data(self, series_text):
        processed_texts = []
        for text in series_text:
            if pd.isna(text):
                processed_texts.append('')
                continue
            tokens = nltk.word_tokenize(text.lower())
            filtered_tokens = [token for token in tokens if token.isalnum() and token not in self.stop_words]
            lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in filtered_tokens]
            processed_texts.append(' '.join(lemmatized_tokens))
        
        text_features = self.text_vectorizer.fit_transform(processed_texts).toarray()
        return pd.DataFrame(text_features, index=series_text.index, columns=[f'tfidf_{i}' for i in range(text_features.shape[1])])

    def preprocess(self, df_numerical, df_text):
        processed_numerical = self._preprocess_numerical_data(df_numerical.copy())
        processed_text = self._preprocess_text_data(df_text.copy())
        
        # Concatenate preprocessed numerical and text features
        # Ensure indices align for proper concatenation
        return pd.concat([processed_numerical, processed_text], axis=1)

if __name__ == '__main__':
    # Example Usage:
    # Create synthetic patient data
    numerical_data = {
        'Age': [30, 45, 60, 25, np.nan, 55, 40, 70, 35, 50],
        'BloodPressure': [120, 135, 140, 110, 128, 150, 125, 160, 115, 130],
        'Cholesterol': [180, 220, 250, 170, 200, np.nan, 190, 280, 160, 210],
        'Glucose': [90, 105, 115, 85, 98, 120, 100, 130, 92, 110]
    }
    text_data = {
        'DoctorNotes': [
            "Patient presented with mild fever and cough. Prescribed antibiotics.",
            "Follow-up visit. Blood pressure stable. Advised lifestyle changes.",
            "Complains of severe headache for 3 days. Ordered MRI.",
            "Routine check-up. All vitals normal. No new concerns.",
            np.nan,
            "Diabetic patient. Glucose levels elevated. Adjusted medication.",
            "Post-surgery recovery going well. Pain management reviewed.",
            "Emergency admission due to chest pain. Suspected cardiac event.",
            "Annual physical. Patient is healthy.",
            "Recurring back pain. Referred to physical therapy."
        ]
    }

    df_numerical = pd.DataFrame(numerical_data)
    df_text = pd.DataFrame(text_data)

    print("Original Numerical Data:\n", df_numerical)
    print("\nOriginal Text Data:\n", df_text)

    preprocessor = PatientDataPreprocessor()
    preprocessed_data = preprocessor.preprocess(df_numerical, df_text['DoctorNotes'])

    print("\nPreprocessed Data (Numerical and Text Features Combined):\n", preprocessed_data.head())
    print("\nShape of Preprocessed Data:", preprocessed_data.shape)

    # Verify data types and absence of NaNs in numerical columns (should be handled by imputer/scaler)
    print("\nNaNs in preprocessed data:", preprocessed_data.isnull().sum().sum())