import pandas as pd
import numpy as np
import re
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
try:
    nltk.data.find("corpora/stopwords")
except nltk.downloader.DownloadError:
    nltk.download("stopwords")
try:
    nltk.data.find("tokenizers/punkt")
except nltk.downloader.DownloadError:
    nltk.download("punkt")
try:
    nltk.data.find("corpora/wordnet")
except nltk.downloader.DownloadError:
    nltk.download("wordnet")

class PatientDataProcessor:
    def __init__(self, numerical_imputation_strategy="mean", numerical_scaler="standard", max_features_text=5000):
        self.numerical_imputer = SimpleImputer(strategy=numerical_imputation_strategy)
        if numerical_scaler == "standard":
            self.numerical_scaler = StandardScaler()
        elif numerical_scaler == "minmax":
            self.numerical_scaler = MinMaxScaler()
        else:
            self.numerical_scaler = None
        self.tfidf_vectorizer = TfidfVectorizer(max_features=max_features_text)
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = set(stopwords.words("english"))

    def _clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r"[^a-z\s]", "", text) # Remove punctuation, numbers, special characters
        return text

    def _tokenize_and_lemmatize(self, text):
        tokens = nltk.word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stopwords]
        return " ".join(tokens)

    def preprocess_numerical_data(self, df, numerical_cols):
        df_numerical = df[numerical_cols].copy()
        df_numerical_imputed = self.numerical_imputer.fit_transform(df_numerical)
        if self.numerical_scaler:
            df_numerical_scaled = self.numerical_scaler.fit_transform(df_numerical_imputed)
            return pd.DataFrame(df_numerical_scaled, columns=numerical_cols, index=df.index)
        return pd.DataFrame(df_numerical_imputed, columns=numerical_cols, index=df.index)

    def preprocess_textual_data(self, df, text_col):
        df_text = df[text_col].apply(self._clean_text)
        df_text_processed = df_text.apply(self._tokenize_and_lemmatize)
        text_vectors = self.tfidf_vectorizer.fit_transform(df_text_processed)
        return pd.DataFrame(text_vectors.toarray(), columns=self.tfidf_vectorizer.get_feature_names_out(), index=df.index)

    def integrate_features(self, numerical_features_df, textual_features_df):
        return pd.concat([numerical_features_df, textual_features_df], axis=1)


if __name__ == "__main__":
    # Example Data Loading (Simulated EHR data)
    data = {
        "patient_id": [1, 2, 3, 4, 5],
        "age": [45, 62, 30, np.nan, 71],
        "blood_pressure": [120, 145, 110, 130, 160],
        "cholesterol": [200, 240, 180, 210, np.nan],
        "doctor_notes": [
            "Patient presented with mild chest pain. No acute distress. History of hypertension.",
            "Severe shortness of breath. Suspected pneumonia. Elevated inflammatory markers.",
            "Routine check-up. Healthy individual. Advised lifestyle changes.",
            "Complains of persistent headache and dizziness. Further investigation needed.",
            "Follow-up for diabetes management. Blood sugar levels stable."
        ],
        "symptoms": [
            "Chest pain, hypertension",
            "Shortness of breath, cough",
            "No specific symptoms",
            "Headache, dizziness",
            "Diabetes, stable"
        ]
    }
    df = pd.DataFrame(data)

    numerical_cols = ["age", "blood_pressure", "cholesterol"]
    text_col = "doctor_notes"

    # Initialize the data processor
    processor = PatientDataProcessor(numerical_imputation_strategy="mean", numerical_scaler="standard", max_features_text=100)

    # Preprocess numerical data
    numerical_features = processor.preprocess_numerical_data(df, numerical_cols)
    print("\n--- Preprocessed Numerical Features ---")
    print(numerical_features.head())

    # Preprocess textual data
    textual_features = processor.preprocess_textual_data(df, text_col)
    print("\n--- Preprocessed Textual Features (TF-IDF) ---")
    print(textual_features.head())

    # Integrate features
    combined_features = processor.integrate_features(numerical_features, textual_features)
    print("\n--- Combined Features (Numerical + Textual) ---")
    print(combined_features.head())
    print("\nShape of combined features:", combined_features.shape)
