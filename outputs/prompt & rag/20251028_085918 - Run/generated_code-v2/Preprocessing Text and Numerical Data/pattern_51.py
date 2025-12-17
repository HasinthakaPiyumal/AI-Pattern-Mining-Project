import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

class DataPreprocessor:
    def __init__(self):
        self.numerical_imputer = SimpleImputer(strategy="mean")
        self.numerical_scaler = StandardScaler()
        self.tfidf_vectorizer = TfidfVectorizer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

    def preprocess_numerical_data(self, df_numerical):
        imputed_data = self.numerical_imputer.fit_transform(df_numerical)
        scaled_data = self.numerical_scaler.fit_transform(imputed_data)
        return scaled_data

    def _clean_text(self, text):
        tokens = word_tokenize(text.lower())
        filtered_tokens = [token for token in tokens if token.isalpha() and token not in self.stop_words]
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in filtered_tokens]
        return " ".join(lemmatized_tokens)

    def preprocess_text_data(self, series_text):
        cleaned_texts = series_text.apply(self._clean_text)
        tfidf_features = self.tfidf_vectorizer.fit_transform(cleaned_texts)
        return tfidf_features.toarray()

    def fit_transform(self, df_numerical, series_text):
        processed_numerical = self.preprocess_numerical_data(df_numerical)
        processed_text = self.preprocess_text_data(series_text)
        
        # Concatenate numerical and text features
        # Ensure consistent shape for concatenation (e.g., convert text features to dense array if sparse)
        all_features = np.hstack((processed_numerical, processed_text))
        return all_features

if __name__ == "__main__":
    # Example Usage:
    # Dummy Patient EHR Data
    numerical_data = {
        "Age": [35, 42, 58, 29, np.nan, 65, 49],
        "BMI": [24.5, 31.2, 28.1, 22.9, 26.7, 33.0, np.nan],
        "Cholesterol": [180, 220, 205, 160, 195, 240, 210],
        "BloodPressure_Systolic": [120, 140, 135, 110, 128, 150, 130]
    }
    df_numerical = pd.DataFrame(numerical_data)

    text_data = [
        "Patient presented with mild cough and fever, no severe symptoms observed. Family history of diabetes.",
        "Admitted due to severe chest pain. ECG shows abnormalities. No prior cardiac issues reported.",
        "Routine check-up. Patient complains about occasional headaches. Advised hydration.",
        "Diagnosed with influenza. Prescribed antiviral medication. No significant medical history.",
        "Follow-up for hypertension. Blood pressure is stable. Notes anxiety symptoms.",
        "New patient with persistent fatigue and unexplained weight loss. Further investigation required.",
        "Emergency visit for a fall. Minor contusion. Patient denies any pain."
    ]
    series_text = pd.Series(text_data)

    # Initialize and use the preprocessor
    preprocessor = DataPreprocessor()
    combined_features = preprocessor.fit_transform(df_numerical, series_text)

    print("Original Numerical Data:\n", df_numerical)
    print("\nOriginal Text Data:\n", series_text)
    print("\nShape of Preprocessed Features (Numerical + Text):", combined_features.shape)
    print("\nFirst 5 rows of Combined Preprocessed Features:\n", combined_features[:5])
