import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# Download necessary NLTK data
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

class EHRPreprocessingPipeline:
    def __init__(self):
        self.numerical_imputer = SimpleImputer(strategy="mean")
        self.scaler = StandardScaler()
        self.onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

    def load_ehr_data(self, numerical_data_path, text_data_path):
        df_numerical = pd.read_csv(numerical_data_path)
        df_text = pd.read_csv(text_data_path)
        return df_numerical, df_text

    def preprocess_numerical_data(self, df_numerical, numerical_cols, categorical_numerical_cols):
        df_processed_numerical = df_numerical.copy()

        # Impute missing values for numerical columns
        if numerical_cols:
            df_processed_numerical[numerical_cols] = self.numerical_imputer.fit_transform(df_processed_numerical[numerical_cols])
        
        # Scale numerical features
        if numerical_cols:
            scaled_numerical_features = self.scaler.fit_transform(df_processed_numerical[numerical_cols])
            df_processed_numerical[numerical_cols] = scaled_numerical_features

        # One-hot encode categorical numerical features
        if categorical_numerical_cols:
            encoded_features = self.onehot_encoder.fit_transform(df_processed_numerical[categorical_numerical_cols])
            encoded_df = pd.DataFrame(encoded_features, columns=self.onehot_encoder.get_feature_names_out(categorical_numerical_cols), index=df_processed_numerical.index)
            df_processed_numerical = df_processed_numerical.drop(columns=categorical_numerical_cols).merge(encoded_df, left_index=True, right_index=True)
        
        return df_processed_numerical

    def _clean_text(self, text):
        if not isinstance(text, str): # Handle non-string inputs, e.g., NaNs
            return ""
        tokens = word_tokenize(text.lower())
        lemmatized_tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in self.stop_words]
        return " ".join(lemmatized_tokens)

    def preprocess_text_data(self, df_text, text_col):
        df_preprocessed_text = df_text.copy()
        df_preprocessed_text["cleaned_text"] = df_preprocessed_text[text_col].apply(self._clean_text)
        return df_preprocessed_text["cleaned_text"]

    def vectorize_text_data(self, preprocessed_text_series):
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(preprocessed_text_series)
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=self.tfidf_vectorizer.get_feature_names_out(), index=preprocessed_text_series.index)
        return tfidf_df

    def integrate_features(self, numerical_features_df, text_features_df):
        # Ensure indices align for concatenation
        numerical_features_df = numerical_features_df.reset_index(drop=True)
        text_features_df = text_features_df.reset_index(drop=True)
        
        combined_df = pd.concat([numerical_features_df, text_features_df], axis=1)
        return combined_df

# --- Main Orchestration Script (for demonstration) ---
if __name__ == "__main__":
    # Create dummy CSV files for demonstration
    dummy_numerical_data = {
        "patient_id": [1, 2, 3, 4, 5, 6],
        "age": [30, 45, 60, 25, np.nan, 70],
        "blood_pressure": [120, 140, 130, 110, 150, 135],
        "cholesterol": [200, 240, 180, 190, 230, np.nan],
        "gender": ["M", "F", "M", "F", "M", "F"]
    }
    df_num_dummy = pd.DataFrame(dummy_numerical_data)
    df_num_dummy.to_csv("numerical_data.csv", index=False)

    dummy_text_data = {
        "patient_id": [1, 2, 3, 4, 5, 6],
        "doctor_notes": [
            "Patient presented with mild fever and cough. Prescribed antibiotics.",
            "Routine check-up, no significant findings. Patient reports feeling well.",
            "Severe chest pain reported, admitted for observation. History of heart disease.",
            "Follow-up after surgery, incision healing well. Advised rest.",
            "Patient complaining of fatigue. Lab results pending.",
            "No new complaints. Continue current medication."
        ]
    }
    df_text_dummy = pd.DataFrame(dummy_text_data)
    df_text_dummy.to_csv("text_data.csv", index=False)

    pipeline = EHRPreprocessingPipeline()

    # Define file paths for the dummy data
    numerical_data_filepath = "numerical_data.csv"
    text_data_filepath = "text_data.csv"

    # 1. Load Data
    df_numerical, df_text = pipeline.load_ehr_data(numerical_data_filepath, text_data_filepath)
    print("\n--- Raw Numerical Data ---")
    print(df_numerical.head())
    print("\n--- Raw Text Data ---")
    print(df_text.head())

    # Define columns for numerical preprocessing
    numerical_cols = ["age", "blood_pressure", "cholesterol"]
    categorical_numerical_cols = ["gender"]

    # 2. Preprocess Numerical Data
    df_processed_numerical = pipeline.preprocess_numerical_data(df_numerical.drop(columns="patient_id"), numerical_cols, categorical_numerical_cols)
    print("\n--- Processed Numerical Data ---")
    print(df_processed_numerical.head())
    print(f"Shape after numerical preprocessing: {df_processed_numerical.shape}")

    # 3. Preprocess Text Data
    preprocessed_text_series = pipeline.preprocess_text_data(df_text, "doctor_notes")
    print("\n--- Preprocessed Text Sample ---")
    print(preprocessed_text_series.head())

    # 4. Vectorize Text Data
    df_vectorized_text = pipeline.vectorize_text_data(preprocessed_text_series)
    print("\n--- Vectorized Text Sample (TF-IDF) ---")
    print(df_vectorized_text.head())
    print(f"Shape after text vectorization: {df_vectorized_text.shape}")

    # 5. Integrate Features
    # Ensure both dataframes have the same number of rows before integration
    # In this example, patient_id is dropped from numerical for processing, 
    # but we assume original data rows match.
    # The reset_index is crucial here for proper concatenation.
    final_combined_features = pipeline.integrate_features(df_processed_numerical, df_vectorized_text)
    print("\n--- Final Combined Features (Head) ---")
    print(final_combined_features.head())
    print(f"Final combined features shape: {final_combined_features.shape}")

    print("\nPreprocessing pipeline completed. The combined dataset is ready for ML model training.")

