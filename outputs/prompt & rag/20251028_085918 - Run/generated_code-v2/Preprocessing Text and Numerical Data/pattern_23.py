import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, SimpleImputer, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

try:
    nltk.data.find("corpora/stopwords")
except nltk.downloader.DownloadError:
    nltk.download("stopwords")
try:
    nltk.data.find("corpora/wordnet")
except nltk.downloader.DownloadError:
    nltk.download("wordnet")
try:
    nltk.data.find("tokenizers/punkt")
except nltk.downloader.DownloadError:
    nltk.download("punkt")

def preprocess_numerical_data(df_numerical):
    numerical_cols = df_numerical.select_dtypes(include=np.number).columns
    categorical_cols = df_numerical.select_dtypes(include="object").columns

    # Impute missing numerical values
    imputer = SimpleImputer(strategy="mean")
    df_numerical[numerical_cols] = imputer.fit_transform(df_numerical[numerical_cols])

    # Scale numerical features
    scaler = StandardScaler()
    df_numerical[numerical_cols] = scaler.fit_transform(df_numerical[numerical_cols])

    # One-hot encode categorical features (if any)
    if len(categorical_cols) > 0:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoded_features = encoder.fit_transform(df_numerical[categorical_cols])
        encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols), index=df_numerical.index)
        df_numerical = pd.concat([df_numerical.drop(columns=categorical_cols), encoded_df], axis=1)

    return df_numerical

def preprocess_text_data(df_text, text_column_name):
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    def clean_text(text):
        if not isinstance(text, str):
            return ""
        tokens = word_tokenize(text.lower())
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
        return " ".join(tokens)

    df_text["cleaned_text"] = df_text[text_column_name].apply(clean_text)

    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(max_features=1000) # Limit features for demonstration
    tfidf_matrix = tfidf_vectorizer.fit_transform(df_text["cleaned_text"])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out(), index=df_text.index)

    return tfidf_df

def main():
    # Sample Data Generation (replace with actual data loading in a real application)
    data = {
        "patient_id": [1, 2, 3, 4, 5],
        "age": [45, 62, 30, 70, 55],
        "blood_pressure": [120, 140, 110, np.nan, 130],
        "cholesterol": [200, 240, 180, 280, np.nan],
        "bmi": [25.1, 30.5, 22.3, 35.0, 28.7],
        "gender": ["Male", "Female", "Female", "Male", "Female"],
        "physician_notes": [
            "Patient presented with mild chest pain, advised rest.",
            "History of diabetes and hypertension, regular check-up.",
            "No significant findings, patient feels well.",
            "Chronic cough, difficulty breathing, needs further investigation.",
            "Routine visit, minor headache, prescribed pain relievers."
        ]
    }
    df = pd.DataFrame(data)

    # Separate numerical and text data
    df_numerical_raw = df[["age", "blood_pressure", "cholesterol", "bmi", "gender"]].copy()
    df_text_raw = df[["patient_id", "physician_notes"]].copy()

    print("--- Original Numerical Data ---")
    print(df_numerical_raw)
    print("\n--- Original Text Data ---")
    print(df_text_raw.head())

    # Preprocess numerical data
    preprocessed_numerical_df = preprocess_numerical_data(df_numerical_raw)
    print("\n--- Preprocessed Numerical Data ---")
    print(preprocessed_numerical_df)

    # Preprocess text data
    preprocessed_text_df = preprocess_text_data(df_text_raw, "physician_notes")
    print("\n--- Preprocessed Text Data (TF-IDF) ---")
    print(preprocessed_text_df.head())

    # Combine preprocessed data
    # Ensure indices align for proper concatenation
    df_combined = pd.concat([preprocessed_numerical_df.reset_index(drop=True), preprocessed_text_df.reset_index(drop=True)], axis=1)

    print("\n--- Combined Preprocessed Dataset (Head) ---")
    print(df_combined.head())
    print(f"Combined dataset shape: {df_combined.shape}")

if __name__ == "__main__":
    main()