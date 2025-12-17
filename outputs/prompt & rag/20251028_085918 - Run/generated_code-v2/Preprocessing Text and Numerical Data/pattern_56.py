import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

try:
    nltk.data.find("corpora/stopwords")
except nltk.downloader.DownloadError:
    nltk.download("stopwords")
try:
    nltk.data.find("tokenizers/punkt")
except nltk.downloader.DownloadError:
    nltk.download("punkt")

def simulate_patient_data(num_records=100):
    np.random.seed(42)
    data = {
        "patient_id": range(1, num_records + 1),
        "age": np.random.randint(20, 80, num_records),
        "gender": np.random.choice(["Male", "Female"], num_records),
        "weight_kg": np.random.normal(70, 10, num_records),
        "height_cm": np.random.normal(170, 10, num_records),
        "blood_pressure_sys": np.random.normal(120, 15, num_records),
        "blood_pressure_dia": np.random.normal(80, 10, num_records),
        "cholesterol_mgdl": np.random.normal(200, 30, num_records),
        "glucose_mgdl": np.random.normal(100, 20, num_records),
        "medical_history": [
            "Patient presented with flu-like symptoms. History of hypertension.",
            "Routine check-up, no major complaints. Mild allergic reactions.",
            "Follow-up on recent surgery. Recovering well. Diabetes managed.",
            "Initial consultation for chronic headaches. Family history of migraines.",
            "Emergency visit due to chest pain. Suspected cardiac issue. Smoker."
        ] * (num_records // 5) + ["" for _ in range(num_records % 5)], # Add some empty strings for robustness
        "doctor_notes": [
            "Prescribed antibiotics. Advised rest.",
            "Patient advised to continue current medication.",
            "Scheduled for another check-up next month. All vitals stable.",
            "Referred to a neurologist. Monitor pain levels.",
            "Immediate admission for further tests. Administered pain relief."
        ] * (num_records // 5) + ["" for _ in range(num_records % 5)], # Add some empty strings for robustness
    }
    df = pd.DataFrame(data)

    # Introduce some missing values for demonstration
    for col in ["weight_kg", "blood_pressure_sys", "cholesterol_mgdl", "medical_history"]:
        missing_indices = np.random.choice(df.index, size=int(num_records * 0.05), replace=False)
        if df[col].dtype == object: # For text data, use None or NaN
            df.loc[missing_indices, col] = None
        else:
            df.loc[missing_indices, col] = np.nan

    return df

class NumericalPreprocessor:
    def __init__(self, numerical_cols, categorical_cols):
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        
        numerical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, self.numerical_cols),
                ("cat", categorical_transformer, self.categorical_cols)
            ])

    def fit_transform(self, df):
        return self.preprocessor.fit_transform(df)
    
    def transform(self, df):
        return self.preprocessor.transform(df)
    
    def get_feature_names_out(self):
        return self.preprocessor.get_feature_names_out()

class TextPreprocessor:
    def __init__(self, text_cols):
        self.text_cols = text_cols
        self.stop_words = set(stopwords.words("english"))
        self.stemmer = PorterStemmer()
        self.vectorizers = {col: TfidfVectorizer(max_features=100) for col in text_cols}

    def _preprocess_text(self, text):
        if pd.isna(text) or text is None or text == "":
            return ""
        tokens = nltk.word_tokenize(text.lower())
        filtered_tokens = [self.stemmer.stem(word) for word in tokens if word.isalnum() and word not in self.stop_words]
        return " ".join(filtered_tokens)

    def fit_transform(self, df):
        processed_texts_dict = {}
        for col in self.text_cols:
            df_copy = df.copy()
            df_copy[col] = df_copy[col].apply(self._preprocess_text)
            processed_texts_dict[col] = self.vectorizers[col].fit_transform(df_copy[col])
        return processed_texts_dict
    
    def transform(self, df):
        processed_texts_dict = {}
        for col in self.text_cols:
            df_copy = df.copy()
            df_copy[col] = df_copy[col].apply(self._preprocess_text)
            processed_texts_dict[col] = self.vectorizers[col].transform(df_copy[col])
        return processed_texts_dict

def main():
    print("Simulating patient data...")
    raw_data_df = simulate_patient_data(num_records=200)
    print("Raw Data Head:\n", raw_data_df.head())
    print("Raw Data Info:\n", raw_data_df.info())

    numerical_features = ["age", "weight_kg", "height_cm", "blood_pressure_sys", "blood_pressure_dia", "cholesterol_mgdl", "glucose_mgdl"]
    categorical_features = ["gender"]
    text_features = ["medical_history", "doctor_notes"]

    print("\nInitializing Numerical Preprocessor...")
    numerical_preprocessor = NumericalPreprocessor(numerical_features, categorical_features)
    numerical_transformed_array = numerical_preprocessor.fit_transform(raw_data_df)
    numerical_feature_names = numerical_preprocessor.get_feature_names_out()
    numerical_transformed_df = pd.DataFrame(numerical_transformed_array, columns=numerical_feature_names)
    print("Numerical Transformed Data Head:\n", numerical_transformed_df.head())

    print("\nInitializing Text Preprocessor...")
    text_preprocessor = TextPreprocessor(text_features)
    text_transformed_dict = text_preprocessor.fit_transform(raw_data_df)
    
    text_transformed_dfs = []
    for col, sparse_matrix in text_transformed_dict.items():
        tfidf_feature_names = text_preprocessor.vectorizers[col].get_feature_names_out()
        col_names = [f"{col}_{name}" for name in tfidf_feature_names]
        text_transformed_dfs.append(pd.DataFrame(sparse_matrix.toarray(), columns=col_names))
    
    combined_text_df = pd.concat(text_transformed_dfs, axis=1)
    print("Text Transformed Data Head:\n", combined_text_df.head())

    print("\nIntegrating preprocessed data...")
    # Align indices for concatenation
    raw_data_df_reset = raw_data_df.reset_index(drop=True)
    integrated_df = pd.concat([raw_data_df_reset[["patient_id"]], numerical_transformed_df, combined_text_df], axis=1)
    
    print("\nFinal Integrated Preprocessed Data Head:\n", integrated_df.head())
    print("Final Integrated Preprocessed Data Info:\n", integrated_df.info())
    print("Final Integrated Preprocessed Data Shape:\n", integrated_df.shape)

if __name__ == "__main__":
    main()