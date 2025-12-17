import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

# Download NLTK data (run once)
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

class MedicalDataPreprocessor:
    def __init__(self):
        self.numerical_imputer = SimpleImputer(strategy='mean')
        self.scaler = StandardScaler()
        self.text_vectorizer = TfidfVectorizer(max_features=5000)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.categorical_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    def preprocess_numerical_data(self, df):
        numerical_cols = df.select_dtypes(include=np.number).columns
        categorical_cols = df.select_dtypes(include='object').columns

        # Impute numerical features
        df[numerical_cols] = self.numerical_imputer.fit_transform(df[numerical_cols])

        # Scale numerical features
        df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        
        # Encode categorical features
        if not categorical_cols.empty:
            encoded_features = self.categorical_encoder.fit_transform(df[categorical_cols])
            encoded_df = pd.DataFrame(encoded_features, columns=self.categorical_encoder.get_feature_names_out(categorical_cols), index=df.index)
            df = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)

        return df

    def preprocess_text_data(self, text_series):
        processed_texts = []
        for text in text_series:
            tokens = word_tokenize(text.lower())
            filtered_tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in self.stop_words]
            processed_texts.append(" ".join(filtered_tokens))
        
        # Vectorize text
        text_vectors = self.text_vectorizer.fit_transform(processed_texts).toarray()
        return pd.DataFrame(text_vectors, columns=[f'tfidf_{i}' for i in range(text_vectors.shape[1])], index=text_series.index)

    def fit_transform(self, numerical_df, text_series):
        processed_numerical_df = self.preprocess_numerical_data(numerical_df.copy())
        processed_text_df = self.preprocess_text_data(text_series.copy())
        
        # Combine processed data (example: can be used separately or concatenated)
        # For this example, we'll return them separately as their applications might differ
        return processed_numerical_df, processed_text_df

# Example Usage
if __name__ == "__main__":
    # Simulate numerical patient data
    data = {
        'Age': [45, 62, 30, 70, 55, np.nan, 68, 40, 50, 75],
        'BloodPressure': [120, 140, 110, 150, 130, 125, 145, 115, 128, 160],
        'Cholesterol': [200, 240, 180, 260, 220, 210, 250, 190, 230, np.nan],
        'Glucose': [90, 110, 85, 130, 95, 100, 120, 88, 105, 140],
        'Smoking': ['No', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'No']
    }
    numerical_data_df = pd.DataFrame(data)

    # Simulate unstructured medical notes
    medical_notes_series = pd.Series([
        "Patient presents with chronic cough and fatigue. History of smoking for 20 years.",
        "Elevated blood pressure and high cholesterol noted. Recommend lifestyle changes.",
        "Young patient with no significant medical history. Routine check-up.",
        "Severe chest pain reported, suggestive of cardiac issues. Further tests ordered.",
        "Diabetes diagnosis confirmed. Medication prescribed. Follow-up in 3 months.",
        "Complains of persistent headaches and dizziness. Neurological examination needed.",
        "Long-standing hypertension and kidney disease. Current medication stable.",
        "Mild discomfort in the abdomen. Advised bland diet.",
        "Follow-up after stent placement. Patient is recovering well.",
        "New onset of joint pain and swelling. Possible autoimmune condition."
    ], name='MedicalNotes')

    preprocessor = MedicalDataPreprocessor()

    # Process data
    processed_numerical_df, processed_text_df = preprocessor.fit_transform(numerical_data_df, medical_notes_series)

    print("\nOriginal Numerical Data Head:")
    print(numerical_data_df.head())
    print("\nProcessed Numerical Data Head (Imputed, Scaled, Encoded):")
    print(processed_numerical_df.head())

    print("\nOriginal Medical Notes Head:")
    print(medical_notes_series.head())
    print("\nProcessed Text Data Head (TF-IDF Vectors):")
    print(processed_text_df.head())
    print("Processed Text Data Shape:", processed_text_df.shape)
