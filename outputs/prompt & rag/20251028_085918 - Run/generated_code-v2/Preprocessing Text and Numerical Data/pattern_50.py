import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy


class MedicalRecordProcessor:
    def __init__(self):
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

        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.nlp = spacy.load('en_core_web_sm') 

        self.numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        self.categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        self.preprocessor = None 

    def ingest_data(self, data_path=None, dummy_data=None):
        if data_path:
            return pd.read_csv(data_path)
        elif dummy_data is not None:
            return pd.DataFrame(dummy_data)
        else:
            raise ValueError("Either data_path or dummy_data must be provided.")

    def _preprocess_text(self, text):
        doc = self.nlp(text.lower())
        tokens = [token.lemma_ for token in doc if token.is_alpha and token.lemma_ not in self.stop_words]
        return " ".join(tokens)

    def process_medical_records(self, df, numerical_cols, categorical_cols, text_cols):
        df_processed_numerical = df[numerical_cols].copy()
        df_processed_categorical = df[categorical_cols].copy()
        df_processed_text = df[text_cols].copy()

        numerical_features = self.numerical_transformer.fit_transform(df_processed_numerical)
        
        categorical_features = self.categorical_transformer.fit_transform(df_processed_categorical)

        cleaned_texts = df_processed_text[text_cols[0]].apply(self._preprocess_text)
        text_features = self.tfidf_vectorizer.fit_transform(cleaned_texts).toarray()
        
        combined_features = np.hstack((numerical_features, categorical_features, text_features))
        return combined_features


if __name__ == '__main__':
    processor = MedicalRecordProcessor()

    dummy_medical_data = {
        'patient_id': [1, 2, 3, 4, 5],
        'age': [45, 62, 30, 78, 55],
        'weight_kg': [70, 85, 60, np.nan, 75],
        'cholesterol_mgdl': [200, 240, 180, 280, 210],
        'blood_type': ['A+', 'B-', 'O+', 'A+', 'AB-'],
        'diagnosis_code': ['D23', 'C18', 'D23', 'I10', 'C18'],
        'doctor_notes': [
            "Patient presented with mild chest pain. Advised for EKG and follow-up.",
            "Colonoscopy revealed polyps. Biopsy sent to lab. Schedule for surgery.",
            "Routine check-up. No significant issues. Patient is healthy.",
            "Severe hypertension. Prescribed medication. Monitor blood pressure daily.",
            "Suspected appendicitis. Sent for emergency surgery. Post-op recovery ongoing."
        ]
    }

    df = processor.ingest_data(dummy_data=dummy_medical_data)

    numerical_columns = ['age', 'weight_kg', 'cholesterol_mgdl']
    categorical_columns = ['blood_type', 'diagnosis_code']
    text_columns = ['doctor_notes']

    processed_features = processor.process_medical_records(df, numerical_columns, categorical_columns, text_columns)

    print("Shape of processed features:", processed_features.shape)
    print("\nFirst 5 rows of processed features:\n", processed_features[:5, :10]) # Print first 10 columns for brevity

    print("\nDemonstrating NER with SpaCy on a sample note:")
    sample_note = "Patient experiencing severe headache and fever. Prescribed ibuprofen and bed rest."
    doc = processor.nlp(sample_note)
    for ent in doc.ents:
        print(f"Entity: {ent.text}, Label: {ent.label_}")