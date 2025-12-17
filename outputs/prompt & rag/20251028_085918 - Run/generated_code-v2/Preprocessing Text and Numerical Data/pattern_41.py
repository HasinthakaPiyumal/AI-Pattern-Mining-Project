import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Download necessary NLTK data
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

class MedicalDiagnosisPipeline:
    def __init__(self):
        self.numerical_imputer = SimpleImputer(strategy='mean')
        self.numerical_scaler = StandardScaler()
        self.onehot_encoder = OneHotEncoder(handle_unknown='ignore')
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        self.classifier = RandomForestClassifier(random_state=42)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        self.numerical_cols = ['age', 'lab_result_A', 'vital_sign_B', 'temperature']
        self.categorical_cols = ['gender', 'blood_type']
        self.text_cols = ['doctors_notes', 'patient_symptoms']
        self.all_feature_cols = self.numerical_cols + self.categorical_cols + self.text_cols
        self.target_col = 'diagnosis'

        self.fitted_numerical_imputer = None
        self.fitted_numerical_scaler = None
        self.fitted_onehot_encoder = None
        self.fitted_tfidf_vectorizer = None
        self.fitted_classifier = None

    def _preprocess_text(self, text):
        if not isinstance(text, str): # Handle non-string inputs (e.g., NaN)
            return ""
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text) # Remove non-alphabetic characters
        tokens = nltk.word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        return ' '.join(tokens)

    def fit(self, X, y):
        # Numerical Preprocessing
        X_numerical = X[self.numerical_cols]
        self.fitted_numerical_imputer = self.numerical_imputer.fit(X_numerical)
        X_numerical_imputed = self.fitted_numerical_imputer.transform(X_numerical)
        self.fitted_numerical_scaler = self.numerical_scaler.fit(X_numerical_imputed)
        X_numerical_scaled = self.fitted_numerical_scaler.transform(X_numerical_imputed)

        # Categorical Preprocessing
        X_categorical = X[self.categorical_cols]
        self.fitted_onehot_encoder = self.onehot_encoder.fit(X_categorical)
        X_categorical_encoded = self.fitted_onehot_encoder.transform(X_categorical).toarray()

        # Text Preprocessing
        X_text_combined = (X[self.text_cols[0]].fillna('') + " " + X[self.text_cols[1]].fillna('')).apply(self._preprocess_text)
        self.fitted_tfidf_vectorizer = self.tfidf_vectorizer.fit(X_text_combined)
        X_text_vectorized = self.fitted_tfidf_vectorizer.transform(X_text_combined).toarray()

        # Concatenate all features
        X_processed = np.hstack((X_numerical_scaled, X_categorical_encoded, X_text_vectorized))

        # Train Classifier
        self.fitted_classifier = self.classifier.fit(X_processed, y)

    def transform(self, X):
        # Numerical Preprocessing
        X_numerical = X[self.numerical_cols]
        X_numerical_imputed = self.fitted_numerical_imputer.transform(X_numerical)
        X_numerical_scaled = self.fitted_numerical_scaler.transform(X_numerical_imputed)

        # Categorical Preprocessing
        X_categorical = X[self.categorical_cols]
        X_categorical_encoded = self.fitted_onehot_encoder.transform(X_categorical).toarray()

        # Text Preprocessing
        X_text_combined = (X[self.text_cols[0]].fillna('') + " " + X[self.text_cols[1]].fillna('')).apply(self._preprocess_text)
        X_text_vectorized = self.fitted_tfidf_vectorizer.transform(X_text_combined).toarray()

        # Concatenate all features
        X_processed = np.hstack((X_numerical_scaled, X_categorical_encoded, X_text_vectorized))
        return X_processed

    def predict(self, X):
        X_processed = self.transform(X)
        return self.fitted_classifier.predict(X_processed)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y, y_pred, average='weighted', zero_division=0)

        print(f"Model Evaluation:\n")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}

def main():
    # 1. Generate Dummy Data
    np.random.seed(42)
    data_size = 500
    data = {
        'age': np.random.randint(18, 90, data_size),
        'gender': np.random.choice(['Male', 'Female'], data_size),
        'blood_type': np.random.choice(['A+', 'B+', 'AB+', 'O+'], data_size),
        'lab_result_A': np.random.rand(data_size) * 100, # Continuous numerical
        'vital_sign_B': np.random.normal(70, 10, data_size), # Continuous numerical
        'temperature': np.random.normal(98.6, 1.5, data_size),
        'doctors_notes': [f"Patient presented with {np.random.choice(['fever', 'cough', 'sore throat'])} and {np.random.choice(['fatigue', 'headache'])}. No severe complications noted." for _ in range(data_size)],
        'patient_symptoms': [f"I have a {np.random.choice(['mild', 'severe'])} {np.random.choice(['fever', 'cough', 'rash'])} and feel {np.random.choice(['tired', 'weak'])}." for _ in range(data_size)],
        'diagnosis': np.random.choice(['Flu', 'Cold', 'Allergy', 'Bronchitis'], data_size, p=[0.3, 0.3, 0.2, 0.2])
    }

    # Introduce some missing values for demonstration
    for col in ['lab_result_A', 'vital_sign_B', 'doctors_notes']:
        missing_indices = np.random.choice(data_size, int(data_size * 0.05), replace=False)
        data[col][missing_indices] = np.nan
    
    df = pd.DataFrame(data)

    X = df.drop(columns=['diagnosis'])
    y = df['diagnosis']

    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. Initialize and Run Pipeline
    medical_pipeline = MedicalDiagnosisPipeline()
    print("\nFitting and transforming data...")
    medical_pipeline.fit(X_train, y_train)

    # 4. Evaluate Model
    print("\nEvaluating model on test data...")
    medical_pipeline.evaluate(X_test, y_test)

    # Example of predicting on new data (demonstrative)
    print("\nDemonstrating prediction on a new (dummy) patient record:")
    new_patient_data = pd.DataFrame({
        'age': [65],
        'gender': ['Female'],
        'blood_type': ['O+'],
        'lab_result_A': [75.0],
        'vital_sign_B': [68.0],
        'temperature': [99.1],
        'doctors_notes': ['Patient reports mild headache and some nausea. Recent travel history.'],
        'patient_symptoms': ['Feeling a bit off, lightheaded, and stomach ache.']
    })
    prediction = medical_pipeline.predict(new_patient_data)
    print(f"Predicted diagnosis for new patient: {prediction[0]}")

if __name__ == "__main__":
    main()
