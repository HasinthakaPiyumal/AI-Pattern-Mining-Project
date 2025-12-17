import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

# Download NLTK resources (if not already downloaded)
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# Custom Transformer for Numerical Data
class NumericalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, imputer_strategy='mean'):
        self.imputer_strategy = imputer_strategy
        self.imputer = SimpleImputer(strategy=self.imputer_strategy)
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.imputer.fit(X)
        self.scaler.fit(self.imputer.transform(X))
        return self

    def transform(self, X):
        X_imputed = self.imputer.transform(X)
        return self.scaler.transform(X_imputed)

# Custom Transformer for Text Data
class TextTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()

    def clean_text(self, text):
        text = text.lower()  # Lowercase
        text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove punctuation
        tokens = nltk.word_tokenize(text)  # Tokenize
        tokens = [word for word in tokens if word not in self.stop_words]  # Remove stopwords
        tokens = [self.stemmer.stem(word) for word in tokens]  # Stemming
        return ' '.join(tokens)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(self.clean_text)

# 1. Generate Synthetic Patient Data
data = {
    'age': [30, 45, 60, 25, 50, 70, 35, 40, 55, 65],
    'blood_pressure': [120, 140, 160, 110, 130, 170, 125, 135, 150, 155],
    'cholesterol': [200, 240, 280, 180, 220, 300, 210, 230, 260, 270],
    'diabetes_status': [0, 1, 1, 0, 0, 1, 0, 0, 1, 1],
    'doctor_notes': [
        'Patient reports mild chest pain and fatigue.',
        'Follow-up for hypertension, good progress.',
        'Diabetes management review, new medication.',
        'Routine check-up, no significant findings.',
        'Symptoms of high cholesterol, advised diet change.',
        'Severe joint pain, needs further investigation.',
        'Annual physical, all vitals normal.',
        'Mild fever, prescribed antibiotics.',
        'Cardiovascular risk assessment, family history.',
        'Chronic back pain, physiotherapist referral.'
    ],
    'outcome': [0, 0, 1, 0, 0, 1, 0, 0, 1, 1] # 0: Healthy, 1: At Risk
}
df = pd.DataFrame(data)

X = df.drop('outcome', axis=1)
y = df['outcome']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define numerical and text features
numerical_features = ['age', 'blood_pressure', 'cholesterol']
text_features = 'doctor_notes'

# Create preprocessing pipelines for numerical and text data
numerical_pipeline = Pipeline([
    ('numerical_transformer', NumericalTransformer())
])

text_pipeline = Pipeline([
    ('text_transformer', TextTransformer()),
    ('tfidf_vectorizer', TfidfVectorizer(max_features=1000))
])

# Combine numerical and text features using FeatureUnion
preprocessor = FeatureUnion(transformer_list=[
    ('numerical_features', numerical_pipeline, numerical_features),
    ('text_features', text_pipeline, text_features)
])

# Create a full pipeline including preprocessing and a classifier
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, solver='liblinear'))
])

# Train the model
full_pipeline.fit(X_train, y_train)

# Make predictions
y_pred = full_pipeline.predict(X_test)

print("Model trained successfully and predictions made.")
print("\nSample Prediction for the first test instance:")
sample_instance = X_test.iloc[0:1]
predicted_outcome = full_pipeline.predict(sample_instance)
print(f"Patient data:\n{sample_instance}")
print(f"Predicted Outcome (0: Healthy, 1: At Risk): {predicted_outcome[0]}")
print(f"Actual Outcome: {y_test.iloc[0]}")

# Example of how to preprocess a new single instance
new_patient_data = pd.DataFrame([
    {
        'age': 58,
        'blood_pressure': 150,
        'cholesterol': 260,
        'diabetes_status': 1,
        'doctor_notes': 'Patient reports high blood sugar and frequent thirst. Needs urgent review.'
    }
])

new_patient_prediction = full_pipeline.predict(new_patient_data)
print(f"\nPrediction for a new patient: {new_patient_prediction[0]} (0: Healthy, 1: At Risk)")