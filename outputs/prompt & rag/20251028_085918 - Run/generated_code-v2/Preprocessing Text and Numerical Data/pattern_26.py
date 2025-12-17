import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import SimpleImputer, StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.sparse import hstack
import xgboost as xgb
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Ensure NLTK resources are downloaded
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except nltk.downloader.DownloadError:
    nltk.download('wordnet')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except nltk.downloader.DownloadError:
    nltk.download('averaged_perceptron_tagger')

# Initialize NLTK components
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower() # Convert to string and lowercase
    text = re.sub(r'[^a-z]', ' ', text) # Remove non-alphabetic characters
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra whitespace
    return text

def preprocess_text_data(text_series, vectorizer=None, fit_vectorizer=True):
    cleaned_texts = text_series.apply(clean_text)
    tokenized_texts = cleaned_texts.apply(lambda x: [lemmatizer.lemmatize(word) for word in x.split() if word not in stop_words])
    processed_texts = tokenized_texts.apply(lambda x: ' '.join(x))
    
    if fit_vectorizer:
        vectorizer = TfidfVectorizer(max_features=5000) # Limit features for demonstration
        text_features = vectorizer.fit_transform(processed_texts)
    else:
        if vectorizer is None:
            raise ValueError("Vectorizer must be provided for transformation.")
        text_features = vectorizer.transform(processed_texts)
        
    return text_features, vectorizer

class PatientData(BaseModel):
    age: float
    gender: str
    vital_signs: float
    lab_result: float
    diagnosis_code: str
    length_of_stay: float
    clinical_notes: str

# --- Main Application Logic ---

# 1. Simulate Data Ingestion
data = {
    'age': [65, 72, 55, 80, 45, 60, 70, 50, 78, 62],
    'gender': ['Male', 'Female', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male'],
    'vital_signs': [120, 130, 110, 140, 115, 125, 135, 118, 142, 128],
    'lab_result': [5.2, 6.1, 4.8, 7.0, 5.5, 5.8, 6.5, 4.9, 7.2, 5.9],
    'diagnosis_code': ['I10', 'J44', 'E11', 'I50', 'G40', 'I10', 'J44', 'E11', 'I50', 'G40'],
    'length_of_stay': [7, 10, 5, 12, 4, 8, 9, 6, 11, 7],
    'clinical_notes': [
        'Patient admitted with severe chest pain. Stabilized and discharged.',
        'COPD exacerbation. Required respiratory support. Follow-up planned.',
        'Diabetes management. Education provided. Good recovery.',
        'Heart failure acute decompensation. Medications adjusted.',
        'Seizure disorder. Medication review. Stable condition.',
        'Hypertension and kidney issues. Close monitoring.',
        'Chronic bronchitis. Breathing exercises taught.',
        'Type 2 diabetes. Diet counseling.',
        'Congestive heart failure. Fluid restriction initiated.',
        'Epilepsy follow-up. No recent events.'
    ],
    'readmitted': [0, 1, 0, 1, 0, 0, 1, 0, 1, 0] # Target variable
}
df = pd.DataFrame(data)

# Define target and features
X = df.drop('readmitted', axis=1)
y = df['readmitted']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Data Preprocessing Module

# Define numerical and categorical features
numerical_features = ['age', 'vital_signs', 'lab_result', 'length_of_stay']
categorical_features = ['gender', 'diagnosis_code']
text_feature = 'clinical_notes'

# Create a preprocessor for numerical and categorical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ], 
    remainder='drop'
)

# Fit and transform numerical/categorical data
X_train_numerical_categorical = preprocessor.fit_transform(X_train)
X_test_numerical_categorical = preprocessor.transform(X_test)

# Preprocess text data
X_train_text_features, text_vectorizer = preprocess_text_data(X_train[text_feature], fit_vectorizer=True)
X_test_text_features, _ = preprocess_text_data(X_test[text_feature], vectorizer=text_vectorizer, fit_vectorizer=False)

# Combine all features
X_train_processed = hstack([X_train_numerical_categorical, X_train_text_features])
X_test_processed = hstack([X_test_numerical_categorical, X_test_text_features])

# 3. Model Training & Evaluation Module

# Initialize and train the XGBoost Classifier
model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    use_label_encoder=False, 
    random_state=42
)
model.fit(X_train_processed, y_train)

# For real-world use, save the model and preprocessors
joblib.dump(model, 'readmission_model.pkl')
joblib.dump(preprocessor, 'numerical_categorical_preprocessor.pkl')
joblib.dump(text_vectorizer, 'text_vectorizer.pkl')

# Load them for API (demonstration purposes, in a real app these would be loaded once on startup)
loaded_model = joblib.load('readmission_model.pkl')
loaded_preprocessor = joblib.load('numerical_categorical_preprocessor.pkl')
loaded_text_vectorizer = joblib.load('text_vectorizer.pkl')

# 4. Prediction & Deployment Module (FastAPI)

app = FastAPI()

@app.post("/predict")
async def predict_readmission(patient_data: PatientData):
    # Convert incoming data to DataFrame
    input_df = pd.DataFrame([patient_data.dict()])

    # Preprocess numerical and categorical features using the fitted preprocessor
    numerical_categorical_features = loaded_preprocessor.transform(input_df)

    # Preprocess text data using the fitted text vectorizer
    text_features, _ = preprocess_text_data(input_df[text_feature], vectorizer=loaded_text_vectorizer, fit_vectorizer=False)

    # Combine all features
    processed_input = hstack([numerical_categorical_features, text_features])

    # Make prediction
    prediction = loaded_model.predict_proba(processed_input)[:, 1][0] # Probability of readmission

    return {"readmission_probability": float(prediction)}

# To run the FastAPI application, save this code as a Python file (e.g., main.py)
# and execute 'uvicorn main:app --reload' in your terminal.
