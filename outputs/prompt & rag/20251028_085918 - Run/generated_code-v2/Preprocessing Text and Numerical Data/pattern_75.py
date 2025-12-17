import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import uvicorn

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

# --- 1. Data Ingestion (Simulated) ---
def load_simulated_data():
    data = {
        'patient_id': range(1, 101),
        'age': np.random.randint(20, 90, 100),
        'gender': np.random.choice(['Male', 'Female'], 100),
        'admission_type': np.random.choice(['Emergency', 'Elective', 'Urgent'], 100),
        'length_of_stay': np.random.randint(1, 30, 100),
        'lab_result_glucose': np.random.normal(90, 15, 100),
        'medication_count': np.random.randint(1, 10, 100),
        'diagnosis_text': [
            "Patient presented with severe chest pain and shortness of breath. History of hypertension.",
            "Routine check-up, no major complaints. Minor fever. Discharged after observation.",
            "Diabetic ketoacidosis. Admitted for insulin therapy and fluid management.",
            "Fractured tibia from fall. Surgical intervention required. Post-op recovery.",
            "Chronic obstructive pulmonary disease exacerbation. Oxygen therapy initiated.",
        ] * 20, # Repeat for 100 entries
        'discharge_summary': [
            "Patient discharged stable. Follow-up with cardiologist in 2 weeks. Advised medication adherence.",
            "Discharged home. No further concerns. Routine follow-up in 3 months.",
            "Condition improved, diabetes well-controlled. Follow-up with endocrinologist.",
            "Surgery successful, discharged with physical therapy instructions.",
            "Respiratory status improved. Discharged with home oxygen. Pulmonology follow-up.",
        ] * 20,
        'readmitted_30_days': np.random.choice([0, 1], 100, p=[0.8, 0.2]) # Target variable
    }
    df = pd.DataFrame(data)
    # Introduce some missing values for demonstration
    df.loc[df.sample(frac=0.05).index, 'lab_result_glucose'] = np.nan
    df.loc[df.sample(frac=0.03).index, 'length_of_stay'] = np.nan
    df.loc[df.sample(frac=0.02).index, 'gender'] = np.nan
    return df

# --- 2. Data Preprocessing Layer ---

# Text Preprocessing Utilities
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text) # Remove punctuation and numbers
    return text

def tokenize_and_lemmatize(text):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# --- Pipeline Setup ---

numerical_features = ['age', 'length_of_stay', 'lab_result_glucose', 'medication_count']
categorical_features = ['gender', 'admission_type']
text_features = ['diagnosis_text', 'discharge_summary']

# Numerical pipeline
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Categorical pipeline
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Text pipeline
text_transformer = Pipeline(steps=[
    ('clean_lemmatize', Pipeline(steps=[
        ('apply_clean', ColumnTransformer(transformers=[('text_clean', 'passthrough', [0])], remainder='drop', verbose_feature_names_out=False)),
        ('apply_lemmatize', ColumnTransformer(transformers=[('text_lemmatize', 'passthrough', [0])], remainder='drop', verbose_feature_names_out=False)) # This is a conceptual pipeline step, actual function application happens before TFIDF
    ], verbose_feature_names_out=False)),
    ('tfidf', TfidfVectorizer())
])

# Combine all preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features),
        ('text_diag', text_transformer, 'diagnosis_text'),
        ('text_disch', text_transformer, 'discharge_summary')
    ], 
    remainder='drop',
    verbose_feature_names_out=False
)

# --- 3. Feature Engineering (Simplified) ---
def feature_engineer(df):
    df['age_group'] = pd.cut(df['age'], bins=[0, 18, 45, 65, 90], labels=['Child', 'Adult', 'Senior', 'Elderly'], right=False)
    df['has_diabetes_keyword'] = df['diagnosis_text'].apply(lambda x: 1 if 'diabet' in x.lower() else 0)
    return df

# --- 4. Model Training Layer ---
def train_readmission_model(df):
    # Apply text cleaning and lemmatization before passing to preprocessor
    df['diagnosis_text'] = df['diagnosis_text'].apply(lambda x: tokenize_and_lemmatize(clean_text(x)))
    df['discharge_summary'] = df['discharge_summary'].apply(lambda x: tokenize_and_lemmatize(clean_text(x)))

    df_fe = feature_engineer(df.copy())
    
    # Add engineered categorical features to the list of categorical features for the preprocessor
    # Note: 'age_group' and 'has_diabetes_keyword' need to be added to the column transformer dynamically 
    # or the column transformer redefined if added after initial definition.
    # For simplicity, we will train a separate model or combine features manually for this example after preprocessing
    # Or, modify the preprocessor to include these engineered features as new categorical/numerical features.
    # Let's adjust the preprocessor to include 'age_group' and 'has_diabetes_keyword' by adding them to the column lists.
    # This assumes feature_engineer is run *before* the preprocessor is fitted/transformed.

    # Redefine categorical features to include engineered ones
    global categorical_features 
    categorical_features_with_engineered = categorical_features + ['age_group']
    numerical_features_with_engineered = numerical_features + ['has_diabetes_keyword'] # Treat as numerical (0/1)

    # Recreate preprocessor with updated feature lists, assuming new features are in the dataframe before fit/transform
    updated_preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features_with_engineered),
            ('cat', categorical_transformer, categorical_features_with_engineered),
            ('text_diag', text_transformer, 'diagnosis_text'),
            ('text_disch', text_transformer, 'discharge_summary')
        ], 
        remainder='drop',
        verbose_feature_names_out=False
    )

    X = df_fe.drop('readmitted_30_days', axis=1)
    y = df_fe['readmitted_30_days']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Create a full pipeline that includes preprocessing and the model
    model_pipeline = Pipeline(steps=[
        ('preprocessor', updated_preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    model_pipeline.fit(X_train, y_train)
    return model_pipeline, X_test, y_test

# --- 5. Model Evaluation Layer ---
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"Model Evaluation:\n")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1, "roc_auc": roc_auc}

# --- 6. Prediction and Deployment Layer (FastAPI) ---

app = FastAPI()

class PatientData(BaseModel):
    patient_id: int
    age: int
    gender: str
    admission_type: str
    length_of_stay: float = None # Allow None for missing values
    lab_result_glucose: float = None # Allow None for missing values
    medication_count: int
    diagnosis_text: str
    discharge_summary: str

# Load the trained model and preprocessor
# These will be loaded once when the application starts
model_path = "readmission_model.joblib"
readmission_model = None

@app.on_event("startup")
async def load_model():
    global readmission_model
    try:
        readmission_model = joblib.load(model_path)
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Model file not found. Please run the training script first.")
        # Optionally, train a dummy model or exit if model is crucial for startup
        # For this example, we'll let it proceed and handle errors in predict endpoint

@app.post("/predict-readmission/")
async def predict_readmission(data: PatientData):
    if readmission_model is None:
        return {"error": "Model not loaded. Please ensure the model is trained and saved."}
    
    # Convert incoming data to DataFrame for preprocessing
    input_df = pd.DataFrame([data.model_dump()])

    # Apply the same preprocessing steps as training
    input_df['diagnosis_text'] = input_df['diagnosis_text'].apply(lambda x: tokenize_and_lemmatize(clean_text(x)))
    input_df['discharge_summary'] = input_df['discharge_summary'].apply(lambda x: tokenize_and_lemmatize(clean_text(x)))
    input_df = feature_engineer(input_df.copy())

    # Predict
    prediction_proba = readmission_model.predict_proba(input_df)[:, 1]
    prediction = int(prediction_proba > 0.5) # Binary prediction based on threshold

    return {
        "patient_id": data.patient_id,
        "predicted_readmission_probability": float(prediction_proba[0]),
        "predicted_readmission": prediction,
        "message": "Patient is likely to be readmitted" if prediction == 1 else "Patient is unlikely to be readmitted"
    }

# Main execution for training and saving the model
if __name__ == "__main__":
    print("Starting data loading and model training...")
    df = load_simulated_data()
    model, X_test, y_test = train_readmission_model(df)
    joblib.dump(model, model_path)
    print(f"Model trained and saved to {model_path}")
    evaluate_model(model, X_test, y_test)
    print("\nTo run the FastAPI application, save this code as a Python file (e.g., app.py) and run:\n")
    print("  uvicorn app:app --reload")
    print("\nThen navigate to http://127.0.0.1:8000/docs for the API documentation.")
