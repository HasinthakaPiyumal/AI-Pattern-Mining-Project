import pandas as pd
import numpy as np
import spacy
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

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

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'. This may take a few minutes...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

class NumericalPreprocessor:
    def __init__(self):
        self.imputer = SimpleImputer(strategy='mean')
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.fitted_num_cols = None
        self.fitted_cat_cols = None

    def fit(self, X_num, X_cat):
        if not X_num.empty:
            self.imputer.fit(X_num)
            self.scaler.fit(self.imputer.transform(X_num))
            self.fitted_num_cols = X_num.columns
        if not X_cat.empty:
            self.encoder.fit(X_cat)
            self.fitted_cat_cols = X_cat.columns

    def transform(self, X_num, X_cat):
        processed_num = pd.DataFrame()
        processed_cat = pd.DataFrame()

        if self.fitted_num_cols is not None and not X_num.empty:
            imputed_data = self.imputer.transform(X_num[self.fitted_num_cols])
            scaled_data = self.scaler.transform(imputed_data)
            processed_num = pd.DataFrame(scaled_data, columns=self.fitted_num_cols, index=X_num.index)
        elif not X_num.empty:
             raise ValueError("NumericalPreprocessor not fitted for numerical data.")

        if self.fitted_cat_cols is not None and not X_cat.empty:
            encoded_data = self.encoder.transform(X_cat[self.fitted_cat_cols])
            encoded_col_names = self.encoder.get_feature_names_out(self.fitted_cat_cols)
            processed_cat = pd.DataFrame(encoded_data, columns=encoded_col_names, index=X_cat.index)
        elif not X_cat.empty:
             raise ValueError("NumericalPreprocessor not fitted for categorical data.")

        return processed_num, processed_cat

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer(max_features=5000)

    def _preprocess_text_single(self, text):
        if not isinstance(text, str): # Handle non-string inputs like NaN
            return ""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        doc = nlp(text)
        lemmas = [token.lemma_ for token in doc if token.text not in self.stop_words and token.is_alpha]
        return " ".join(lemmas)

    def fit_transform(self, texts):
        processed_texts = texts.apply(self._preprocess_text_single)
        tfidf_matrix = self.vectorizer.fit_transform(processed_texts)
        return tfidf_matrix, self.vectorizer.get_feature_names_out()

    def transform(self, texts):
        processed_texts = texts.apply(self._preprocess_text_single)
        tfidf_matrix = self.vectorizer.transform(processed_texts)
        return tfidf_matrix, self.vectorizer.get_feature_names_out()

class FeatureEngineer:
    def engineer_features(self, df):
        df_copy = df.copy()
        if 'height_cm' in df_copy.columns and 'weight_kg' in df_copy.columns:
            df_copy['bmi'] = df_copy['weight_kg'] / ((df_copy['height_cm'] / 100)**2)
        if 'age' in df_copy.columns:
            bins = [0, 18, 35, 60, 100]
            labels = ['child', 'young_adult', 'adult', 'senior']
            df_copy['age_group'] = pd.cut(df_copy['age'], bins=bins, labels=labels, right=False)
        return df_copy

class ModelTrainer:
    def __init__(self):
        self.model = RandomForestClassifier(random_state=42)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

# --- Main Workflow Demonstration ---

# 1. Simulate Data Ingestion
data = {
    'patient_id': range(100),
    'age': np.random.randint(10, 80, 100),
    'gender': np.random.choice(['Male', 'Female'], 100),
    'height_cm': np.random.normal(170, 10, 100),
    'weight_kg': np.random.normal(70, 15, 100),
    'blood_type': np.random.choice(['A+', 'B+', 'O+', 'AB-'], 100),
    'lab_result_a': np.random.normal(100, 10, 100),
    'lab_result_b': np.random.normal(50, 5, 100),
    'medical_history': [
        'Patient has a history of hypertension and diabetes.',
        'No significant medical history, routine check-up.',
        'Mild asthma, controlled with medication.',
        'Diagnosed with chronic fatigue syndrome last year.',
        'Family history of heart disease.',
        np.nan, # Simulate missing text
    ] * 17 + ['Patient has a history of hypertension and diabetes.'] * 6 + ['No significant medical history, routine check-up.'] * 6,
    'diagnosis_text': [
        'Hypertension, Diabetes Mellitus Type 2.',
        'Healthy patient, no specific diagnosis.',
        'Asthma exacerbation, prescribed inhaler.',
        'Chronic fatigue syndrome.',
        'Atrial fibrillation, on anticoagulants.',
        'Urinary tract infection, treated with antibiotics.',
        np.nan,
    ] * 14 + ['Hypertension, Diabetes Mellitus Type 2.'] * 2,
    'outcome': np.random.choice([0, 1], 100, p=[0.7, 0.3]) # 0: Good, 1: Poor outcome
}
df = pd.DataFrame(data)

# Introduce some missing numerical values
df.loc[np.random.choice(df.index, 10, replace=False), 'height_cm'] = np.nan
df.loc[np.random.choice(df.index, 5, replace=False), 'lab_result_a'] = np.nan

# Define target and features
TARGET = 'outcome'
X = df.drop(columns=[TARGET, 'patient_id'])
y = df[TARGET]

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Data Preprocessing

# Separate numerical and text columns for processing
numerical_cols = ['age', 'height_cm', 'weight_kg', 'lab_result_a', 'lab_result_b']
categorical_cols = ['gender', 'blood_type']
text_cols = ['medical_history', 'diagnosis_text']

X_train_num = X_train[numerical_cols]
X_train_cat = X_train[categorical_cols]
X_train_text_mh = X_train['medical_history']
X_train_text_dt = X_train['diagnosis_text']

X_test_num = X_test[numerical_cols]
X_test_cat = X_test[categorical_cols]
X_test_text_mh = X_test['medical_history']
X_test_text_dt = X_test['diagnosis_text']

# Initialize Preprocessors
num_preprocessor = NumericalPreprocessor()
text_preprocessor_mh = TextPreprocessor()
text_preprocessor_dt = TextPreprocessor()
feature_engineer = FeatureEngineer()

# Fit and Transform Numerical/Categorical Data (Train Set)
num_preprocessor.fit(X_train_num, X_train_cat)
processed_num_train, processed_cat_train = num_preprocessor.transform(X_train_num, X_train_cat)

# Fit and Transform Text Data (Train Set)
tfidf_mh_train, tfidf_mh_features = text_preprocessor_mh.fit_transform(X_train_text_mh.fillna(''))
tfidf_dt_train, tfidf_dt_features = text_preprocessor_dt.fit_transform(X_train_text_dt.fillna(''))

tfidf_mh_df_train = pd.DataFrame(tfidf_mh_train.toarray(), columns=[f'mh_tfidf_{f}' for f in tfidf_mh_features], index=X_train.index)
tfidf_dt_df_train = pd.DataFrame(tfidf_dt_train.toarray(), columns=[f'dt_tfidf_{f}' for f in tfidf_dt_features], index=X_train.index)

# Transform Numerical/Categorical Data (Test Set)
processed_num_test, processed_cat_test = num_preprocessor.transform(X_test_num, X_test_cat)

# Transform Text Data (Test Set)
tfidf_mh_test, _ = text_preprocessor_mh.transform(X_test_text_mh.fillna(''))
tfidf_dt_test, _ = text_preprocessor_dt.transform(X_test_text_dt.fillna(''))

tfidf_mh_df_test = pd.DataFrame(tfidf_mh_test.toarray(), columns=[f'mh_tfidf_{f}' for f in tfidf_mh_features], index=X_test.index)
tfidf_dt_df_test = pd.DataFrame(tfidf_dt_test.toarray(), columns=[f'dt_tfidf_{f}' for f in tfidf_dt_features], index=X_test.index)

# Combine preprocessed data for Feature Engineering
X_train_processed = pd.concat([
    processed_num_train,
    processed_cat_train,
    tfidf_mh_df_train,
    tfidf_dt_df_train
], axis=1)
X_test_processed = pd.concat([
    processed_num_test,
    processed_cat_test,
    tfidf_mh_df_test,
    tfidf_dt_df_test
], axis=1)

# Ensure columns match after concatenation (important for test set if train had different text features)
train_cols = X_train_processed.columns
X_test_processed = X_test_processed.reindex(columns=train_cols, fill_value=0)

# 3. Feature Engineering
X_train_fe = feature_engineer.engineer_features(X_train_processed.copy())
X_test_fe = feature_engineer.engineer_features(X_test_processed.copy())

# Handle the newly created categorical 'age_group' from feature engineering
# It needs to be one-hot encoded and aligned with the main features

# Define a preprocessor for the engineered categorical column
engineered_cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

if 'age_group' in X_train_fe.columns:
    X_train_fe_age_group = engineered_cat_pipeline.fit_transform(X_train_fe[['age_group']])
    engineered_age_group_features = engineered_cat_pipeline.named_steps['onehot'].get_feature_names_out(['age_group'])
    X_train_fe_age_group_df = pd.DataFrame(X_train_fe_age_group, columns=engineered_age_group_features, index=X_train_fe.index)
    X_train_fe = pd.concat([X_train_fe.drop(columns=['age_group']), X_train_fe_age_group_df], axis=1)

    X_test_fe_age_group = engineered_cat_pipeline.transform(X_test_fe[['age_group']])
    X_test_fe_age_group_df = pd.DataFrame(X_test_fe_age_group, columns=engineered_age_group_features, index=X_test_fe.index)
    X_test_fe = pd.concat([X_test_fe.drop(columns=['age_group']), X_test_fe_age_group_df], axis=1)


# Align columns again after feature engineering
final_train_cols = X_train_fe.columns
X_test_fe = X_test_fe.reindex(columns=final_train_cols, fill_value=0)

# 4. Machine Learning Model Training
model_trainer = ModelTrainer()
model_trainer.train(X_train_fe, y_train)

# Evaluate the model (simple accuracy for demonstration)
from sklearn.metrics import accuracy_score
y_pred = model_trainer.predict(X_test_fe)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred)}")

# Save the model and preprocessors
joblib.dump(model_trainer.model, 'trained_model.joblib')
joblib.dump(num_preprocessor, 'num_preprocessor.joblib')
joblib.dump(text_preprocessor_mh, 'text_preprocessor_mh.joblib')
joblib.dump(text_preprocessor_dt, 'text_preprocessor_dt.joblib')
joblib.dump(feature_engineer, 'feature_engineer.joblib')
joblib.dump(engineered_cat_pipeline, 'engineered_cat_pipeline.joblib')

print("Model and preprocessors saved.")

# 5. Model Deployment & Inference Layer (FastAPI - simplified)
# To run this part, save it as a separate file (e.g., 'api.py') and run 'uvicorn api:app --reload'
# For demonstration, this part is commented out as it requires a running server and external libs

# from fastapi import FastAPI
# from pydantic import BaseModel

# app = FastAPI()

# # Load saved components
# loaded_model = joblib.load('trained_model.joblib')
# loaded_num_preprocessor = joblib.load('num_preprocessor.joblib')
# loaded_text_preprocessor_mh = joblib.load('text_preprocessor_mh.joblib')
# loaded_text_preprocessor_dt = joblib.load('text_preprocessor_dt.joblib')
# loaded_feature_engineer = joblib.load('feature_engineer.joblib')
# loaded_engineered_cat_pipeline = joblib.load('engineered_cat_pipeline.joblib')

# class PatientData(BaseModel:
#     age: int
#     gender: str
#     height_cm: float
#     weight_kg: float
#     blood_type: str
#     lab_result_a: float
#     lab_result_b: float
#     medical_history: str
#     diagnosis_text: str

# @app.post("/predict/")
# async def predict_outcome(data: PatientData):
#     input_df = pd.DataFrame([data.dict()])

#     # Preprocessing steps (mirroring training)
#     num_data = input_df[numerical_cols]
#     cat_data = input_df[categorical_cols]
#     text_mh = input_df['medical_history']
#     text_dt = input_df['diagnosis_text']

#     processed_num, processed_cat = loaded_num_preprocessor.transform(num_data, cat_data)

#     tfidf_mh, _ = loaded_text_preprocessor_mh.transform(text_mh.fillna(''))
#     tfidf_dt, _ = loaded_text_preprocessor_dt.transform(text_dt.fillna(''))

#     tfidf_mh_df = pd.DataFrame(tfidf_mh.toarray(), columns=[f'mh_tfidf_{f}' for f in loaded_text_preprocessor_mh.vectorizer.get_feature_names_out()], index=input_df.index)
#     tfidf_dt_df = pd.DataFrame(tfidf_dt.toarray(), columns=[f'dt_tfidf_{f}' for f in loaded_text_preprocessor_dt.vectorizer.get_feature_names_out()], index=input_df.index)

#     combined_processed_data = pd.concat([
#         processed_num,
#         processed_cat,
#         tfidf_mh_df,
#         tfidf_dt_df
#     ], axis=1)

#     # Feature Engineering
#     engineered_data = loaded_feature_engineer.engineer_features(combined_processed_data.copy())

#     # Handle engineered categorical 'age_group'
#     if 'age_group' in engineered_data.columns:
#         engineered_age_group_transformed = loaded_engineered_cat_pipeline.transform(engineered_data[['age_group']])
#         engineered_age_group_features = loaded_engineered_cat_pipeline.named_steps['onehot'].get_feature_names_out(['age_group'])
#         engineered_age_group_df = pd.DataFrame(engineered_age_group_transformed, columns=engineered_age_group_features, index=engineered_data.index)
#         engineered_data = pd.concat([engineered_data.drop(columns=['age_group']), engineered_age_group_df], axis=1)

#     # Ensure column order matches training
#     engineered_data = engineered_data.reindex(columns=final_train_cols, fill_value=0)

#     prediction = loaded_model.predict(engineered_data)[0]
#     prediction_proba = loaded_model.predict_proba(engineered_data)[0].tolist()

#     return {"prediction": int(prediction), "probability": prediction_proba}

# 6. Monitoring & Feedback Layer (Placeholder)
print("\n--- Monitoring & Feedback Placeholder ---")
print("In a real-world scenario, tools like MLflow or Weights & Biases would be used here ")
print("to track experiments, monitor model performance, and detect data drift.")
