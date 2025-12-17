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
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import xgboost as xgb
import joblib

# Download NLTK data (if not already downloaded)
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except nltk.downloader.DownloadError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/omw-1.4')
except nltk.downloader.DownloadError:
    nltk.download('omw-1.4')

# 1. Data Ingestion Layer - Simulate data loading
def load_data():
    data = {
        "patient_id": range(1, 101),
        "age": np.random.randint(20, 90, 100),
        "gender": np.random.choice(["Male", "Female"], 100),
        "admission_type": np.random.choice(["Emergency", "Elective", "Urgent", "Unknown"], 100),
        "length_of_stay_days": np.random.randint(1, 30, 100),
        "num_diagnoses": np.random.randint(1, 10, 100),
        "medication_count": np.random.randint(1, 25, 100),
        "has_chronic_disease": np.random.choice([0, 1], 100, p=[0.4, 0.6]),
        "discharge_summary": [
            "Patient presented with chest pain. Diagnosed with pneumonia. Discharged after 5 days. Follow-up recommended.",
            "Routine check-up. No significant findings. Patient healthy.",
            "Admitted for surgery. Post-op recovery was good. Medications adjusted. No complications.",
            "High fever and cough. Viral infection suspected. Improved with rest and fluids.",
            "Emergency admission for fracture. Surgery performed successfully. Physical therapy advised.",
            "Chronic diabetes management. Medication review. Stable condition.",
            "Patient complained of headache. MRI negative. Symptomatic treatment given.",
            "Heart failure exacerbation. Diuretics administered. Education on diet given.",
            "Asthma attack. Nebulizer treatment. Discharged with inhaler.",
            "Routine follow-up for hypertension. Blood pressure controlled. New medication started.",
        ] * 10, # Repeat for 100 entries
        "target_readmission": np.random.choice([0, 1], 100, p=[0.8, 0.2]) # 0: No Readmission, 1: Readmission
    }
    df = pd.DataFrame(data)
    
    # Introduce some missing values for demonstration
    df.loc[df.sample(frac=0.05).index, "age"] = np.nan
    df.loc[df.sample(frac=0.03).index, "discharge_summary"] = np.nan
    df.loc[df.sample(frac=0.02).index, "admission_type"] = np.nan
    
    return df

# Text Preprocessing Helper Function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Remove numbers and punctuation
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# 2. Data Preprocessing Layer
def create_preprocessor(numerical_features, categorical_features, text_features):
    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    
    text_transformer = Pipeline(steps=[
        ("tfidf", TfidfVectorizer(max_features=100))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
            ("text", text_transformer, text_features)
        ], 
        remainder="passthrough"
    )
    return preprocessor

# 3. Feature Engineering Layer
def engineer_features(df):
    df["length_of_stay_risk"] = df["length_of_stay_days"] / df["age"]
    return df

# Main script execution
if __name__ == "__main__":
    # Load Data
    df = load_data()
    print("Initial Data Head:")
    print(df.head())
    print("\nInitial Data Info:")
    df.info()

    # Apply text preprocessing
    df["processed_discharge_summary"] = df["discharge_summary"].apply(preprocess_text)
    print("\nData after text preprocessing:")
    print(df[["discharge_summary", "processed_discharge_summary"]].head())

    # Engineer Features
    df = engineer_features(df)
    print("\nData after feature engineering:")
    print(df[["length_of_stay_days", "age", "length_of_stay_risk"]].head())

    # Define features and target
    numerical_features = ["age", "length_of_stay_days", "num_diagnoses", "medication_count", "has_chronic_disease", "length_of_stay_risk"]
    categorical_features = ["gender", "admission_type"]
    text_features = "processed_discharge_summary"
    target = "target_readmission"

    X = df.drop(columns=[target, "patient_id", "discharge_summary"])
    y = df[target]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Create and fit preprocessing pipeline
    preprocessor = create_preprocessor(numerical_features, categorical_features, [text_features])
    
    # 4. Model Training Layer
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", use_label_encoder=False, random_state=42))
    ])

    print("\nTraining Model...")
    model.fit(X_train, y_train)
    print("Model Training Complete.")

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    print(f"Model ROC AUC: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 5. Model Persistence Layer
    model_filename = "patient_readmission_model.joblib"
    joblib.dump(model, model_filename)
    print(f"\nModel saved as {model_filename}")

    # 6. Prediction/Inference Layer - Simulate new data for prediction
    print("\nSimulating new patient data for inference...")
    new_patient_data = pd.DataFrame({
        "patient_id": [101, 102],
        "age": [75, 45],
        "gender": ["Male", "Female"],
        "admission_type": ["Emergency", "Elective"],
        "length_of_stay_days": [12, 3],
        "num_diagnoses": [7, 2],
        "medication_count": [15, 4],
        "has_chronic_disease": [1, 0],
        "discharge_summary": [
            "Elderly patient with multiple comorbidities. Required extended stay due to complications. High risk for readmission.",
            "Young patient for routine check-up. Discharged quickly. No major health concerns."
        ]
    })
    
    # Preprocess new data using the same pipeline
    new_patient_data["processed_discharge_summary"] = new_patient_data["discharge_summary"].apply(preprocess_text)
    new_patient_data_engineered = engineer_features(new_patient_data.copy())
    X_new = new_patient_data_engineered.drop(columns=["patient_id", "discharge_summary"])

    # Load the trained model and make predictions
    loaded_model = joblib.load(model_filename)
    new_predictions = loaded_model.predict(X_new)
    new_probabilities = loaded_model.predict_proba(X_new)[:, 1]

    print("\nPredictions for new patients:")
    for i, pred in enumerate(new_predictions):
        status = "Readmitted" if pred == 1 else "Not Readmitted"
        print(f"Patient ID: {new_patient_data.loc[i, 'patient_id']}, Predicted Status: {status}, Probability of Readmission: {new_probabilities[i]:.4f}")

