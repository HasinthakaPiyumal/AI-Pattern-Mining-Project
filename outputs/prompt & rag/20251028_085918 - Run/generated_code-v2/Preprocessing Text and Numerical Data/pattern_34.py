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
from sklearn.metrics import accuracy_score, classification_report

# Ensure NLTK data is downloaded
try:
    nltk.data.find("corpora/stopwords")
except nltk.downloader.DownloadError:
    nltk.download("stopwords", quiet=True)
try:
    nltk.data.find("corpora/wordnet")
except nltk.downloader.DownloadError:
    nltk.download("wordnet", quiet=True)

# Initialize NLTK components
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def generate_ehr_data(num_samples=1000):
    np.random.seed(42)
    data = {
        "patient_id": np.arange(num_samples),
        "age": np.random.randint(20, 90, num_samples),
        "gender": np.random.choice(["Male", "Female"], num_samples),
        "lab_result_a": np.random.normal(100, 15, num_samples),
        "lab_result_b": np.random.normal(50, 10, num_samples),
        "num_diagnoses": np.random.randint(1, 10, num_samples),
        "medication_count": np.random.randint(1, 15, num_samples),
        "admission_type": np.random.choice(["Emergency", "Elective", "Urgent"], num_samples),
        "doctor_notes": [
            "Patient presented with severe chest pain. Stabilized and discharged with medication. Follow-up recommended.",
            "Complains of persistent cough and fever. Diagnosed with pneumonia. Prescribed antibiotics.",
            "Routine check-up, no significant findings. Advised lifestyle changes for blood pressure.",
            "Fractured arm after a fall. Surgery performed. Physiotherapy scheduled.",
            "Diabetic patient with high blood sugar. Insulin dosage adjusted. Dietary advice given.",
            "Mild headache, given painkillers. Sent home. No further issues reported.",
            "Emergency appendectomy performed. Recovering well. Discharge in 3 days.",
            "Chronic back pain. Referred to specialist. Pain management plan initiated.",
            "Fever and body aches. Suspected viral infection. Rest and fluids.",
            "Patient stable. Discharged.",
            "Heart palpitations. ECG normal. Stress management techniques discussed.",
            "Blood pressure spike. Medication review. Dietician consultation.",
            "Minor burn, dressed and discharged. Advised to keep area clean.",
            "Severe allergic reaction. Treated with antihistamines. Identified allergen.",
            "Post-operative care. Wound healing progressing well. Follow-up.",
            "Depression screening. Referred to mental health services. Medication started.",
            "Asthma exacerbation. Nebulizer treatment given. Inhaler technique reviewed.",
            "Abdominal pain, ruled out serious conditions. Prescribed antacids.",
            "Routine vaccination. No adverse effects.",
            "Migraine episode. Given acute treatment. Lifestyle modifications suggested."
        ] * (num_samples // 20 + 1)[:num_samples], 
        "discharge_summary": [
            "Discharged home, stable. Follow-up arranged.",
            "Condition improved. Released with instructions for home care.",
            "Patient recovered well. No complications.",
            "Home with family. Detailed care plan provided.",
            "Good prognosis. Next appointment scheduled.",
            "Discharged to rehabilitation facility.",
            "Patient declined further treatment. Released against medical advice.",
            "Transferred to long-term care.",
            "Fully recovered. Self-care instructions given.",
            "Discharged without significant concerns."
        ] * (num_samples // 10 + 1)[:num_samples],
        "readmitted_30_days": np.random.choice([0, 1], num_samples, p=[0.85, 0.15])
    }

    df = pd.DataFrame(data)

    for col in ["lab_result_a", "lab_result_b", "medication_count"]:
        missing_indices = np.random.choice(num_samples, int(num_samples * 0.05), replace=False)
        df.loc[missing_indices, col] = np.nan

    for i in range(50):
        idx = np.random.randint(0, num_samples)
        original_note = df.loc[idx, "doctor_notes"]
        if "chest pain" in original_note:
            df.loc[idx, "doctor_notes"] = original_note.replace("chest pain", "cp")
        elif "fever" in original_note:
            df.loc[idx, "doctor_notes"] = original_note + " temp elevated."

    return df

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

if __name__ == "__main__":
    print("Generating synthetic EHR data...")
    df = generate_ehr_data(num_samples=2000)
    print("Data generated. Shape:", df.shape)
    print("\nSample Data Head:")
    print(df.head())

    X = df.drop("readmitted_30_days", axis=1)
    y = df["readmitted_30_days"]

    numerical_features = ["age", "lab_result_a", "lab_result_b", "num_diagnoses", "medication_count"]
    categorical_features = ["gender", "admission_type"]
    text_features = ["doctor_notes", "discharge_summary"]

    print("\nApplying text cleaning...")
    for col in text_features:
        X[col + "_processed"] = X[col].apply(preprocess_text)
    
    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    text_transformer_doctor_notes = TfidfVectorizer(max_features=1000)
    text_transformer_discharge_summary = TfidfVectorizer(max_features=1000)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
            ("text_doc", text_transformer_doctor_notes, "doctor_notes_processed"),
            ("text_sum", text_transformer_discharge_summary, "discharge_summary_processed")
        ],
        remainder="drop"
    )

    print("Setting up ML pipeline...")
    model_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\nTraining data shape: {X_train.shape}, Test data shape: {X_test.shape}")

    print("Training the model...")
    model_pipeline.fit(X_train, y_train)
    print("Model training complete.")

    print("Making predictions on the test set...")
    y_pred = model_pipeline.predict(X_test)

    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nPreprocessing and Model Training Complete for Patient Readmission Prediction System.")
    print("This script demonstrates data generation, numerical/text preprocessing, feature engineering, and a basic ML pipeline.")