import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Download necessary NLTK data (run once)
try:
    nltk.data.find("corpora/stopwords")
except nltk.downloader.DownloadError:
    nltk.download("stopwords")
try:
    nltk.data.find("corpora/wordnet")
except nltk.downloader.DownloadError:
    nltk.download("wordnet")

# Sample Data Generation (simulating EHR)
def generate_ehr_data(num_samples=1000):
    np.random.seed(42)
    data = {
        "patient_id": range(num_samples),
        "age": np.random.randint(20, 90, num_samples),
        "num_previous_admissions": np.random.randint(0, 5, num_samples),
        "lab_result_a": np.random.normal(100, 15, num_samples),
        "lab_result_b": np.random.normal(50, 10, num_samples),
        "vital_sign_bp_sys": np.random.normal(120, 10, num_samples),
        "vital_sign_bp_dia": np.random.normal(80, 8, num_samples),
        "doctor_notes": [
            "Patient presented with severe chest pain. Admitted for observation.",
            "Routine check-up. No significant findings.",
            "Follow-up after surgery. Recovery progressing well.",
            "Diabetic patient with high blood sugar. Diet counseling provided.",
            "Emergency admission due to fall. Minor injuries.",
            "Chronic condition management. Medications adjusted.",
            "Discharged home. Stable condition.",
            "Referred to specialist for further evaluation."
        ] * (num_samples // 8) + ["Other note."] * (num_samples % 8),
        "medication_description": [
            "Prescribed ibuprofen and antibiotics.",
            "Daily insulin and metformin.",
            "Blood pressure medication and statins.",
            "Pain relief and physical therapy.",
            "No new medications."
        ] * (num_samples // 5) + ["Other medication."] * (num_samples % 5),
        "readmitted": np.random.randint(0, 2, num_samples) # Target variable (0 or 1)
    }

    # Introduce some missing values
    for col in ["lab_result_a", "vital_sign_bp_sys", "num_previous_admissions"]:
        missing_indices = np.random.choice(num_samples, int(num_samples * 0.05), replace=False)
        data[col][missing_indices] = np.nan

    df = pd.DataFrame(data)
    return df

df = generate_ehr_data(num_samples=1000)

# Separate features (X) and target (y)
X = df.drop("readmitted", axis=1)
y = df["readmitted"]

# Define numerical and text features
numerical_features = ["age", "num_previous_admissions", "lab_result_a", "lab_result_b", "vital_sign_bp_sys", "vital_sign_bp_dia"]
text_features = ["doctor_notes", "medication_description"]

# Custom text preprocessing function
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def text_preprocessor(text):
    text = str(text).lower()  # Convert to string and lowercase
    tokens = nltk.word_tokenize(text) # Tokenize
    tokens = [word for word in tokens if word.isalpha()] # Remove non-alphabetic tokens
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words] # Remove stop words and lemmatize
    return " ".join(tokens)

# Create preprocessing pipelines
numerical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

text_pipeline = Pipeline(steps=[
    ("tfidf", TfidfVectorizer(preprocessor=text_preprocessor))
])

# Combine numerical and text preprocessing using ColumnTransformer
# We will apply TF-IDF to each text column separately and then concatenate them.
# This requires a slightly more complex setup for ColumnTransformer to handle multiple TFIDF outputs.
# For simplicity, let's create a single text feature by concatenating text columns first, 
# or handle them individually and then merge. Let's merge for simplicity here.

# For this example, let's process text features individually and then combine them later
# in the overall pipeline if needed, or create a single aggregated text column first.
# A common approach is to create a combined text column before vectorization if the model expects one.

X["combined_text"] = X[text_features].apply(lambda x: " ".join(x.dropna().astype(str)), axis=1)
text_feature_to_process = "combined_text"

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, numerical_features),
        ("text", text_pipeline, text_feature_to_process)
    ], 
    remainder="drop" # Drop other columns not specified (like patient_id)
)

# Create the full pipeline
model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train the model
print("Training the model...")
model_pipeline.fit(X_train, y_train)
print("Model training complete.")

# Make predictions
y_pred = model_pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(report)

print("\nPreprocessing and prediction pipeline successfully demonstrated for EHR data.")