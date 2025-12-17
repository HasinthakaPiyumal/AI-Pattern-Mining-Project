import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download NLTK data (if not already downloaded)
try:
    nltk.data.find("tokenizers/punkt")
except nltk.downloader.DownloadError:
    nltk.download("punkt")
try:
    nltk.data.find("corpora/stopwords")
except nltk.downloader.DownloadError:
    nltk.download("stopwords")
try:
    nltk.data.find("corpora/wordnet")
except nltk.downloader.DownloadError:
    nltk.download("wordnet")

# --- 1. Generate Dummy Data ---
np.random.seed(42)
data_size = 1000

dummy_data = {
    "patient_id": range(data_size),
    "age": np.random.randint(20, 80, data_size),
    "weight": np.random.normal(70, 15, data_size),
    "height": np.random.normal(170, 10, data_size),
    "blood_pressure_sys": np.random.randint(90, 180, data_size),
    "blood_pressure_dia": np.random.randint(60, 120, data_size),
    "cholesterol": np.random.normal(200, 40, data_size),
    "glucose": np.random.normal(100, 20, data_size),
    "doctor_notes": [
        "Patient reports mild fatigue and occasional headache. No other significant complaints.",
        "Follow-up for hypertension. Medication adjusted. Blood pressure stable.",
        "Routine check-up. Patient in good health. Advised on diet.",
        "Symptoms of flu-like illness. Prescribed antiviral medication.",
        "Diabetic patient. Glucose levels elevated. Diet counseling provided.",
        "Chest pain reported. Referred to cardiology for further evaluation.",
        "Allergic reaction to new medication. Switched to alternative.",
        "Knee pain after exercise. Recommended physical therapy.",
        "Anxiety disorder. Discussed coping mechanisms. Follow-up in 2 weeks.",
        "Possible bacterial infection. Sent for lab tests. Awaiting results.",
    ] * (data_size // 10),
    "medical_history": [
        "No significant medical history.",
        "History of asthma since childhood.",
        "Previous heart attack 5 years ago.",
        "Family history of diabetes.",
        "Undergone appendectomy in 2010.",
        "Chronic migraines.",
        "Diagnosed with irritable bowel syndrome.",
        "Seasonal allergies.",
        "History of anxiety and depression.",
        "No major surgeries or chronic conditions.",
    ] * (data_size // 10),
    "disease_target": np.random.randint(0, 2, data_size) # 0: No Disease, 1: Disease Present
}

df = pd.DataFrame(dummy_data)

# Introduce some missing values for demonstration
for col in ['weight', 'cholesterol', 'doctor_notes']:
    df.loc[df.sample(frac=0.05).index, col] = np.nan

print("Original DataFrame head:")
print(df.head())
print("\nMissing values before preprocessing:")
print(df.isnull().sum())

# --- 2. Define Preprocessing Steps ---

# Numerical Preprocessing Pipeline
numerical_features = ['age', 'weight', 'height', 'blood_pressure_sys', 'blood_pressure_dia', 'cholesterol', 'glucose']
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Text Preprocessing Function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(tokens)

# --- 3. Create ColumnTransformer for combined preprocessing ---
text_features = ['doctor_notes', 'medical_history']

# Apply custom text preprocessing function before TF-IDF
df['doctor_notes_preprocessed'] = df['doctor_notes'].apply(preprocess_text)
df['medical_history_preprocessed'] = df['medical_history'].apply(preprocess_text)

# TF-IDF Vectorization for preprocessed text
text_transformer = TfidfVectorizer(max_features=100) # Limit features for simplicity

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('doc_notes_tfidf', text_transformer, 'doctor_notes_preprocessed'),
        ('med_hist_tfidf', text_transformer, 'medical_history_preprocessed')
    ], 
    remainder='passthrough' # Keep other columns if any, like patient_id (though we'll drop it)
)

# --- 4. Build the full ML Pipeline ---
X = df.drop(['disease_target', 'patient_id', 'doctor_notes', 'medical_history'], axis=1)
y = df['disease_target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# The final pipeline includes preprocessing and the model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, solver='liblinear'))
])

# --- 5. Train the Model ---
print("\nTraining the model...")
model_pipeline.fit(X_train, y_train)
print("Model training complete.")

# --- 6. Evaluate the Model ---
y_pred = model_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy on test set: {accuracy:.4f}")

# --- Example of Preprocessing an unseen data point ---
print("\n--- Demonstrating Preprocessing on a new data point ---")
new_patient_data = pd.DataFrame({
    "age": [65],
    "weight": [75.0],
    "height": [168.0],
    "blood_pressure_sys": [150],
    "blood_pressure_dia": [95],
    "cholesterol": [230.0],
    "glucose": [130.0],
    "doctor_notes": ["Patient complains of persistent cough and mild fever. History of smoking."],
    "medical_history": ["Diagnosed with chronic bronchitis 3 years ago."]
})

# Apply the same text preprocessing function manually for demonstration
new_patient_data['doctor_notes_preprocessed'] = new_patient_data['doctor_notes'].apply(preprocess_text)
new_patient_data['medical_history_preprocessed'] = new_patient_data['medical_history'].apply(preprocess_text)

# Select only the features the model expects (matching X_train columns)
new_patient_features = new_patient_data[X_train.drop(['doctor_notes_preprocessed', 'medical_history_preprocessed'], axis=1).columns.tolist() +
                                        ['doctor_notes_preprocessed', 'medical_history_preprocessed']]

# Predict using the trained pipeline
prediction = model_pipeline.predict(new_patient_features)
prediction_proba = model_pipeline.predict_proba(new_patient_features)

print(f"\nPrediction for new patient: {'Disease Present' if prediction[0] == 1 else 'No Disease'}")
print(f"Prediction probabilities: {prediction_proba[0]}")