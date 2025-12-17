import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.sparse import hstack

# Download NLTK resources if not already downloaded
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

# 1. Data Loading (Simulated)
# Create a dummy dataset for demonstration
data = {
    'patient_id': range(1, 101),
    'age': np.random.randint(20, 80, 100),
    'blood_pressure_systolic': np.random.randint(90, 180, 100),
    'cholesterol_ldl': np.random.randint(70, 200, 100),
    'glucose': np.random.randint(70, 250, 100),
    'doctor_notes': [
        "Patient presents with mild fever and cough. No significant findings.",
        "Admitted due to severe chest pain. History of cardiac issues.",
        "Routine check-up. Reports feeling tired frequently.",
        "Follow-up for diabetes. Blood sugar levels are stable.",
        "Emergency visit, severe headache and blurred vision. Suspected stroke."
    ] * 20, # Repeat to get 100 entries
    'patient_symptoms': [
        "fever, cough",
        "chest pain, shortness of breath",
        "fatigue",
        "none",
        "headache, blurred vision, dizziness"
    ] * 20,
    'risk_label': np.random.randint(0, 2, 100) # 0 for low risk, 1 for high risk
}

df = pd.DataFrame(data)

# Introduce some missing values for demonstration
for col in ['age', 'blood_pressure_systolic', 'cholesterol_ldl', 'glucose']:
    df.loc[df.sample(frac=0.1).index, col] = np.nan
df.loc[df.sample(frac=0.05).index, 'doctor_notes'] = np.nan

print("Original DataFrame head:")
print(df.head())
print("\nMissing values before imputation:")
print(df.isnull().sum())

# Separate features and target
X = df.drop('risk_label', axis=1)
y = df['risk_label']

# Identify numerical and text columns
numerical_cols = ['age', 'blood_pressure_systolic', 'cholesterol_ldl', 'glucose']
text_cols = ['doctor_notes', 'patient_symptoms']

# 2. Numerical Data Preprocessing Module
print("\n--- Numerical Data Preprocessing ---")
# Imputation
imputer = SimpleImputer(strategy='mean') # Using mean strategy
X_numerical_imputed = pd.DataFrame(imputer.fit_transform(X[numerical_cols]), columns=numerical_cols, index=X.index)

# Scaling
scaler = StandardScaler()
X_numerical_scaled = pd.DataFrame(scaler.fit_transform(X_numerical_imputed), columns=numerical_cols, index=X.index)

print("Numerical data after imputation and scaling (first 5 rows):\n", X_numerical_scaled.head())

# 3. Text Data Preprocessing Module
print("\n--- Text Data Preprocessing ---")
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if pd.isna(text): # Handle NaN values
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text) # Remove punctuation and numbers
    return text

def preprocess_text(text):
    text = clean_text(text)
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    lemmas = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    return ' '.join(lemmas)

# Apply text preprocessing
X_text_processed = pd.DataFrame(index=X.index)
for col in text_cols:
    X_text_processed[col + '_processed'] = X[col].apply(preprocess_text)

# Combine preprocessed text into a single column for TF-IDF
X_text_combined = X_text_processed['doctor_notes_processed'] + " " + X_text_processed['patient_symptoms_processed']

# Text Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=1000) # Limit features for demonstration
X_text_vectorized = tfidf_vectorizer.fit_transform(X_text_combined)

print("Text data after preprocessing and vectorization (shape):", X_text_vectorized.shape)

# 4. Feature Concatenation
print("\n--- Feature Concatenation ---")
# Convert numerical features to sparse matrix to concatenate with TF-IDF output
X_numerical_sparse = X_numerical_scaled.sparse.to_coo() if pd.api.types.is_sparse(X_numerical_scaled) else X_numerical_scaled.values
X_combined = hstack([X_numerical_sparse, X_text_vectorized])

print("Combined feature matrix shape:", X_combined.shape)

# 5. Model Training
print("\n--- Model Training ---")
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Model training complete.")

# 6. Model Evaluation
print("\n--- Model Evaluation ---")
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")