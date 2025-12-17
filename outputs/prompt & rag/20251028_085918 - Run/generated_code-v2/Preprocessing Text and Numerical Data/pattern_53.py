import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download necessary NLTK data (run once)
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except nltk.downloader.DownloadError:
    nltk.download('wordnet')

# 1. Simulate Data Ingestion
# Create a sample DataFrame resembling patient medical records
data = {
    'patient_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'age': [34, 56, 78, 23, np.nan, 45, 62, 39, 50, 71],
    'blood_pressure_sys': [120, 145, 160, 110, 130, np.nan, 150, 125, 135, 165],
    'blood_pressure_dia': [80, 90, 95, 70, 85, np.nan, 90, 80, 88, 100],
    'cholesterol': [200, 240, 280, 180, 210, 250, np.nan, 205, 220, 290],
    'gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
    'diagnosis_notes': [
        'Patient reports mild chest pain and fatigue. History of smoking.',
        'Routine check-up, no significant findings. Family history of diabetes.',
        'Severe headache and blurred vision. Possible hypertension.',
        'Minor sports injury, sprained ankle. No underlying conditions.',
        'Follow-up for elevated cholesterol. Diet changes recommended.',
        'Complaints of joint pain and stiffness. Suspected arthritis.',
        'Shortness of breath. Patient is a heavy smoker. Potential respiratory issues.',
        'Annual physical exam. All vitals normal. Stress management discussed.',
        'Abdominal discomfort, intermittent. Gastrointestinal consultation pending.',
        'High fever and body aches. Suspected influenza. Prescribed medication.'
    ],
    'disease_label': [0, 0, 1, 0, 0, 0, 1, 0, 0, 1] # 0: No disease, 1: Disease present
}
medical_records_df = pd.DataFrame(data)

print("Original Data:\n", medical_records_df.head())
print("\nMissing values before preprocessing:\n", medical_records_df.isnull().sum())

# 2. Data Preprocessing Layer

# --- Numerical Data Preprocessing ---

numerical_cols = ['age', 'blood_pressure_sys', 'blood_pressure_dia', 'cholesterol']
categorical_numerical_cols = ['gender'] # Example of a categorical feature that will be encoded

# Imputation for numerical columns
imputer = SimpleImputer(strategy='mean')
medical_records_df[numerical_cols] = imputer.fit_transform(medical_records_df[numerical_cols])

# Scaling numerical columns
scaler = StandardScaler()
scaled_numerical_features = scaler.fit_transform(medical_records_df[numerical_cols])
scaled_numerical_df = pd.DataFrame(scaled_numerical_features, columns=[f'{col}_scaled' for col in numerical_cols])

# One-hot encoding for categorical numerical features
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_categorical_features = ohe.fit_transform(medical_records_df[categorical_numerical_cols])
encoded_categorical_df = pd.DataFrame(encoded_categorical_features, columns=ohe.get_feature_names_out(categorical_numerical_cols))

# --- Text Data Preprocessing ---

stop_words = set(stopwords.words('english'))
potter_stemmer = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Tokenization and lowercasing
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]  # Stop-word removal
    # Applying both stemming and lemmatization for demonstration; usually one is chosen.
    stemmed_tokens = [potter_stemmer.stem(word) for word in filtered_tokens] # Stemming
    lemmatized_tokens = [wordnet_lemmatizer.lemmatize(word) for word in stemmed_tokens] # Lemmatization on stemmed words
    return ' '.join(lemmatized_tokens)

medical_records_df['processed_notes'] = medical_records_df['diagnosis_notes'].apply(preprocess_text)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=100) # Limiting features for demonstration
tfidf_features = tfidf_vectorizer.fit_transform(medical_records_df['processed_notes'])

print("\nNumerical features after imputation and scaling (first 5 rows):\n", scaled_numerical_df.head())
print("\nCategorical features after one-hot encoding (first 5 rows):\n", encoded_categorical_df.head())
print("\nProcessed text notes (first 5 rows):\n", medical_records_df[['diagnosis_notes', 'processed_notes']].head())
print("\nTF-IDF features shape:", tfidf_features.shape)

# 3. Feature Integration Layer
# Combine all processed features

# Convert dense numerical features to a sparse format to hstack with TF-IDF features
combined_numerical_features = hstack([scaled_numerical_features, encoded_categorical_features])

# Final combined feature matrix
X_processed = hstack([combined_numerical_features, tfidf_features])

# Target variable
y = medical_records_df['disease_label']

print("\nShape of combined numerical features (scaled + encoded):", combined_numerical_features.shape)
print("Shape of final integrated features (X_processed):", X_processed.shape)
print("Shape of target variable (y):", y.shape)

# 4. Model Training/Prediction Layer (Placeholder)
# This is where a machine learning model would be trained and used for prediction.
# For example:
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
# model = LogisticRegression()
# model.fit(X_train, y_train)
# predictions = model.predict(X_test)
# print("\nModel predictions (placeholder):", predictions)

# 5. Output/Evaluation Layer (Placeholder)
# This is where the results would be evaluated and presented.
# For example:
# from sklearn.metrics import accuracy_score, classification_report
# print("\nAccuracy (placeholder):", accuracy_score(y_test, predictions))
# print("\nClassification Report (placeholder):\n", classification_report(y_test, predictions))
