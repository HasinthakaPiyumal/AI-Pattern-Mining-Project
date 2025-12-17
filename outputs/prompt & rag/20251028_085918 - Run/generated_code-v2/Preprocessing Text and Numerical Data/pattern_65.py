import pandas as pd
import numpy as np
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Ensure NLTK resources are downloaded
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

# --- 1. Data Loading Module (Simulated Data) ---
# Simulate structured numerical data (e.g., patient demographics, lab results)
data_numerical = pd.DataFrame({
    'patient_id': range(1, 101),
    'age': np.random.randint(20, 90, 100),
    'blood_pressure_systolic': np.random.randint(90, 180, 100),
    'blood_pressure_diastolic': np.random.randint(60, 120, 100),
    'cholesterol': np.random.randint(150, 300, 100),
    'bmi': np.random.uniform(18.0, 35.0, 100),
    'gender': np.random.choice(['Male', 'Female'], 100),
    'smoker': np.random.choice([0, 1], 100),
    'outcome': np.random.choice([0, 1], 100) # 0 for healthy, 1 for disease
})

# Introduce some missing values
for col in ['age', 'blood_pressure_systolic', 'cholesterol']:
    data_numerical.loc[np.random.choice(data_numerical.index, 10), col] = np.nan

# Simulate unstructured text data (e.g., physician notes)
data_text = pd.DataFrame({
    'patient_id': range(1, 101),
    'physician_notes': [
        "Patient presented with mild fever and cough. No significant distress noted. Advised rest and fluids.",
        "History of hypertension. Current blood pressure elevated. Prescribed medication and follow-up.",
        "Routine check-up. Patient reports feeling well. No new concerns. Cholesterol levels stable.",
        "Severe abdominal pain. Suspected appendicitis. Emergency consultation initiated.",
        "Follow-up after surgery. Wound healing well. Patient recovering as expected. Minimal pain reported.",
        "New onset of headache and dizziness. Further neurological examination recommended. Family history of migraines.",
        "Diabetic patient. Blood sugar levels well controlled. Diet and exercise plan reviewed.",
        "Annual physical. All vitals within normal limits. Discussed preventative care.",
        "Respiratory infection symptoms. Shortness of breath reported. Chest X-ray ordered.",
        "Patient complains of chronic fatigue. Thyroid function tests ordered. Advised lifestyle changes."
    ] * 10 # Repeat to match 100 patients
})

# Combine patient data
patient_data = pd.merge(data_numerical, data_text, on='patient_id')

X = patient_data.drop('outcome', axis=1)
y = patient_data['outcome']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 3. Text Data Preprocessing Module ---
class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text) # Remove punctuation and special characters
        return text

    def tokenize_and_lemmatize(self, text):
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in self.stop_words]
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        return " ".join(tokens)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.Series):
            X_cleaned = X.apply(self.clean_text)
            X_processed = X_cleaned.apply(self.tokenize_and_lemmatize)
        elif isinstance(X, np.ndarray):
            X_series = pd.Series(X.flatten())
            X_cleaned = X_series.apply(self.clean_text)
            X_processed = X_cleaned.apply(self.tokenize_and_lemmatize)
        else:
            raise ValueError("Input must be a pandas Series or numpy array.")
        return X_processed

# --- 2. Numerical Data Preprocessing Module ---
numerical_features = ['age', 'blood_pressure_systolic', 'blood_pressure_diastolic', 'cholesterol', 'bmi']
categorical_features = ['gender', 'smoker']
text_features = 'physician_notes'

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

text_transformer = Pipeline(steps=[
    ('text_preprocessor', TextPreprocessor()),
    ('tfidf', TfidfVectorizer())
])

# Create a ColumnTransformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features),
        ('text', text_transformer, text_features)
    ],
    remainder='drop'
)

# --- 4. Feature Integration and Model Training Module ---
# Create the full pipeline with preprocessing and a Logistic Regression model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(solver='liblinear', random_state=42))
])

# Train the model
model_pipeline.fit(X_train, y_train)

# --- 5. Evaluation Module ---
# Make predictions on the test set
y_pred = model_pipeline.predict(X_test)

# Evaluate the model
print("\nModel Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nPreprocessing and prediction complete. The model pipeline is ready for deployment.")
