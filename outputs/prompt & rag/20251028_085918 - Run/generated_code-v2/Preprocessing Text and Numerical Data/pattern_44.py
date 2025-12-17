import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data (run once)
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

# --- 1. Synthetic Data Generation ---
def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)
    data = {
        'patient_id': range(num_samples),
        'age': np.random.randint(18, 90, num_samples),
        'gender': np.random.choice(['Male', 'Female'], num_samples),
        'num_previous_admissions': np.random.randint(0, 5, num_samples),
        'lab_result_creatinine': np.random.normal(1.0, 0.3, num_samples),
        'medication_count': np.random.randint(1, 15, num_samples),
        'diagnosis_code': np.random.choice(['I10', 'E11', 'J45', 'K21', 'N18'], num_samples),
        'clinical_notes': ['Patient presented with symptoms of ' + 
                           np.random.choice(['fever and cough', 'chest pain', 'abdominal discomfort', 'headache and nausea']) + 
                           '. Discharge plan involves ' + 
                           np.random.choice(['follow-up in 2 weeks', 'medication adherence', 'dietary changes']) + '.'
                           for _ in range(num_samples)],
        'readmitted_30_days': np.random.choice([0, 1], num_samples, p=[0.7, 0.3]) # 0: No, 1: Yes
    }
    df = pd.DataFrame(data)
    
    # Introduce some missing values for demonstration
    for col in ['lab_result_creatinine', 'medication_count']:
        df.loc[df.sample(frac=0.05).index, col] = np.nan
    df.loc[df.sample(frac=0.02).index, 'diagnosis_code'] = np.nan
    
    return df

# --- 2. Text Preprocessing Function ---
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if isinstance(text, float) and np.isnan(text): # Handle NaN values which might occur if column has NaNs after other processing
        return ""
    text = str(text).lower() # Convert to string and lowercase
    text = re.sub(r'[^a-z\s]', '', text) # Remove non-alphabetic characters
    tokens = nltk.word_tokenize(text) # Tokenize
    tokens = [word for word in tokens if word not in stop_words] # Remove stopwords
    tokens = [lemmatizer.lemmatize(word) for word in tokens] # Lemmatize
    return ' '.join(tokens)

# --- 3. Build Preprocessing and ML Pipeline ---
def build_readmission_prediction_pipeline():
    # Define features by type
    numerical_features = ['age', 'num_previous_admissions', 'lab_result_creatinine', 'medication_count']
    categorical_features = ['gender', 'diagnosis_code']
    text_features = 'clinical_notes'

    # Create preprocessing pipelines for numerical and categorical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Create a text processing pipeline
    text_transformer = Pipeline(steps=[
        ('tfidf', TfidfVectorizer(preprocessor=preprocess_text, max_features=5000)) # Using custom preprocessor
    ])

    # Combine all preprocessing steps using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features),
            ('txt', text_transformer, text_features)
        ], 
        remainder='drop'
    )

    # Create the full machine learning pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, solver='liblinear')) # Using Logistic Regression as a base model
    ])
    
    return model_pipeline

# --- 4. Main Execution ---
if __name__ == "__main__":
    print("Generating synthetic patient data...")
    df = generate_synthetic_data(num_samples=2000)
    
    X = df.drop('readmitted_30_days', axis=1)
    y = df['readmitted_30_days']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    print("Building and training the patient readmission prediction pipeline...")
    pipeline = build_readmission_prediction_pipeline()
    pipeline.fit(X_train, y_train)
    
    print("Evaluating the model...")
    y_pred = pipeline.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"\nModel Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", report)
    
    print("\nDemonstrating prediction on a new (synthetic) patient:")
    # Create a new synthetic patient for prediction
    new_patient_data = pd.DataFrame({
        'patient_id': [9999],
        'age': [75],
        'gender': ['Female'],
        'num_previous_admissions': [2],
        'lab_result_creatinine': [1.5],
        'medication_count': [8],
        'diagnosis_code': ['I10'],
        'clinical_notes': ['Patient discharged recently, complaining of fatigue. Follow-up recommended in 1 week.']
    })
    
    predicted_readmission = pipeline.predict(new_patient_data)[0]
    probability_readmission = pipeline.predict_proba(new_patient_data)[0][1]
    
    status = "likely to be readmitted" if predicted_readmission == 1 else "unlikely to be readmitted"
    print(f"The new patient is {status} (Probability: {probability_readmission:.4f}).")
