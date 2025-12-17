import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Ensure NLTK resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except nltk.downloader.DownloadError:
    nltk.download('wordnet')

# --- Text Preprocessing Function ---
def preprocess_text(text):
    if not isinstance(text, str): # Handle potential non-string inputs
        return ""
    
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Tokenization
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords and lemmatize
    processed_tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token.isalpha() and token not in stop_words
    ]
    
    return " ".join(processed_tokens)

# --- Main Patient Readmission Prediction System Class ---
class PatientReadmissionPredictor:
    def __init__(self):
        # Define preprocessing for numerical features
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        # Define preprocessing for text features
        # We apply the custom text preprocessing function first, then TF-IDF
        text_transformer = Pipeline(steps=[
            ('tfidf', TfidfVectorizer(preprocessor=preprocess_text, max_features=5000)) # Adjust max_features as needed
        ])

        # Combine numerical and text transformers using ColumnTransformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, []), # Placeholder for numerical column names
                ('text', text_transformer, [])   # Placeholder for text column names
            ], 
            remainder='passthrough' # Keep other columns as is (or 'drop')
        )

        # Define the full pipeline: preprocessing + model
        self.model_pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', LogisticRegression(random_state=42, solver='liblinear')) # Use a robust solver
        ])
        
        self.numerical_features = []
        self.text_features = []

    def fit(self, X: pd.DataFrame, y: pd.Series, numerical_cols: list, text_cols: list):
        """
        Fits the readmission prediction model.

        Args:
            X (pd.DataFrame): Input features (numerical and text).
            y (pd.Series): Target variable (e.g., 0 for no readmission, 1 for readmission).
            numerical_cols (list): List of column names corresponding to numerical features.
            text_cols (list): List of column names corresponding to text features.
        """
        if not numerical_cols and not text_cols:
            raise ValueError("At least one of numerical_cols or text_cols must be provided.")

        self.numerical_features = numerical_cols
        self.text_features = text_cols

        # Update ColumnTransformer with actual column names
        self.preprocessor.transformers[0] = ('num', self.preprocessor.transformers[0][1], self.numerical_features)
        self.preprocessor.transformers[1] = ('text', self.preprocessor.transformers[1][1], self.text_features)

        print("Fitting model...")
        self.model_pipeline.fit(X, y)
        print("Model fitted successfully.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts readmission likelihood for new patient data.

        Args:
            X (pd.DataFrame): New input features.

        Returns:
            np.ndarray: Predicted readmission labels (0 or 1).
        """
        if not self.numerical_features and not self.text_features:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        print("Making predictions...")
        return self.model_pipeline.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts readmission probabilities for new patient data.

        Args:
            X (pd.DataFrame): New input features.

        Returns:
            np.ndarray: Predicted readmission probabilities.
        """
        if not self.numerical_features and not self.text_features:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        print("Making probability predictions...")
        return self.model_pipeline.predict_proba(X)

# --- Example Usage ---
if __name__ == "__main__":
    # 1. Generate Dummy Data
    np.random.seed(42)
    data_size = 100

    dummy_data = {
        'patient_id': range(data_size),
        'age': np.random.randint(20, 90, data_size),
        'systolic_bp': np.random.randint(100, 180, data_size),
        'diastolic_bp': np.random.randint(60, 110, data_size),
        'lab_result_a': np.random.rand(data_size) * 100,
        'previous_admissions': np.random.randint(0, 5, data_size),
        'doctor_notes': [
            "Patient presented with severe chest pain and shortness of breath. History of cardiac issues.",
            "Follow-up visit. Patient feeling much better after medication change. No new symptoms reported.",
            "Diabetic patient with high blood sugar levels. Advised dietary changes and insulin adjustment.",
            "Routine check-up, no significant findings. Patient stable.",
            "Acute pneumonia, admitted for observation and IV antibiotics. Coughing heavily."
        ] * (data_size // 5),
        'discharge_summary': [
            "Discharged after successful cardiac intervention. Advised strict adherence to medication.",
            "Patient stable, no complications. Discharged with follow-up appointment.",
            "Diabetic ketoacidosis resolved. Discharged with new insulin regimen and diet plan.",
            "Standard discharge after minor procedure. Good recovery.",
            "Pneumonia resolved, breathing improved. Discharged with oral antibiotics and rest advice."
        ] * (data_size // 5),
        'readmitted': np.random.randint(0, 2, data_size) # 0: No, 1: Yes
    }
    # Introduce some missing values for demonstration
    for col in ['systolic_bp', 'lab_result_a']:
        dummy_data[col][np.random.choice(data_size, 5, replace=False)] = np.nan
    dummy_data['doctor_notes'][np.random.choice(data_size, 3, replace=False)] = np.nan

    df = pd.DataFrame(dummy_data)

    # Define feature and target columns
    numerical_features = ['age', 'systolic_bp', 'diastolic_bp', 'lab_result_a', 'previous_admissions']
    text_features = ['doctor_notes', 'discharge_summary']
    target_feature = 'readmitted'

    # Split data
    X = df[numerical_features + text_features]
    y = df[target_feature]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")

    # 2. Initialize and Train the Predictor
    predictor = PatientReadmissionPredictor()
    predictor.fit(X_train, y_train, numerical_features, text_features)

    # 3. Make Predictions
    y_pred = predictor.predict(X_test)
    y_proba = predictor.predict_proba(X_test)

    print("\n--- Prediction Results ---")
    print("Sample actual values (test set):\n", y_test.head())
    print("Sample predicted values:\n", pd.Series(y_pred).head())
    print("Sample predicted probabilities (for class 1):\n", pd.Series(y_proba[:, 1]).head())

    # 4. Evaluate (Basic evaluation)
    from sklearn.metrics import accuracy_score, classification_report

    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))

    print("\nPreprocessing and prediction pipeline demonstrated successfully!")
