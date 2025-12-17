import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download NLTK resources if not already present
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

# Custom Transformer for Text Preprocessing
class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        processed_texts = []
        for text in X:
            tokens = word_tokenize(str(text).lower())
            filtered_tokens = [word for word in tokens if word.isalpha() and word not in self.stop_words]
            lemmatized_tokens = [self.lemmatizer.lemmatize(word) for word in filtered_tokens]
            processed_texts.append(" ".join(lemmatized_tokens))
        return pd.Series(processed_texts)

# 1. Load patient data (dummy data for demonstration)
data = {
    'age': [30, 65, 45, 70, 55, 25, 80, 40, 60, 50],
    'lab_results': [1.2, np.nan, 3.5, 2.1, 1.8, 0.9, 4.0, 2.5, np.nan, 1.5],
    'diagnosis_code': ['I10', 'E11', 'J44', 'I10', 'K21', 'R51', 'E11', 'I10', 'J44', 'K21'],
    'doctor_notes': [
        'Patient admitted for routine check-up. No significant issues.',
        'Diabetes management, high blood pressure. Readmission risk.',
        'COPD exacerbation, shortness of breath.',
        'Hypertension, previous heart attack. Follow-up needed.',
        'GERD, stable condition.',
        'Headache, mild symptoms. Discharged.',
        'Diabetic ketoacidosis. High readmission probability.',
        'Chest pain, ruled out MI. Discharged.',
        'Asthma attack, requires close monitoring.',
        'Stomach ulcer, routine medication.'
    ],
    'readmitted': [0, 1, 1, 0, 0, 0, 1, 0, 1, 0]
}
df = pd.DataFrame(data)

# Define features and target
X = df.drop('readmitted', axis=1)
y = df['readmitted']

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify numerical and textual features
numerical_features = ['age', 'lab_results']
categorical_features = ['diagnosis_code'] # Example of a categorical feature that might be one-hot encoded
text_features = ['doctor_notes']

# 3. Create separate preprocessing pipelines

# Numerical Pipeline: Imputation -> Scaling
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Categorical Pipeline: One-Hot Encoding
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Text Pipeline: Text Preprocessor -> TF-IDF Vectorization
text_transformer = Pipeline(steps=[
    ('text_preprocess', TextPreprocessor()),
    ('tfidf', TfidfVectorizer())
])

# 4. Use ColumnTransformer to apply these pipelines to the respective columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features),
        ('text', text_transformer, text_features)
    ],
    remainder='passthrough' # Keep other columns if any, though not expected here
)

# Create a full pipeline including preprocessing and model training
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# 5. Train a classification model on the preprocessed data
model_pipeline.fit(X_train, y_train)

# 6. Evaluate the model's performance
y_pred = model_pipeline.predict(X_test)
y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")