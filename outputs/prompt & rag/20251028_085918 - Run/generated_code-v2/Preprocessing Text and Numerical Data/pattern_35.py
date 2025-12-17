import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer

# Download NLTK data (if not already downloaded)
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except nltk.downloader.DownloadError:
    nltk.download('wordnet')

# 1. Simulate Raw Patient Data
np.random.seed(42)
data = {
    'patient_id': range(1, 11),
    'age': np.random.randint(20, 80, 10),
    'blood_pressure': np.random.normal(120, 15, 10),
    'cholesterol': np.random.normal(200, 30, 10),
    'gender': np.random.choice(['Male', 'Female'], 10),
    'symptoms_text': [
        'Patient complains of severe headache and fever. history of migraine.',
        'Mild cough and sore throat. no fever reported.',
        'Abdominal pain after eating. nausea sometimes.',
        'Fatigue and joint pain. recent viral infection.',
        'No specific complaints, routine check-up. good health.',
        'High blood pressure reading, advised lifestyle changes.',
        'Experiencing dizziness and blurred vision. diabetes history.',
        'Skin rash on arms and legs. possible allergic reaction.',
        'Follow-up for chronic back pain. needs physical therapy.',
        'Difficulty breathing, shortness of breath. smoker.'
    ],
    'doctor_notes': [
        'Initial assessment: Migraine episode, prescribed pain relievers.',
        'Symptoms mild, advised rest and hydration.',
        'Gastrointestinal discomfort, recommended dietary adjustments.',
        'Post-viral fatigue, advised rest and monitor.',
        'Patient in good health, preventive care discussed.',
        'Hypertension management, follow-up in 3 months.',
        'Neurological consultation recommended due to vision issues.',
        'Dermatology referral for rash.',
        'Chronic pain management, referred to physiotherapy.',
        'Respiratory issues, advised to stop smoking, lung function tests scheduled.'
    ]
}

# Introduce some missing values for demonstration
data['blood_pressure'][np.random.choice(10, 2, replace=False)] = np.nan
data['cholesterol'][np.random.choice(10, 1, replace=False)] = np.nan
data['gender'][np.random.choice(10, 1, replace=False)] = np.nan
data['symptoms_text'][np.random.choice(10, 1, replace=False)] = np.nan

df = pd.DataFrame(data)

# Define numerical and text features
numerical_features = ['age', 'blood_pressure', 'cholesterol']
categorical_features = ['gender'] # Numerical features that are categorical
text_features = ['symptoms_text', 'doctor_notes']

# 2. Preprocessing Layer

# Numerical Preprocessing Pipeline
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Categorical Numerical Preprocessing Pipeline
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Text Preprocessing Function (for NLTK components)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if pd.isna(text):
        return ""
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(tokens)

# Text Preprocessing Pipeline
# TF-IDF Vectorizer will handle text preprocessing after our custom function
text_transformer = Pipeline(steps=[
    ('fillna', SimpleImputer(strategy='constant', fill_value='')), # Handle NaN in text
    ('tfidf', TfidfVectorizer(preprocessor=preprocess_text, tokenizer=lambda x: x.split()))
    # Custom preprocessor handles tokenization, stop-word removal, lemmatization
    # tokenizer=lambda x: x.split() is needed because preprocessor returns a string already tokenized and joined
])

# Combine all preprocessors using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features),
        ('text_symptoms', text_transformer, 'symptoms_text'), # Apply to each text column separately
        ('text_doctor', text_transformer, 'doctor_notes')
    ], 
    remainder='passthrough' # Keep other columns if any, or 'drop' if not needed
)

# 3. Apply Preprocessor to Data
# We need to ensure that the ColumnTransformer output can be easily converted back to a DataFrame for clarity.
# Get feature names after transformation for better output interpretation.

# Fit and transform the data
preprocessed_data_array = preprocessor.fit_transform(df)

# Get feature names after transformation
feature_names = []

# Numerical features
for col in numerical_features:
    feature_names.append(col + '_scaled')

# Categorical features
onehot_features = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
feature_names.extend(onehot_features)

# Text features (TF-IDF)
# TF-IDF vectorizer produces a sparse matrix of features. The number of features can be large.
# For demonstration, let's just indicate the original text column name.
# In a real scenario, you'd get the actual TF-IDF feature names (words) if you need to inspect them.
# Here, we'll create a placeholder for the number of features each TF-IDF vectorizer generates.

symptoms_tfidf_transformer = preprocessor.named_transformers_['text_symptoms']['tfidf']
doctor_notes_tfidf_transformer = preprocessor.named_transformers_['text_doctor']['tfidf']

# Since TF-IDF outputs many features, we'll represent them by the original column name and a count for explanation.
# For actual feature names, you'd call .get_feature_names_out() on the fitted TF-IDF vectorizer.
# This might result in a very wide DataFrame, so for this example, we'll indicate the concept.

# Create a dummy DataFrame to get the column names correctly from ColumnTransformer
dummy_df = pd.DataFrame(np.zeros((1, len(numerical_features) + len(categorical_features))), 
                        columns=numerical_features + categorical_features)
dummy_preprocessed = preprocessor.named_transformers_['num'].fit_transform(df[numerical_features])
dummy_cat_preprocessed = preprocessor.named_transformers_['cat'].fit_transform(df[categorical_features])

# Getting TF-IDF feature names requires fitting on a text series first to build vocabulary
# Let's fit separate vectorizers temporarily to get feature names or approximate the size
symptoms_temp_vectorizer = TfidfVectorizer(preprocessor=preprocess_text, tokenizer=lambda x: x.split())
symptoms_temp_vectorizer.fit(df['symptoms_text'].fillna(''))
symptoms_tfidf_feature_names = [f'symptoms_text_tfidf_{f}' for f in symptoms_temp_vectorizer.get_feature_names_out()]

doctor_notes_temp_vectorizer = TfidfVectorizer(preprocessor=preprocess_text, tokenizer=lambda x: x.split())
doctor_notes_temp_vectorizer.fit(df['doctor_notes'].fillna(''))
doctor_notes_tfidf_feature_names = [f'doctor_notes_tfidf_{f}' for f in doctor_notes_temp_vectorizer.get_feature_names_out()]

# Combine all feature names
final_feature_names = \
    [f + '_scaled' for f in numerical_features] + \
    list(preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)) + \
    symptoms_tfidf_feature_names + \
    doctor_notes_tfidf_feature_names

# Create the preprocessed DataFrame
# The ColumnTransformer output is a sparse matrix if TF-IDF is used, or dense numpy array otherwise.
# We need to handle this to convert to DataFrame.
if isinstance(preprocessed_data_array, np.ndarray):
    preprocessed_df = pd.DataFrame(preprocessed_data_array, columns=final_feature_names)
elif hasattr(preprocessed_data_array, 'tocoo'): # It's a sparse matrix
    preprocessed_df = pd.DataFrame(preprocessed_data_array.toarray(), columns=final_feature_names)


# Display original and preprocessed data snippets
print("Original Data Head:")
print(df.head())
print("\nOriginal Data Info:")
print(df.info())
print("\nPreprocessed Data Head (First 5 rows and a few columns, TF-IDF features can be numerous):")
# Display only a subset of TF-IDF features for readability if they are too many
if len(preprocessed_df.columns) > 10:
    print(preprocessed_df.iloc[:, :10].head())
else:
    print(preprocessed_df.head())

print("\nPreprocessed Data Info:")
print(preprocessed_df.info())
print("\nShape of Preprocessed Data:", preprocessed_df.shape)

# Example of how to access individual preprocessed components (e.g., TF-IDF features for 'symptoms_text')
# This would be done within a downstream ML model or for specific analysis.
# For example, to get the TF-IDF features for 'symptoms_text':
# symptoms_tfidf_matrix = preprocessor.named_transformers_['text_symptoms'].transform(df['symptoms_text'].fillna('').to_frame())
# print("\nShape of symptoms_text TF-IDF matrix:", symptoms_tfidf_matrix.shape)
