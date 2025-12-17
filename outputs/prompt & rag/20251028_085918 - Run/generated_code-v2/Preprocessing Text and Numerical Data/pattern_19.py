import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
import numpy as np

# Download NLTK data (run once)
try:
    nltk.data.find('corpora/wordnet')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# --- 1. Text Data Pipeline (Sentiment Analysis) ---

# Simulate product review text data
text_data = {
    'review': [
        "This product is amazing! I love it so much.",
        "Absolutely terrible, a complete waste of money.",
        "It's okay, nothing special, but not bad either.",
        "Highly recommended, will definitely buy again.",
        "Worst experience ever, customer service was awful.",
        "Decent quality for the price, quite satisfied.",
        "Not what I expected, a bit disappointed."
    ],
    'sentiment': ['positive', 'negative', 'neutral', 'positive', 'negative', 'positive', 'negative']
}
text_df = pd.DataFrame(text_data)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text) # Remove punctuation and special characters
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

print("\n--- Text Data Preprocessing and Sentiment Analysis ---")
print("Original Reviews:")
print(text_df)

text_df['cleaned_review'] = text_df['review'].apply(preprocess_text)

print("\nCleaned Reviews:")
print(text_df[['review', 'cleaned_review']])

# Split data for training and testing
X_text = text_df['cleaned_review']
y_text = text_df['sentiment']
X_train_text, X_test_text, y_train_text, y_test_text = train_test_split(X_text, y_text, test_size=0.3, random_state=42, stratify=y_text)

# TF-IDF Vectorization and Logistic Regression Model in a Pipeline
text_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000)), # Limiting features for demonstration
    ('classifier', LogisticRegression(max_iter=1000)) # Increased max_iter for convergence
])

# Train the sentiment analysis model
text_pipeline.fit(X_train_text, y_train_text)

# Make predictions
y_pred_text = text_pipeline.predict(X_test_text)

print("\nSentiment Analysis Predictions:")
print(f"Test Accuracy: {accuracy_score(y_test_text, y_pred_text):.2f}")
print("Classification Report:\n", classification_report(y_test_text, y_pred_text))

# --- 2. Numerical Data Pipeline (Fraud Detection) ---

# Simulate numerical transaction data
np.random.seed(42)
n_samples = 200
fraud_ratio = 0.05
n_fraud = int(n_samples * fraud_ratio)
n_normal = n_samples - n_fraud

# Normal transactions
normal_data = {
    'purchase_amount': np.random.normal(50, 20, n_normal).round(2),
    'items_in_cart': np.random.randint(1, 10, n_normal),
    'user_rating': np.random.uniform(3, 5, n_normal).round(1),
    'transaction_frequency': np.random.randint(1, 30, n_normal),
    'payment_method': np.random.choice(['Credit Card', 'PayPal', 'Debit Card'], n_normal),
    'is_fraud': np.zeros(n_normal, dtype=int)
}

# Fraudulent transactions (with some anomalies)
fraud_data = {
    'purchase_amount': np.random.normal(200, 50, n_fraud).round(2), # Higher amounts
    'items_in_cart': np.random.randint(10, 25, n_fraud), # More items
    'user_rating': np.random.uniform(1, 2, n_fraud).round(1), # Lower ratings
    'transaction_frequency': np.random.randint(30, 60, n_fraud), # High frequency
    'payment_method': np.random.choice(['Credit Card', 'Gift Card'], n_fraud),
    'is_fraud': np.ones(n_fraud, dtype=int)
}

# Introduce some missing values and outliers for demonstration
normal_data['purchase_amount'][np.random.choice(n_normal, 5, replace=False)] = np.nan
normal_data['user_rating'][np.random.choice(n_normal, 3, replace=False)] = np.nan

fraud_df = pd.DataFrame(fraud_data)
normal_df = pd.DataFrame(normal_data)
numerical_df = pd.concat([normal_df, fraud_df], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

print("\n--- Numerical Data Preprocessing and Fraud Detection ---")
print("Original Numerical Data (first 5 rows):")
print(numerical_df.head())

# Define numerical and categorical features
numerical_features = ['purchase_amount', 'items_in_cart', 'user_rating', 'transaction_frequency']
categorical_features = ['payment_method']

# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create a preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create the full fraud detection pipeline
fraud_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('detector', IsolationForest(random_state=42))
])

# Separate features (X) and target (y)
X_numerical = numerical_df.drop('is_fraud', axis=1)
y_numerical = numerical_df['is_fraud']

# Fit the fraud detection pipeline
fraud_pipeline.fit(X_numerical)

# Predict anomalies (fraud) - IsolationForest returns -1 for outliers/anomalies, 1 for inliers
anomaly_scores = fraud_pipeline.decision_function(X_numerical)
predictions = fraud_pipeline.predict(X_numerical)

# Convert predictions to 0 for normal, 1 for fraud for easier comparison
# IsolationForest: -1 (anomaly/fraud), 1 (normal)
# We want: 1 (fraud), 0 (normal)
fraud_predictions = np.where(predictions == -1, 1, 0)

print("\nFraud Detection Results (first 10 rows):")
result_df = X_numerical.copy()
result_df['actual_is_fraud'] = y_numerical
result_df['anomaly_score'] = anomaly_scores
result_df['predicted_is_fraud'] = fraud_predictions

print(result_df[['purchase_amount', 'payment_method', 'actual_is_fraud', 'predicted_is_fraud', 'anomaly_score']].head(10))

print("\nSummary of Fraud Detection:")
print(f"Total transactions: {len(result_df)}")
print(f"Actual fraud instances: {result_df['actual_is_fraud'].sum()}")
print(f"Predicted fraud instances: {result_df['predicted_is_fraud'].sum()}")
print(f"Correctly identified fraud (True Positives): {((result_df['actual_is_fraud'] == 1) & (result_df['predicted_is_fraud'] == 1)).sum()}")
print(f"Missed fraud (False Negatives): {((result_df['actual_is_fraud'] == 1) & (result_df['predicted_is_fraud'] == 0)).sum()}")
print(f"False alarms (False Positives): {((result_df['actual_is_fraud'] == 0) & (result_df['predicted_is_fraud'] == 1)).sum()}")
