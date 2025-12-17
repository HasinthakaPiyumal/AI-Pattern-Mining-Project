import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Ensure NLTK data is downloaded
try:
    nltk.data.find("corpora/stopwords")
except nltk.downloader.DownloadError:
    nltk.download("stopwords")
try:
    nltk.data.find("corpora/wordnet")
except nltk.downloader.DownloadError:
    nltk.download("wordnet")
try:
    nltk.data.find("tokenizers/punkt")
except nltk.downloader.DownloadError:
    nltk.download("punkt")

# 1. Simulate Data Ingestion
data = {
    "ticket_id": range(1, 11),
    "ticket_description": [
        "My internet is not working, very urgent!",
        "I need to change my billing address.",
        "Can I get a refund for my last purchase?",
        "System crashed after the update, major bug.",
        "How to reset my password?",
        "Question about the new feature, please explain.",
        "My account is locked, help me unlock it.",
        "Billing inquiry, incorrect charge.",
        "Feature request: add dark mode.",
        "Slow performance, application freezing often."
    ],
    "customer_seniority": [5, 2, 1, 8, 3, 6, 4, 2, 7, 5],
    "previous_tickets_count": [10, 2, 1, 15, 3, 8, 5, 2, 12, 7],
    "submission_time_hour": [9, 14, 11, 10, 16, 13, 8, 15, 17, 9],
    "category": [
        "Technical Issue", "Billing", "Billing", "Technical Issue", "Technical Issue",
        "Feature Request", "Technical Issue", "Billing", "Feature Request", "Technical Issue"
    ],
    "priority": [
        "High", "Medium", "Low", "High", "Medium",
        "Medium", "High", "Medium", "Low", "High"
    ]
}
df = pd.DataFrame(data)

# Define target variables
X = df[["ticket_description", "customer_seniority", "previous_tickets_count", "submission_time_hour"]]
y_category = df["category"]
y_priority = df["priority"]

# 2. Data Preprocessing Pipeline

# a. Text Preprocessing Function
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def text_preprocessor(text):
    tokens = nltk.word_tokenize(text.lower())
    filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(filtered_tokens)

# Create a transformer for text preprocessing
text_processing_pipeline = Pipeline([
    ("vectorizer", TfidfVectorizer(preprocessor=text_preprocessor))
])

# b. Numerical Preprocessing Pipeline
numerical_features = ["customer_seniority", "previous_tickets_count", "submission_time_hour"]
numerical_processing_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

# 3. Feature Union/Combination
preprocessor = ColumnTransformer(
    transformers=[
        ("text", text_processing_pipeline, "ticket_description"),
        ("num", numerical_processing_pipeline, numerical_features)
    ])

# 4. Machine Learning Model

# Pipeline for Category Prediction
category_model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

# Pipeline for Priority Prediction
priority_model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

# 5. Training and Prediction

# Split data for Category prediction
X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(X, y_category, test_size=0.2, random_state=42)
category_model_pipeline.fit(X_train_cat, y_train_cat)
y_pred_cat = category_model_pipeline.predict(X_test_cat)

print("Category Prediction Report:")
print(classification_report(y_test_cat, y_pred_cat, zero_division=0))

# Split data for Priority prediction
X_train_prio, X_test_prio, y_train_prio, y_test_prio = train_test_split(X, y_priority, test_size=0.2, random_state=42)
priority_model_pipeline.fit(X_train_prio, y_train_prio)
y_pred_prio = priority_model_pipeline.predict(X_test_prio)

print("\nPriority Prediction Report:")
print(classification_report(y_test_prio, y_pred_prio, zero_division=0))

# Example of predicting a new ticket
new_ticket = pd.DataFrame({
    "ticket_description": ["My product is broken, urgent support needed."],
    "customer_seniority": [6],
    "previous_tickets_count": [11],
    "submission_time_hour": [10]
})

predicted_category = category_model_pipeline.predict(new_ticket)[0]
predicted_priority = priority_model_pipeline.predict(new_ticket)[0]

print(f"\nNew Ticket Prediction:")
print(f"  Description: {new_ticket['ticket_description'].iloc[0]}")
print(f"  Predicted Category: {predicted_category}")
print(f"  Predicted Priority: {predicted_priority}")