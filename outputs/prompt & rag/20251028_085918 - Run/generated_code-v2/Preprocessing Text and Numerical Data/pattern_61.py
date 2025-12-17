import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

# Download NLTK resources if not already present
try:
    nltk.data.find('corpora/wordnet')
except nltk.downloader.DownloadError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# --- 1. Simulate Data Input ---
data = {
    'query': [
        'My internet is not working. I need help.',
        'How can I reset my password?',
        'I want to upgrade my plan.',
        'Billing issue with my last invoice.',
        'The mobile app crashed again.',
        'Can I get a refund for a service?',
        'Technical support needed for network.',
        'Complaint about slow connection speed.',
        'Question about my data usage.',
        'Account activation problem.',
        'I received a wrong item in my order.'
    ],
    'priority': [1, 2, 1, 0, 1, 0, 1, 1, 2, 0, 1],
    'ticket_age_days': [2, 0, 1, 3, 0, 1, 2, 1, 0, 1, 2],
    'num_previous_interactions': [5, 1, 2, 8, 0, 3, 4, 6, 1, 2, 0],
    'department': [
        'Technical', 'Account', 'Sales', 'Billing', 'Technical', 
        'Billing', 'Technical', 'Technical', 'Account', 'Account', 'Support'
    ],
    'sentiment_label': [
        'Negative', 'Neutral', 'Positive', 'Negative', 'Negative', 
        'Neutral', 'Negative', 'Negative', 'Neutral', 'Negative', 'Negative'
    ]
}
df = pd.DataFrame(data)

# Introduce some missing numerical values for demonstration
df.loc[[1, 5, 8], 'ticket_age_days'] = np.nan
df.loc[[0, 3], 'num_previous_interactions'] = np.nan

print("--- Original Data ---")
print(df.head())
print("\n")

# --- 2. Preprocessing Module ---

# 2.1 Text Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(filtered_tokens)

df['processed_query'] = df['query'].apply(preprocess_text)

tfidf_vectorizer = TfidfVectorizer(max_features=100) # Limit features for demonstration
X_text = tfidf_vectorizer.fit_transform(df['processed_query']).toarray()

# 2.2 Numerical Preprocessing
numerical_features = ['priority', 'ticket_age_days', 'num_previous_interactions']
X_numerical = df[numerical_features].copy()

imputer = SimpleImputer(strategy='mean')
X_numerical_imputed = imputer.fit_transform(X_numerical)

scaler = StandardScaler()
X_numerical_scaled = scaler.fit_transform(X_numerical_imputed)

# 2.3 Feature Combination
X_combined = np.hstack((X_text, X_numerical_scaled))

print("--- Preprocessing Complete ---")
print(f"Shape of TF-IDF features: {X_text.shape}")
print(f"Shape of scaled numerical features: {X_numerical_scaled.shape}")
print(f"Shape of combined features: {X_combined.shape}")
print("\n")

# --- 3. Machine Learning Models ---

# 3.1 Ticket Routing Model
y_routing = df['department']
X_train_route, X_test_route, y_train_route, y_test_route = train_test_split(
    X_combined, y_routing, test_size=0.2, random_state=42, stratify=y_routing
)

routing_model = RandomForestClassifier(n_estimators=100, random_state=42)
routing_model.fit(X_train_route, y_train_route)

print("Ticket Routing Model Trained.")
# print(f"Routing Model Accuracy: {routing_model.score(X_test_route, y_test_route):.2f}") # For evaluation

# 3.2 Sentiment Analysis Model
y_sentiment = df['sentiment_label']
X_train_sent, X_test_sent, y_train_sent, y_test_sent = train_test_split(
    X_combined, y_sentiment, test_size=0.2, random_state=42, stratify=y_sentiment
)

sentiment_model = LogisticRegression(max_iter=1000, random_state=42)
sentiment_model.fit(X_train_sent, y_train_sent)

print("Sentiment Analysis Model Trained.\n")
# print(f"Sentiment Model Accuracy: {sentiment_model.score(X_test_sent, y_test_sent):.2f}") # For evaluation

# --- 4. Prediction Demonstration ---
print("--- Demonstrating Predictions for New Tickets ---")

# New sample tickets
new_tickets_data = [
    {'query': 'My internet is extremely slow today, I am very frustrated.', 'priority': 1, 'ticket_age_days': 0, 'num_previous_interactions': 7},
    {'query': 'I want to know more about the premium package features.', 'priority': 2, 'ticket_age_days': 1, 'num_previous_interactions': 1},
    {'query': 'My bill seems incorrect, please check the charges.', 'priority': 0, 'ticket_age_days': np.nan, 'num_previous_interactions': 3}
]
new_df = pd.DataFrame(new_tickets_data)

# Apply the same preprocessing steps to new data
new_df['processed_query'] = new_df['query'].apply(preprocess_text)
new_X_text = tfidf_vectorizer.transform(new_df['processed_query']).toarray()

new_X_numerical = new_df[numerical_features].copy()
new_X_numerical_imputed = imputer.transform(new_X_numerical) # Use fitted imputer
new_X_numerical_scaled = scaler.transform(new_X_numerical_imputed) # Use fitted scaler

new_X_combined = np.hstack((new_X_text, new_X_numerical_scaled))

# Make predictions
predicted_departments = routing_model.predict(new_X_combined)
predicted_sentiments = sentiment_model.predict(new_X_combined)

for i, row in new_df.iterrows():
    print(f"Original Query: '{row['query']}'")
    print(f"  Predicted Department: {predicted_departments[i]}")
    print(f"  Predicted Sentiment: {predicted_sentiments[i]}")
    print("\n")
