import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.sparse import hstack

import nltk
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

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# 1. Data Ingestion & Loading (Simulated Data)
def load_simulated_data():
    data = {
        'customer_id': range(1, 101),
        'total_spend': np.random.rand(100) * 1000 + 50,
        'num_orders': np.random.randint(1, 30, 100),
        'avg_items_per_order': np.random.rand(100) * 5 + 1,
        'days_since_last_purchase': np.random.randint(1, 365, 100),
        'review_sentiment': [
            "This product is amazing! I love it so much. Highly recommend.",
            "Terrible quality, completely broken after a week. Don't buy.",
            "It's okay, nothing special. Works as expected.",
            "Best purchase ever! So happy with everything.",
            "Very disappointed, customer service was unhelpful.",
            "Good value for money, fast delivery.",
            "Mediocre at best, would not repurchase.",
            "Excellent product, exceeded my expectations.",
            "Absolute garbage, waste of money.",
            "Decent for the price, no complaints."
        ] * 10,
        'churn': np.random.randint(0, 2, 100) # 0 for no churn, 1 for churn
    }
    df = pd.DataFrame(data)
    # Introduce some missing values for demonstration
    df.loc[df.sample(frac=0.05).index, 'total_spend'] = np.nan
    df.loc[df.sample(frac=0.03).index, 'review_sentiment'] = np.nan
    return df

df = load_simulated_data()

# Define numerical and text features
numerical_features = ['total_spend', 'num_orders', 'avg_items_per_order', 'days_since_last_purchase']
text_feature = 'review_sentiment'
target = 'churn'

# Separate features and target
X = df.drop(columns=[target, 'customer_id'])
y = df[target]

# 2. Data Preprocessing Pipeline

# 2.1 Numerical Data Preprocessing
# Imputation
数値補完器 = SimpleImputer(strategy='mean')
X[numerical_features] = 数値補完器.fit_transform(X[numerical_features])

# Feature Scaling
数値スケール = StandardScaler()
X_numerical_scaled = 数値スケール.fit_transform(X[numerical_features])
X_numerical_scaled_df = pd.DataFrame(X_numerical_scaled, columns=numerical_features, index=X.index)

# 2.2 Textual Data Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Remove punctuation, numbers, special characters
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

X['processed_review_sentiment'] = X[text_feature].apply(preprocess_text)

# Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=1000) # Limit features for simplicity
X_text_tfidf = tfidf_vectorizer.fit_transform(X['processed_review_sentiment'])

# 3. Feature Combination
# Convert numerical scaled features back to a sparse matrix for hstack compatibility
X_numerical_sparse = X_numerical_scaled_df.sparse.to_coo()
X_combined = hstack([X_numerical_sparse, X_text_tfidf])

# 4. Model Training & Prediction
# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Model Training (Logistic Regression)
model = LogisticRegression(solver='liblinear', random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1] # Probability of churn

# 5. Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("Churn Prediction Model Evaluation:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

print("\nExample Predictions on Test Set:")
for i in range(min(5, len(y_test))):
    print(f"Actual: {y_test.iloc[i]}, Predicted: {y_pred[i]}, Churn Probability: {y_prob[i]:.4f}")