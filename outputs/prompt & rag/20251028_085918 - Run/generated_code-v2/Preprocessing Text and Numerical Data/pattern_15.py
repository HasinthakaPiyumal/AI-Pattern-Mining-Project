import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download necessary NLTK data
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

# 1. Sample Data Loading
data = {
    'product_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'product_name': ['Laptop Pro', 'Smartphone X', 'Wireless Earbuds', 'Smartwatch Z', 'Gaming Mouse', 'Keyboard G', 'Monitor M', 'Webcam C', 'Router R', 'External SSD'],
    'review_text': [
        "This laptop is amazing, super fast and great battery life. Highly recommend!",
        "Phone is decent, but the camera is not as good as expected.",
        "Earbuds are comfortable and sound great, good value for money.",
        "Smartwatch has many features, but battery drains quickly. Okay purchase.",
        "Best gaming mouse ever! Very precise and comfortable.",
        "Keyboard is okay, keys are a bit mushy.",
        "Monitor has excellent display quality. Perfect for work.",
        "Webcam resolution is poor, disappointed with the quality.",
        "Router provides stable connection, easy to set up.",
        "SSD is fast for data transfer, a bit expensive though."
    ],
    'rating': [5, 3, 4, 3, 5, 3, 5, 2, 4, 4],
    'price': [1200, 800, 100, 250, 70, 90, 300, 50, 120, 150],
    'sales_volume': [1500, 800, 2000, 500, 1200, 700, 1000, 300, 600, 900],
    'sentiment': ['positive', 'negative', 'positive', 'neutral', 'positive', 'neutral', 'positive', 'negative', 'positive', 'neutral'] # Manual sentiment for demonstration
}
df = pd.DataFrame(data)

# Introduce some missing values for demonstration of imputation
df.loc[[1, 7], 'rating'] = np.nan
df.loc[[3, 9], 'sales_volume'] = np.nan

print("Original DataFrame head:\n", df.head())

# 2. Numerical Data Preprocessing
# Imputation for missing numerical values
num_cols = ['rating', 'price', 'sales_volume']
imputer = SimpleImputer(strategy='mean')
df[num_cols] = imputer.fit_transform(df[num_cols])

# Scaling numerical features
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

print("\nDataFrame after numerical preprocessing head:\n", df.head())

# 3. Text Data Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    return " ".join(lemmatized_tokens)

df['processed_review'] = df['review_text'].apply(preprocess_text)

print("\nDataFrame after text preprocessing head:\n", df[['review_text', 'processed_review']].head())

# 4. Text Vectorization (TF-IDF)
tfidf_vectorizer = TfidfVectorizer(max_features=1000) # Limit features for simplicity
tfidf_features = tfidf_vectorizer.fit_transform(df['processed_review']).toarray()
tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
tfidf_df = pd.DataFrame(tfidf_features, columns=tfidf_feature_names)

print("\nTF-IDF features shape:", tfidf_df.shape)

# 5. Feature Combination
# Combine numerical features and TF-IDF features
combined_features = pd.concat([df[num_cols], tfidf_df], axis=1)

print("\nCombined features shape:", combined_features.shape)
print("Combined features head:\n", combined_features.head())

# 6. Sentiment Analysis Model
X = combined_features
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = LogisticRegression(max_iter=1000, solver='liblinear')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nSentiment Analysis Model Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 7. Recommendation System (Conceptual Feature Preparation)
print("\nFeatures prepared for a Recommendation System (e.g., for similarity calculations):")
print("Each row in 'combined_features' represents a product vector that can be used for finding similar products.")

# Example: Find a product most similar to 'Laptop Pro' (product_id=1)
# This is a conceptual demonstration, actual similarity would use cosine similarity etc.
product_id_to_compare = 1
product_vector_index = df[df['product_id'] == product_id_to_compare].index[0]

print(f"\nVector for product_id {product_id_to_compare} ('{df.loc[product_vector_index, 'product_name']}') is ready for similarity calculation:")
print(combined_features.iloc[product_vector_index].head())
