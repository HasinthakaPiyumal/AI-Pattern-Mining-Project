import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re

# Download NLTK data (run once)
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


# 1. Simulate Data Loading
data = {
    'product_id': range(1, 11),
    'product_name': [
        'Laptop Pro X', 'Gaming Mouse', 'Ergonomic Keyboard', '4K Monitor', 'Webcam HD',
        'Smart Speaker', 'Fitness Tracker', 'Wireless Earbuds', 'Portable SSD', 'USB Hub'
    ],
    'description': [
        'High-performance laptop with 16GB RAM and fast SSD. Ideal for professionals.',
        'Precision gaming mouse with customizable RGB lighting and high DPI.',
        'Comfortable mechanical keyboard with silent switches. Perfect for long typing sessions.',
        'Stunning 4K resolution monitor for immersive viewing. Great for design and gaming.',
        'Full HD 1080p webcam with built-in microphone. Excellent for video calls.',
        'Voice-controlled smart speaker with impressive sound quality. Integrates with smart home.',
        'Monitor your health with this fitness tracker. Tracks steps, heart rate, and sleep.',
        'Premium wireless earbuds with noise cancellation and long battery life.',
        'Compact and fast portable SSD for quick data transfers. 1TB storage.',
        'Multi-port USB hub with fast charging. Expands connectivity options.'
    ],
    'price': [1200.00, 49.99, 75.50, 350.00, 29.99, 99.00, 55.00, 129.99, 150.00, None],
    'rating': [4.7, 4.5, 4.2, 4.8, None, 4.6, 4.1, 4.3, 4.7, 3.9]
}
df = pd.DataFrame(data)

# Add some missing values for demonstration
df.loc[4, 'rating'] = np.nan
df.loc[9, 'price'] = np.nan

# 2. Numerical Preprocessing
# Define numerical columns
numerical_cols = ['price', 'rating']

# Imputation
imputer = SimpleImputer(strategy='median')
df[numerical_cols] = imputer.fit_transform(df[numerical_cols])

# Scaling
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# 3. Text Preprocessing
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()  # Lowercasing
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove special characters
    tokens = nltk.word_tokenize(text)  # Tokenization
    tokens = [word for word in tokens if word not in stop_words]  # Stop-word removal
    # Lemmatization (can choose between stemming or lemmatization, here using lemmatization)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

df['processed_description'] = df['description'].apply(preprocess_text)

# Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=100)  # Limiting features for example
text_features = tfidf_vectorizer.fit_transform(df['processed_description']).toarray()

# Convert text_features to DataFrame for easier concatenation
text_features_df = pd.DataFrame(text_features, index=df.index)

# 4. Feature Combination
# Drop original text and numerical columns, keep preprocessed ones
preprocessed_df = df.drop(columns=['product_name', 'description', 'processed_description'])

# Concatenate numerical and text features
final_features = pd.concat([preprocessed_df[numerical_cols], text_features_df], axis=1)

# 5. Recommendation Model (Example)
# Fit Nearest Neighbors model on the final features
model_knn = NearestNeighbors(n_neighbors=3, algorithm='brute', metric='cosine')
model_knn.fit(final_features)

def get_recommendations(product_id, data_frame, model, features_df, num_recommendations=2):
    product_index = data_frame[data_frame['product_id'] == product_id].index[0]
    distances, indices = model.kneighbors(features_df.iloc[product_index].values.reshape(1, -1), n_neighbors=num_recommendations + 1)

    # Get product IDs of recommended items (excluding itself)
    recommended_product_indices = indices.flatten()[1:]
    recommended_products = data_frame.iloc[recommended_product_indices]

    print(f"Recommendations for '{data_frame.loc[product_index, 'product_name']}':")
    for i, row in recommended_products.iterrows():
        print(f"  - {row['product_name']} (ID: {row['product_id']})")

# Example usage of the recommendation system
get_recommendations(product_id=1, data_frame=df, model=model_knn, features_df=final_features)
get_recommendations(product_id=6, data_frame=df, model=model_knn, features_df=final_features)
