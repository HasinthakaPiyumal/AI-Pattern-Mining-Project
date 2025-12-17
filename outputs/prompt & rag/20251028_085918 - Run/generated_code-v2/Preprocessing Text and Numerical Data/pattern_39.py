import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import NMF
from sklearn.metrics import classification_report
from scipy.sparse import hstack

# Download NLTK resources
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

# 1. Data Ingestion (Simulated Data)
def load_simulated_data():
    data = {
        'product_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'star_rating': [5, 4, 2, 5, 3, 1, 4, 5, 2, 3],
        'price': [25.99, 12.50, np.nan, 89.00, 15.75, 5.00, 30.00, 45.00, 10.00, 20.00],
        'review_text': [
            "This product is amazing! Highly recommend.",
            "Good value for money, but could be better.",
            "Very disappointed, broke after a week.",
            "Excellent quality and fast shipping.",
            "It's okay, nothing special. A bit overpriced.",
            "Worst purchase ever, completely useless.",
            "Works as expected, happy with the purchase.",
            "Absolutely love it! Perfect for my needs.",
            "Poor design and difficult to use.",
            "Decent product, but I've seen better."
        ],
        'sentiment': ['positive', 'neutral', 'negative', 'positive', 'neutral', 'negative', 'positive', 'positive', 'negative', 'neutral']
    }
    return pd.DataFrame(data)

df = load_simulated_data()

# Separate features and target
X_num = df[['star_rating', 'price']]
X_text = df['review_text']
y = df['sentiment']

# 2. Preprocessing Numerical Data
# Imputation
num_imputer = SimpleImputer(strategy='mean')
X_num_imputed = num_imputer.fit_transform(X_num)

# Scaling
scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(X_num_imputed)

# 3. Preprocessing Text Data
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(tokens)

X_text_preprocessed = X_text.apply(preprocess_text)

# Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=1000) # Limiting features for this example
X_text_vectorized = tfidf_vectorizer.fit_transform(X_text_preprocessed)

# 4. Feature Engineering & Integration
# Convert scaled numerical features to a sparse matrix to concatenate with TF-IDF
X_num_scaled_sparse = pd.DataFrame(X_num_scaled).sparse.from_spmatrix(X_num_scaled)
X_combined = hstack([X_num_scaled_sparse, X_text_vectorized])

# Encode target variable
sentiment_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
y_encoded = y.map(sentiment_mapping)

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)

# 5. Machine Learning Models
# Sentiment Analysis Model
sentiment_model = LogisticRegression(max_iter=1000, random_state=42)
sentiment_model.fit(X_train, y_train)

# Aspect-Based Feature Extraction (using NMF on original TF-IDF features)
n_components = 3 # Number of aspects to discover
nmf_model = NMF(n_components=n_components, random_state=42, init='nndsvda', tol=5e-3, max_iter=200)
# Fit NMF on the text features before splitting for simplicity in this example
# In a real pipeline, NMF would be fit on the training text data only
X_text_nmf_features = nmf_model.fit_transform(X_text_vectorized)

feature_names = tfidf_vectorizer.get_feature_names_out()
def display_topics(model, feature_names, no_top_words):
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-no_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        topics[f"Topic {topic_idx}"] = ", ".join(top_features)
    return topics

# 6. Model Training and Evaluation
# Evaluate Sentiment Model
y_pred = sentiment_model.predict(X_test)
print("\n--- Sentiment Analysis Model Performance ---")
print(classification_report(y_test, y_pred, target_names=sentiment_mapping.keys()))

# Display Aspect-Based Features (Topics)
print("\n--- Aspect-Based Feature Extraction (NMF Topics) ---")
discovered_topics = display_topics(nmf_model, feature_names, 5)
for topic_name, words in discovered_topics.items():
    print(f"{topic_name}: {words}")

print("\n--- Example Usage ---")
# Simulate new data for prediction
new_review_text = ["This phone is great, but the battery life is short.", "The quality is horrible, waste of money."]
new_star_rating = [4, 1]
new_price = [700.00, 15.00]

new_data_df = pd.DataFrame({
    'star_rating': new_star_rating,
    'price': new_price,
    'review_text': new_review_text
})

# Preprocess new numerical data
new_num_imputed = num_imputer.transform(new_data_df[['star_rating', 'price']])
new_num_scaled = scaler.transform(new_num_imputed)
new_num_scaled_sparse = pd.DataFrame(new_num_scaled).sparse.from_spmatrix(new_num_scaled)

# Preprocess new text data
new_text_preprocessed = new_data_df['review_text'].apply(preprocess_text)
new_text_vectorized = tfidf_vectorizer.transform(new_text_preprocessed)

# Combine new features
new_combined_features = hstack([new_num_scaled_sparse, new_text_vectorized])

# Predict sentiment
new_sentiment_predictions = sentiment_model.predict(new_combined_features)
reverse_sentiment_mapping = {v: k for k, v in sentiment_mapping.items()}
predicted_sentiments = [reverse_sentiment_mapping[pred] for pred in new_sentiment_predictions]

print(f"\nNew Reviews: {new_review_text}")
print(f"Predicted Sentiments: {predicted_sentiments}")

# Extract aspects for new reviews
new_text_nmf_features = nmf_model.transform(new_text_vectorized)
print("\nAspects for New Reviews (Top 2 components):")
for i, review in enumerate(new_review_text):
    print(f"Review: '{review}'")
    top_aspect_idx = np.argsort(new_text_nmf_features[i])[::-1][0] # Get index of highest scoring aspect
    print(f"  Most dominant aspect: {discovered_topics[f'Topic {top_aspect_idx}']}")