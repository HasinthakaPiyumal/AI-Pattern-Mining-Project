import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

# Download NLTK resources (run once)
try:
    nltk.data.find("corpora/stopwords")
except nltk.downloader.DownloadError:
    nltk.download("stopwords")
try:
    nltk.data.find("tokenizers/punkt")
except nltk.downloader.DownloadError:
    nltk.download("punkt")
try:
    nltk.data.find("corpora/wordnet")
except nltk.downloader.DownloadError:
    nltk.download("wordnet")

# 1. Create a Sample DataFrame
data = {
    "product_id": [1, 2, 3, 4, 5, 6],
    "product_name": ["Laptop", "Smartphone", "Headphones", "Smartwatch", "Monitor", "Keyboard"],
    "product_category": ["Electronics", "Electronics", "Audio", "Wearable", "Electronics", "Peripherals"],
    "price": [1200.00, 800.00, 150.00, 250.00, 300.00, 75.00],
    "rating": [4.5, 3.8, 4.2, 4.0, 4.7, np.nan],
    "sales_volume": [1500, 2000, 800, 1200, np.nan, 900],
    "review_text": [
        "This laptop is amazing, fast and great for work.",
        "Smartphone is okay, but battery life is not good.",
        "Excellent sound quality and comfortable.",
        "Smartwatch has many features, a bit slow sometimes.",
        "Great monitor for gaming and everyday use, very clear display.",
        "Good keyboard for the price, keys feel a bit flimsy."
    ]
}
df = pd.DataFrame(data)

# Define numerical and categorical features
numerical_features = ["price", "rating", "sales_volume"]
categorical_features = ["product_category"]
text_feature = "review_text"

# 2. Numerical Data Preprocessing Pipeline
numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ], 
    remainder="passthrough"
)

# 3. Text Data Preprocessing Function
lemmatizer = WordNetLemmatizer()
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stopwords.words("english")]
    return " ".join(tokens)

df["preprocessed_review"] = df[text_feature].apply(preprocess_text)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=1000) # Limiting features for demonstration
tfidf_features = tfidf_vectorizer.fit_transform(df["preprocessed_review"])

# Apply numerical and categorical preprocessing
preprocessed_numerical_categorical = preprocessor.fit_transform(df)

# 4. Feature Combination
# Ensure both are sparse matrices or convert appropriately for hstack
if isinstance(preprocessed_numerical_categorical, np.ndarray):
    # Convert numpy array to sparse matrix if it's dense, for efficient hstacking with TF-IDF
    preprocessed_numerical_categorical = csr_matrix(preprocessed_numerical_categorical)

combined_features = hstack([preprocessed_numerical_categorical, tfidf_features])

print("Shape of combined features:", combined_features.shape)
print("\nFirst 5 rows of preprocessed numerical/categorical features (dense view):")
print(preprocessed_numerical_categorical.toarray()[:5])
print("\nExample of TF-IDF features (first 5 rows, first 10 columns):")
print(tfidf_features[:5, :10].toarray())

# The 'combined_features' matrix is now ready for model training
