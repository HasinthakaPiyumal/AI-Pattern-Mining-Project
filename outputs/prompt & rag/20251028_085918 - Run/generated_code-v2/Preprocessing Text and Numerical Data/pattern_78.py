import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download NLTK data (if not already downloaded)
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

class NLTKTextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, stem=True, lemmatize=False, remove_stopwords=True):
        self.stem = stem
        self.lemmatize = lemmatize
        self.remove_stopwords = remove_stopwords
        if self.stem:
            self.stemmer = PorterStemmer()
        if self.lemmatize:
            self.lemmatizer = WordNetLemmatizer()
        if self.remove_stopwords:
            self.stopwords = set(stopwords.words('english'))

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        processed_texts = []
        for text in X:
            tokens = word_tokenize(str(text).lower())
            if self.remove_stopwords:
                tokens = [word for word in tokens if word.isalpha() and word not in self.stopwords]
            if self.stem:
                tokens = [self.stemmer.stem(word) for word in tokens]
            if self.lemmatize:
                tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
            processed_texts.append(" ".join(tokens))
        return np.array(processed_texts)

# Sample Data
data = {
    'review_text': [
        'This product is amazing! I love it so much.',
        'Terrible quality, broke after one use. Very disappointed.',
        'It is okay, nothing special but gets the job done.',
        'The best thing I have bought this year. Highly recommend!',
        'Not worth the money. Customer service was also bad.',
        'Good value for money, fast delivery. Will buy again.',
        'Too expensive for what it is. Average performance.',
        np.nan
    ],
    'rating': [5, 1, 3, 5, 2, 4, 3, np.nan],
    'product_price': [25.99, 12.50, 45.00, 19.99, 75.00, 30.00, 55.00, 35.00],
    'purchase_frequency': ['high', 'low', 'medium', 'high', 'low', 'medium', 'medium', 'low']
}
df = pd.DataFrame(data)

# Define preprocessing pipelines

# Text Preprocessing Pipeline
text_preprocessing_pipeline = Pipeline([
    ('nltk_preprocessor', NLTKTextPreprocessor(stem=True, lemmatize=False, remove_stopwords=True)),
    ('tfidf_vectorizer', TfidfVectorizer(max_features=1000))
])

# Numerical Preprocessing Pipeline
numerical_preprocessing_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Categorical Numerical Preprocessing Pipeline (for 'purchase_frequency')
categorical_numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# ColumnTransformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('text_features', text_preprocessing_pipeline, 'review_text'),
        ('numerical_features', numerical_preprocessing_pipeline, ['rating', 'product_price']),
        ('categorical_features', categorical_numerical_pipeline, ['purchase_frequency'])
    ],
    remainder='drop' # Drop any columns not specified
)

# Fit and transform the data
preprocessed_features = preprocessor.fit_transform(df)

print("Original DataFrame head:")
print(df.head())
print("\nShape of preprocessed features:", preprocessed_features.shape)
print("\nExample of preprocessed features (first 5 rows, first 10 columns if available):\n", preprocessed_features[:5, :10].toarray() if hasattr(preprocessed_features, 'toarray') else preprocessed_features[:5, :10])

# To get feature names (more complex for ColumnTransformer, depends on output type)
def get_feature_names(column_transformer):
    output_features = []
    for name, estimator, columns in column_transformer.transformers_:
        if name == 'remainder':
            continue
        if hasattr(estimator, 'get_feature_names_out'):
            if isinstance(columns, str):
                col_names = [columns]
            else:
                col_names = list(columns)
            output_features.extend(estimator.get_feature_names_out(col_names))
        elif hasattr(estimator, 'steps'): # For pipelines
            if name == 'text_features':
                # TF-IDF vectorizer will have its own feature names
                tfidf_vectorizer = estimator.named_steps['tfidf_vectorizer']
                output_features.extend(tfidf_vectorizer.get_feature_names_out())
            elif name == 'numerical_features':
                # Numerical features will retain their names after scaling/imputation
                output_features.extend([f'{c}_scaled' for c in columns])
            elif name == 'categorical_features':
                # OneHotEncoder will have its own feature names
                onehot_encoder = estimator.named_steps['onehot']
                output_features.extend(onehot_encoder.get_feature_names_out(columns))
        else:
            # Default for simple transformers, might need refinement
            output_features.extend(columns)
    return output_features

try:
    feature_names = get_feature_names(preprocessor)
    print("\nExample Feature Names (first 10 if available):\n", feature_names[:10])
except Exception as e:
    print(f"\nCould not retrieve all feature names: {e}")