
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class DataPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        self.numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        self.categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.text_vectorizer = TfidfVectorizer(stop_words='english', min_df=5)

        self.preprocessor = None
        self.text_columns = None

    def _preprocess_text(self, text):
        if isinstance(text, float) and np.isnan(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = nltk.word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        return " ".join(tokens)

    def fit(self, df, numerical_cols, categorical_cols, text_cols):
        self.text_columns = text_cols
        df_copy = df.copy()

        for col in self.text_columns:
            df_copy[col] = df_copy[col].apply(self._preprocess_text)

        combined_text = df_copy[self.text_columns].agg(' '.join, axis=1)
        self.text_vectorizer.fit(combined_text)

        transformers = []
        if numerical_cols:
            transformers.append(('num', self.numerical_transformer, numerical_cols))
        if categorical_cols:
            transformers.append(('cat', self.categorical_transformer, categorical_cols))

        self.preprocessor = ColumnTransformer(transformers, remainder='passthrough')
        self.preprocessor.fit(df)

    def transform(self, df):
        df_copy = df.copy()

        for col in self.text_columns:
            df_copy[col] = df_copy[col].apply(self._preprocess_text)

        combined_text = df_copy[self.text_columns].agg(' '.join, axis=1)
        text_features = self.text_vectorizer.transform(combined_text)

        numerical_categorical_features = self.preprocessor.transform(df_copy)

        return np.hstack((numerical_categorical_features, text_features.toarray()))

class RecommendationSystem:
    def __init__(self, data_preprocessor, products_df):
        self.data_preprocessor = data_preprocessor
        self.products_df = products_df
        self.feature_matrix = None

    def train(self):
        numerical_cols = ['price', 'rating', 'num_reviews']
        categorical_cols = ['category']
        text_cols = ['name', 'description']

        self.data_preprocessor.fit(self.products_df, numerical_cols, categorical_cols, text_cols)
        self.feature_matrix = self.data_preprocessor.transform(self.products_df)

    def get_recommendations(self, product_id, top_n=5):
        if self.feature_matrix is None:
            raise RuntimeError("Recommendation system not trained. Call 'train()' first.")

        if product_id not in self.products_df['product_id'].values:
            print(f"Product ID {product_id} not found.")
            return pd.DataFrame()

        idx = self.products_df[self.products_df['product_id'] == product_id].index[0]
        product_features = self.feature_matrix[idx]

        similarities = cosine_similarity([product_features], self.feature_matrix)[0]

        similar_indices = similarities.argsort()[-top_n-1:-1][::-1]
        similar_products = self.products_df.iloc[similar_indices]
        similar_products['similarity'] = similarities[similar_indices]

        return similar_products

if __name__ == '__main__':
    # 1. Dummy Data Creation
    data = {
        'product_id': [1, 2, 3, 4, 5, 6, 7],
        'name': [
            'Smartphone X',
            'Laptop Pro',
            'Smartwatch 3',
            'Bluetooth Speaker',
            'Gaming Headset',
            'Fitness Tracker',
            'Wireless Earbuds'
        ],
        'description': [
            'A powerful smartphone with a great camera and long battery life.',
            'High-performance laptop for professionals and gamers alike.',
            'Next-gen smartwatch with health tracking and NFC payments.',
            'Portable speaker with rich bass and 10-hour battery.',
            'Immersive gaming experience with noise-cancelling microphone.',
            'Track your daily steps, heart rate, and sleep patterns.',
            'Compact and comfortable earbuds with clear audio.'
        ],
        'price': [799.99, 1200.00, 249.99, 89.99, 149.99, 79.99, 129.99],
        'rating': [4.5, 4.7, 4.2, 4.0, 4.6, 3.9, 4.3],
        'num_reviews': [1500, 800, 300, 500, 200, 100, 400],
        'category': [
            'Electronics',
            'Electronics',
            'Wearable',
            'Audio',
            'Audio',
            'Wearable',
            'Audio'
        ]
    }
    products_df = pd.DataFrame(data)

    # 2. Instantiate and Fit DataPreprocessor
    preprocessor = DataPreprocessor()

    # 3. Instantiate and Train RecommendationSystem
    recommender = RecommendationSystem(preprocessor, products_df)
    recommender.train()

    # 4. Get Recommendations for a specific product
    product_to_recommend_for = 1  # Example: Smartphone X
    top_n_recommendations = 3

    print(f"\nRecommendations for Product ID {product_to_recommend_for} ({products_df[products_df['product_id'] == product_to_recommend_for]['name'].iloc[0]}):")
    recommendations = recommender.get_recommendations(product_to_recommend_for, top_n=top_n_recommendations)

    if not recommendations.empty:
        for index, row in recommendations.iterrows():
            print(f"  - {row['name']} (Category: {row['category']}, Price: ${row['price']:.2f}, Similarity: {row['similarity']:.4f})")
    else:
        print("No recommendations found.")

    product_to_recommend_for = 3  # Example: Smartwatch 3
    print(f"\nRecommendations for Product ID {product_to_recommend_for} ({products_df[products_df['product_id'] == product_to_recommend_for]['name'].iloc[0]}):")
    recommendations = recommender.get_recommendations(product_to_recommend_for, top_n=top_n_recommendations)

    if not recommendations.empty:
        for index, row in recommendations.iterrows():
            print(f"  - {row['name']} (Category: {row['category']}, Price: ${row['price']:.2f}, Similarity: {row['similarity']:.4f})")
    else:
        print("No recommendations found.")
