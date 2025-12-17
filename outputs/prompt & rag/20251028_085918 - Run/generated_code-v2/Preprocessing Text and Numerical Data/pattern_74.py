import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import nltk
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('taggers/averaged_perceptron_tagger')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')

class EcomProductRecommender:
    def __init__(self):
        self.text_preprocessor = None
        self.numerical_transformer = None
        self.categorical_transformer = None
        self.preprocessor = None
        self.tfidf_vectorizer = None
        self.product_features = None
        self.product_ids = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def _preprocess_text(self, text):
        if not isinstance(text, str): # Handle non-string input gracefully
            return ""
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        return " ".join(tokens)

    def fit(self, products_df, reviews_df):
        # Ensure product_id is the index for easier lookup
        products_df = products_df.set_index('product_id')
        self.product_ids = products_df.index.tolist()

        # Preprocess text descriptions
        products_df['processed_description'] = products_df['description'].apply(self._preprocess_text)

        # Combine product descriptions and reviews (for richer text features if needed)
        # For simplicity, let's just use product descriptions for now for item-item similarity
        # If reviews were to be incorporated, they would be aggregated per product.

        # TF-IDF Vectorization for text features
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        text_features_sparse = self.tfidf_vectorizer.fit_transform(products_df['processed_description'])
        text_feature_names = self.tfidf_vectorizer.get_feature_names_out()
        text_features_df = pd.DataFrame(text_features_sparse.toarray(), index=products_df.index, columns=text_feature_names)

        # Numerical and Categorical Preprocessing
        numerical_cols = ['price', 'sales', 'rating']
        categorical_cols = ['category']

        # Impute missing numerical values with mean, then scale
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        # Impute missing categorical values with most frequent, then one-hot encode
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ], 
            remainder='drop'
        )

        numerical_categorical_features_transformed = self.preprocessor.fit_transform(products_df)
        
        # Get feature names for numerical and one-hot encoded categorical features
        cat_feature_names = self.preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_cols)
        numerical_categorical_feature_names = numerical_cols + list(cat_feature_names)
        
        numerical_categorical_features_df = pd.DataFrame(
            numerical_categorical_features_transformed,
            index=products_df.index,
            columns=numerical_categorical_feature_names
        )

        # Combine all features
        self.product_features = pd.concat([text_features_df, numerical_categorical_features_df], axis=1)
        

    def recommend_products(self, product_id, num_recommendations=5):
        if product_id not in self.product_features.index:
            print(f"Product ID {product_id} not found.")
            return []

        target_product_features = self.product_features.loc[product_id].values.reshape(1, -1)
        
        # Calculate cosine similarity with all other products
        similarities = cosine_similarity(target_product_features, self.product_features)
        
        # Get indices of most similar products (excluding itself)
        similar_product_indices = similarities.argsort()[0][::-1]
        
        recommended_product_ids = []
        count = 0
        for idx in similar_product_indices:
            if self.product_features.index[idx] != product_id:
                recommended_product_ids.append(self.product_features.index[idx])
                count += 1
            if count >= num_recommendations:
                break
        
        return recommended_product_ids

# --- Demo Usage ---
if __name__ == "__main__":
    # Sample Data
    products_data = {
        'product_id': [1, 2, 3, 4, 5, 6],
        'description': [
            'Excellent smartphone with great camera and long battery life.',
            'Affordable tablet for students, good for reading and browsing.',
            'High-performance gaming laptop with powerful graphics card.',
            'Compact digital camera, perfect for travel and vlogging.',
            'Smartwatch with fitness tracking and heart rate monitor.',
            'Basic feature phone, durable and simple to use.'
        ],
        'price': [799.99, 249.00, 1499.50, 399.00, 199.99, 59.99],
        'sales': [1200, 3500, 500, 800, 2000, 1500],
        'rating': [4.5, 4.0, 4.7, 4.2, 4.3, 3.8],
        'category': ['Electronics', 'Electronics', 'Electronics', 'Electronics', 'Wearable', 'Electronics']
    }
    products_df = pd.DataFrame(products_data)

    reviews_data = {
        'product_id': [1, 1, 2, 3, 4, 5, 5],
        'review_text': [
            'Love this phone, camera is amazing!',
            'Battery lasts all day, very happy.',
            'Good for my online classes, a bit slow sometimes.',
            'Blazing fast for all my games, highly recommend.',
            'Takes decent photos, easy to carry around.',
            'Great for tracking my steps, comfortable to wear.',
            'Heart rate monitor is accurate, battery life is okay.'
        ]
    }
    reviews_df = pd.DataFrame(reviews_data)

    recommender = EcomProductRecommender()
    recommender.fit(products_df.copy(), reviews_df.copy()) # Use .copy() to avoid SettingWithCopyWarning

    print("\n--- Recommendations for Product ID 1 (Smartphone) ---")
    recommendations = recommender.recommend_products(product_id=1, num_recommendations=3)
    print(f"Recommended Product IDs: {recommendations}")

    print("\n--- Recommendations for Product ID 2 (Tablet) ---")
    recommendations = recommender.recommend_products(product_id=2, num_recommendations=3)
    print(f"Recommended Product IDs: {recommendations}")

    print("\n--- Recommendations for Product ID 5 (Smartwatch) ---")
    recommendations = recommender.recommend_products(product_id=5, num_recommendations=3)
    print(f"Recommended Product IDs: {recommendations}")

    print("\n--- Recommendations for non-existent Product ID 99 ---")
    recommendations = recommender.recommend_products(product_id=99, num_recommendations=3)
    print(f"Recommended Product IDs: {recommendations}")