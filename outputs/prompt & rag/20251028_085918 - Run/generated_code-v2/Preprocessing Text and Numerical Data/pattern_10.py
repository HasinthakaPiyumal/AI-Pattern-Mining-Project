
import pandas as pd
import numpy as np
import re
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from sklearn.neighbors import NearestNeighbors
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

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

class PreprocessingPipeline:
    def __init__(self):
        self.numerical_imputer = SimpleImputer(strategy='mean')
        self.numerical_scaler = StandardScaler()
        self.one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def preprocess_numerical_data(self, df, numerical_cols, categorical_cols):
        # Impute numerical columns
        df[numerical_cols] = self.numerical_imputer.fit_transform(df[numerical_cols])

        # Scale numerical columns
        df[numerical_cols] = self.numerical_scaler.fit_transform(df[numerical_cols])

        # One-hot encode categorical columns
        if categorical_cols:
            encoded_features = self.one_hot_encoder.fit_transform(df[categorical_cols]).toarray()
            encoded_feature_names = self.one_hot_encoder.get_feature_names_out(categorical_cols)
            encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=df.index)
            df = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)
        return df

    def preprocess_text_data(self, series):
        processed_texts = []
        sentiments = []
        for text in series:
            # Clean and tokenize
            text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
            tokens = word_tokenize(text)
            
            # Stop-word removal and lemmatization
            filtered_tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
            processed_text = ' '.join(filtered_tokens)
            processed_texts.append(processed_text)
            
            # Sentiment analysis
            analysis = TextBlob(text)
            sentiments.append(analysis.sentiment.polarity) # Use polarity as a numerical feature
        
        # TF-IDF Vectorization
        tfidf_vectors = self.tfidf_vectorizer.fit_transform(processed_texts).toarray()
        tfidf_df = pd.DataFrame(tfidf_vectors, columns=self.tfidf_vectorizer.get_feature_names_out())
        
        sentiment_df = pd.DataFrame(sentiments, columns=['sentiment_polarity'])
        
        return tfidf_df, sentiment_df

class RecommendationSystem:
    def __init__(self):
        self.model = NearestNeighbors(n_neighbors=5, metric='cosine')
        self.product_data = None
        self.feature_matrix = None

    def train(self, preprocessed_features, product_data):
        self.feature_matrix = preprocessed_features
        self.product_data = product_data
        self.model.fit(self.feature_matrix)

    def recommend_products(self, product_id, n_recommendations=5):
        if product_id not in self.product_data['product_id'].values:
            print(f"Product ID {product_id} not found.")
            return pd.DataFrame()
        
        product_idx = self.product_data[self.product_data['product_id'] == product_id].index[0]
        distances, indices = self.model.kneighbors(self.feature_matrix.iloc[product_idx].values.reshape(1, -1), n_neighbors=n_recommendations+1)
        
        # Exclude the product itself
        recommended_product_indices = indices.flatten()[1:]
        recommended_products = self.product_data.iloc[recommended_product_indices].copy()
        recommended_products['distance'] = distances.flatten()[1:]
        return recommended_products.sort_values(by='distance')


# --- Main execution --- #
if __name__ == "__main__":
    # 1. Sample Data Ingestion & Storage
    product_data = pd.DataFrame({
        'product_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'price': [50.0, 120.0, 30.0, 80.0, 200.0, 60.0, 150.0, 45.0, 90.0, 110.0],
        'rating': [4.5, 3.8, 4.9, 4.2, 3.5, 4.7, 4.0, 4.6, 3.9, 4.3],
        'sales': [100, 50, 200, 75, 30, 150, 60, 120, 80, 90],
        'category_id': ['electronics', 'clothing', 'electronics', 'home', 'clothing', 'home', 'electronics', 'clothing', 'home', 'electronics']
    })

    review_data = pd.DataFrame({
        'user_id': [101, 102, 103, 104, 105, 101, 102, 103, 104, 105],
        'product_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'review_text': [
            'Great product, very happy with the purchase. Highly recommend!',
            'Not bad, but a bit overpriced for what it offers. Average quality.',
            'Absolutely amazing! Best gadget ever. Fast shipping too.',
            'It does the job, nothing spectacular. Customer service was helpful though.',
            'Disappointing quality, broke after a week. Would not buy again.',
            'Good value for money. Works as expected.',
            'Excellent performance, a bit expensive but worth it.',
            'Decent item for the price. Fast delivery.',
            'Okay, but not revolutionary. Could be better.',
            'Love this! Makes my life so much easier. Fantastic!'
        ],
        'rating': [5, 3, 5, 4, 1, 4, 5, 3, 2, 5]
    })
    
    # Merge dataframes for combined feature creation
    # For this example, we'll assume a 1:1 mapping of product_id to a single representative review.
    # In a real system, multiple reviews per product would be aggregated or processed differently.
    combined_data = pd.merge(product_data, review_data[['product_id', 'review_text']], on='product_id', how='left')

    # 2. Preprocessing Module
    pipeline = PreprocessingPipeline()
    
    # Preprocess numerical data
    numerical_cols = ['price', 'rating', 'sales']
    categorical_cols = ['category_id']
    preprocessed_numerical_df = pipeline.preprocess_numerical_data(combined_data.copy(), numerical_cols, categorical_cols)

    # Preprocess text data
    tfidf_df, sentiment_df = pipeline.preprocess_text_data(combined_data['review_text'].fillna(''))
    
    # Ensure indices align after text preprocessing
    tfidf_df.index = combined_data.index
    sentiment_df.index = combined_data.index

    # 3. Feature Engineering: Combine all features
    # Drop original numerical and categorical columns from preprocessed_numerical_df as they are now transformed
    features_to_drop = numerical_cols + categorical_cols
    # Filter out columns that might not exist if categorical_cols was empty or already dropped in preprocess_numerical_data
    preprocessed_numerical_df_cleaned = preprocessed_numerical_df.drop(columns=[col for col in features_to_drop if col in preprocessed_numerical_df.columns], errors='ignore')

    # Re-align indices for concatenation
    preprocessed_numerical_df_cleaned = preprocessed_numerical_df_cleaned.reset_index(drop=True)
    tfidf_df = tfidf_df.reset_index(drop=True)
    sentiment_df = sentiment_df.reset_index(drop=True)

    # Remove 'product_id' before concatenation for feature matrix, keep it for mapping
    product_ids_for_mapping = preprocessed_numerical_df_cleaned['product_id']
    preprocessed_numerical_df_cleaned = preprocessed_numerical_df_cleaned.drop(columns=['product_id'], errors='ignore')

    # Concatenate all preprocessed features
    final_feature_matrix = pd.concat([
        preprocessed_numerical_df_cleaned,
        tfidf_df,
        sentiment_df
    ], axis=1)
    
    # Re-insert product_id for the recommendation system to use
    product_data_with_features = product_data.copy()
    product_data_with_features['feature_index'] = final_feature_matrix.index # Map original product to its feature row

    # 4. Recommendation Model
    recommender = RecommendationSystem()
    recommender.train(final_feature_matrix, product_data_with_features)

    # Get recommendations for a specific product
    product_to_recommend_for = 1
    print(f"\nRecommendations for Product ID {product_to_recommend_for}:")
    recommendations = recommender.recommend_products(product_to_recommend_for)
    if not recommendations.empty:
        print(recommendations[['product_id', 'price', 'rating', 'review_text', 'distance']])
    else:
        print("No recommendations found.")

    product_to_recommend_for = 5
    print(f"\nRecommendations for Product ID {product_to_recommend_for}:")
    recommendations = recommender.recommend_products(product_to_recommend_for)
    if not recommendations.empty:
        print(recommendations[['product_id', 'price', 'rating', 'review_text', 'distance']])
    else:
        print("No recommendations found.")
