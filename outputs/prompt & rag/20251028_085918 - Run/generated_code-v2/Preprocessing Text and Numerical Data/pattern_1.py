import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Ensure NLTK data is available (uncomment and run once if not already downloaded)
# try:
#     nltk.data.find('corpora/stopwords')
# except nltk.downloader.DownloadError:
#     nltk.download('stopwords')
# try:
#     nltk.data.find('corpora/wordnet')
# except nltk.downloader.DownloadError:
#     nltk.download('wordnet')
# try:
#     nltk.data.find('tokenizers/punkt')
# except nltk.downloader.DownloadError:
#     nltk.download('punkt')

# Simulate Data
def create_simulated_data():
    products_data = {
        "product_id": ["P001", "P002", "P003", "P004", "P005", "P006"],
        "product_name": [
            "Laptop Pro X",
            "Wireless Ergonomic Mouse",
            "4K Ultra HD Monitor",
            "Noise Cancelling Headphones",
            "Mechanical Gaming Keyboard",
            "Portable SSD Drive"
        ],
        "product_description": [
            "Powerful laptop with fast processor and ample storage for professionals.",
            "Comfortable mouse with customizable buttons for office work.",
            "Stunning display with vibrant colors for entertainment and productivity.",
            "Immersive audio experience with advanced noise cancellation.",
            "Durable keyboard with tactile switches for gamers.",
            "High-speed external solid state drive for data backup and transfer."
        ],
        "price": [1200.00, 35.50, 450.00, 199.99, 89.99, 150.00],
        "rating": [4.8, 4.2, 4.5, 4.7, 4.6, None]  # Missing rating for P006
    }
    products_df = pd.DataFrame(products_data)

    user_interactions_data = {
        "user_id": ["U001", "U001", "U002", "U003", "U001", "U002", "U003"],
        "product_id": ["P001", "P003", "P002", "P004", "P005", "P001", "P005"],
        "interaction_type": ["view", "purchase", "view", "add_to_cart", "view", "purchase", "purchase"],
        "timestamp": [
            "2023-01-01 10:00:00",
            "2023-01-01 10:30:00",
            "2023-01-02 11:00:00",
            "2023-01-03 12:00:00",
            "2023-01-04 13:00:00",
            "2023-01-05 14:00:00",
            "2023-01-06 15:00:00"
        ]
    }
    user_interactions_df = pd.DataFrame(user_interactions_data)

    return products_df, user_interactions_df

# Text Preprocessing Function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower() # Lowercasing
    text = re.sub(r'[^a-z0-9\s]', '', text) # Remove punctuation
    tokens = nltk.word_tokenize(text) # Tokenization
    tokens = [word for word in tokens if word not in stop_words] # Stop-word removal
    tokens = [lemmatizer.lemmatize(word) for word in tokens] # Lemmatization
    return " ".join(tokens)

# Main Preprocessing and Recommendation System
class ECommerceRecommendationSystem:
    def __init__(self):
        self.preprocessor = None
        self.product_features = None
        self.products_df = None

    def fit(self, products_df):
        self.products_df = products_df.copy()

        # Define numerical and text features
        numerical_features = ["price", "rating"]
        text_features = ["product_name", "product_description"]

        # Numerical Preprocessing Pipeline
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        # Text Preprocessing Pipeline
        # TfidfVectorizer can take a preprocessor function
        text_transformer = TfidfVectorizer(preprocessor=preprocess_text, stop_words='english', max_features=1000)

        # Create a ColumnTransformer to apply different transformations to different columns
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('text', text_transformer, text_features[0]), # Applying TF-IDF to product_name
                ('text_desc', text_transformer, text_features[1]) # Applying TF-IDF to product_description
            ],
            remainder='drop'
        )

        # Fit and transform the product data
        product_features_transformed = self.preprocessor.fit_transform(self.products_df)

        # Combine the features (simple concatenation for demonstration)
        # Note: TfidfVectorizer outputs sparse matrices, so we convert to array for concatenation
        num_features = self.preprocessor.named_transformers_['num'].transform(self.products_df[numerical_features])
        text_name_features = self.preprocessor.named_transformers_['text'].transform(self.products_df[text_features[0]]).toarray()
        text_desc_features = self.preprocessor.named_transformers_['text_desc'].transform(self.products_df[text_features[1]]).toarray()

        self.product_features = np.hstack((num_features, text_name_features, text_desc_features))

    def recommend_products(self, product_id, top_n=5):
        if product_id not in self.products_df['product_id'].values:
            return f"Product ID {product_id} not found."

        idx = self.products_df[self.products_df['product_id'] == product_id].index[0]
        target_product_features = self.product_features[idx].reshape(1, -1)

        # Calculate cosine similarity between the target product and all other products
        similarities = cosine_similarity(target_product_features, self.product_features).flatten()

        # Exclude the product itself from recommendations
        similarities[idx] = -1

        # Get top_n most similar product indices
        most_similar_indices = similarities.argsort()[-top_n:][::-1]

        recommended_products = self.products_df.iloc[most_similar_indices]
        return recommended_products[['product_id', 'product_name', 'price', 'rating']]

# Example Usage
if __name__ == "__main__":
    products_df, user_interactions_df = create_simulated_data()

    recommender = ECommerceRecommendationSystem()
    recommender.fit(products_df)

    print("Original Products DataFrame:")
    print(products_df)
    print("\n")

    # Get recommendations for a product (e.g., P001 - Laptop Pro X)
    product_to_recommend_for = "P001"
    recommendations = recommender.recommend_products(product_to_recommend_for)

    print(f"Recommendations for product {product_to_recommend_for} (Laptop Pro X):")
    print(recommendations)
    print("\n")

    # Get recommendations for a product with missing data (e.g., P006 - Portable SSD Drive)
    product_to_recommend_for_missing = "P006"
    recommendations_missing = recommender.recommend_products(product_to_recommend_for_missing)

    print(f"Recommendations for product {product_to_recommend_for_missing} (Portable SSD Drive):")
    print(recommendations_missing)
    print("\n")

    # Demonstrate the preprocessed features (for verification, not part of actual output)
    # product_features_df = pd.DataFrame(recommender.product_features)
    # print("Shape of preprocessed product features:", product_features_df.shape)
    # print(product_features_df.head())
