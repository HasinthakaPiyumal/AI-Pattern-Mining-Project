import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK data (run once)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")

class DataGenerator:
    def generate_synthetic_data(self, num_products=100, num_reviews_per_product=10):
        product_ids = [f"P{i:03d}" for i in range(num_products)]
        product_names = [f"Product {i}" for i in range(num_products)]
        
        data = []
        for pid, pname in zip(product_ids, product_names):
            for _ in range(num_reviews_per_product):
                rating = np.random.randint(1, 6)
                price = round(np.random.uniform(10, 500), 2)
                purchase_count = np.random.randint(1, 100)
                
                review_templates = {
                    "positive": [
                        f"This {pname} is amazing! Highly recommend. The quality is great.",
                        f"Loved the {pname}. It exceeded my expectations. Very happy with the purchase.",
                        f"Excellent {pname}. Works perfectly and arrived quickly.",
                        f"Fantastic value for the {pname}. A must-buy!",
                        f"So impressed with this {pname}. It's exactly what I needed."
                    ],
                    "negative": [
                        f"Disappointed with the {pname}. It broke after a week. Poor quality.",
                        f"The {pname} is not what I expected. Very low quality and overpriced.",
                        f"Had issues with this {pname} from day one. Would not recommend.",
                        f"Terrible experience with the {pname}. Complete waste of money.",
                        f"Very frustrated with this {pname}. It doesn't work as advertised."
                    ]
                }
                
                sentiment_label = "positive" if rating >= 4 and np.random.rand() < 0.9 else "negative"
                review_text = np.random.choice(review_templates[sentiment_label])
                
                data.append({"product_id": pid, "product_name": pname, "rating": rating,
                             "price": price, "purchase_count": purchase_count,
                             "review_text": review_text, "sentiment": sentiment_label})
                
        return pd.DataFrame(data)

class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000)

    def preprocess_text(self, text):
        tokens = word_tokenize(text.lower())
        filtered_tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in self.stop_words]
        return " ".join(filtered_tokens)

    def fit_transform_tfidf(self, texts):
        processed_texts = [self.preprocess_text(text) for text in texts]
        return self.tfidf_vectorizer.fit_transform(processed_texts)

    def transform_tfidf(self, texts):
        processed_texts = [self.preprocess_text(text) for text in texts]
        return self.tfidf_vectorizer.transform(processed_texts)

class NumericalPreprocessor:
    def __init__(self):
        self.imputer = SimpleImputer(strategy="mean")
        self.scaler = StandardScaler()

    def fit_transform(self, data):
        imputed_data = self.imputer.fit_transform(data)
        scaled_data = self.scaler.fit_transform(imputed_data)
        return scaled_data

    def transform(self, data):
        imputed_data = self.imputer.transform(data)
        scaled_data = self.scaler.transform(imputed_data)
        return scaled_data

class SentimentAnalyzer:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)

class ProductRecommender:
    def __init__(self):
        pass

    def get_recommendations(self, product_id, product_features_scaled_df, product_df, top_n=5):
        # Ensure product_id exists and get its features
        if product_id not in product_features_scaled_df.index:
            return pd.DataFrame()
        
        target_product_features = product_features_scaled_df.loc[product_id].values.reshape(1, -1)
        
        # Calculate cosine similarity with all products
        similarities = cosine_similarity(target_product_features, product_features_scaled_df)
        
        # Get product indices sorted by similarity (descending)
        # Exclude the product itself
        similar_product_indices = similarities[0].argsort()[::-1]
        
        # Map indices back to product_ids and filter out the target product
        recommended_product_ids = [product_features_scaled_df.index[i] for i in similar_product_indices if product_features_scaled_df.index[i] != product_id]
        
        # Get the top N recommendations
        top_recommendations_ids = recommended_product_ids[:top_n]
        
        # Retrieve details for recommended products
        recommendations = product_df[product_df["product_id"].isin(top_recommendations_ids)].drop_duplicates(subset="product_id")
        
        return recommendations

if __name__ == "__main__":
    print("Starting E-commerce System with Sentiment Analysis and Product Recommendation...")

    # 1. Data Generation/Simulation Module
    print("\n1. Generating synthetic E-commerce data...")
    data_generator = DataGenerator()
    df = data_generator.generate_synthetic_data(num_products=50, num_reviews_per_product=5)
    print(f"Generated {len(df)} records.")
    print("Sample data head:")
    print(df.head())

    # 2. Text Preprocessing Module (for Sentiment Analysis)
    print("\n2. Preprocessing text data for sentiment analysis...")
    text_preprocessor = TextPreprocessor()
    X_tfidf = text_preprocessor.fit_transform_tfidf(df["review_text"])
    y_sentiment = df["sentiment"].apply(lambda x: 1 if x == "positive" else 0)
    
    X_train_tfidf, X_test_tfidf, y_train_sentiment, y_test_sentiment = train_test_split(X_tfidf, y_sentiment, test_size=0.2, random_state=42)
    print(f"TF-IDF features shape: {X_tfidf.shape}")
    print(f"Train set size: {X_train_tfidf.shape[0]}, Test set size: {X_test_tfidf.shape[0]}")

    # 3. Numerical Data Preprocessing Module (for Product Recommendation)
    print("\n3. Preprocessing numerical data for product recommendation...")
    numerical_features = df[["rating", "price", "purchase_count"]].copy()
    
    # Aggregate numerical features per product for recommendation
    product_numerical_features = numerical_features.groupby(df["product_id"]).mean()
    
    numerical_preprocessor = NumericalPreprocessor()
    product_features_scaled = numerical_preprocessor.fit_transform(product_numerical_features)
    
    # Convert back to DataFrame for easier indexing
    product_features_scaled_df = pd.DataFrame(product_features_scaled, 
                                              columns=product_numerical_features.columns, 
                                              index=product_numerical_features.index)
    print(f"Scaled numerical features shape for products: {product_features_scaled_df.shape}")
    print("Sample scaled product features head:")
    print(product_features_scaled_df.head())

    # 4. Sentiment Analysis Model Module
    print("\n4. Training Sentiment Analysis Model...")
    sentiment_analyzer = SentimentAnalyzer()
    sentiment_analyzer.train(X_train_tfidf, y_train_sentiment)
    y_pred_sentiment = sentiment_analyzer.predict(X_test_tfidf)
    accuracy = sentiment_analyzer.evaluate(y_test_sentiment, y_pred_sentiment)
    print(f"Sentiment Model Accuracy: {accuracy:.4f}")

    # Example sentiment prediction for a new review
    sample_review = "This product is absolutely fantastic! I love it."
    processed_sample_review = text_preprocessor.transform_tfidf([sample_review])
    predicted_sentiment = sentiment_analyzer.predict(processed_sample_review)[0]
    print(f"Sample review: '{sample_review}' -> Predicted Sentiment: {'Positive' if predicted_sentiment == 1 else 'Negative'}")

    sample_review_neg = "This item was terrible and broke quickly. Very disappointed."
    processed_sample_review_neg = text_preprocessor.transform_tfidf([sample_review_neg])
    predicted_sentiment_neg = sentiment_analyzer.predict(processed_sample_review_neg)[0]
    print(f"Sample review: '{sample_review_neg}' -> Predicted Sentiment: {'Positive' if predicted_sentiment_neg == 1 else 'Negative'}")

    # 5. Product Recommendation Model Module
    print("\n5. Generating Product Recommendations...")
    recommender = ProductRecommender()
    
    # Choose a sample product_id for recommendation
    sample_product_id = df["product_id"].sample(1).iloc[0]
    print(f"Getting recommendations for product_id: {sample_product_id} (Product Name: {df[df['product_id'] == sample_product_id]['product_name'].iloc[0]})\n")
    
    recommendations = recommender.get_recommendations(sample_product_id, product_features_scaled_df, df, top_n=5)
    
    if not recommendations.empty:
        print("Recommended Products:")
        for index, row in recommendations.iterrows():
            print(f"  - Product ID: {row['product_id']}, Name: {row['product_name']}, Avg Rating: {product_numerical_features.loc[row['product_id']]['rating']:.2f}, Avg Price: {product_numerical_features.loc[row['product_id']]['price']:.2f}")
    else:
        print(f"No recommendations found for product_id: {sample_product_id}.")

    print("\nE-commerce System demonstration complete.")