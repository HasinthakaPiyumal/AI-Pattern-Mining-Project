import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^a-z ]", "", text) # Remove special characters and numbers, keep spaces
    tokens = word_tokenize(text)
    stopwords_list = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stopwords_list]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

class ECommerceMLSystem:
    def __init__(self):
        self.product_data = None
        self.review_data = None
        self.preprocessor = None
        self.sentiment_model = None
        self.tfidf_vectorizer = None
        self.sentence_transformer_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.product_features_matrix = None
        self.product_ids = None

    def load_data(self):
        # Simulate loading product data
        products_data = {
            "ProductID": ["P1", "P2", "P3", "P4", "P5"],
            "Name": ["Laptop X", "Smartphone Y", "Headphones Z", "Smartwatch A", "Tablet B"],
            "Category": ["Electronics", "Electronics", "Audio", "Wearable", "Electronics"],
            "Price": [1200.00, 700.00, 150.00, 250.00, 400.00],
            "Rating": [4.5, 4.2, 3.8, 4.0, 4.1],
            "Description": [
                "Powerful laptop for work and gaming with high-resolution display.",
                "Latest smartphone with excellent camera and long battery life.",
                "Comfortable over-ear headphones with noise cancellation.",
                "Fitness tracker and smartwatch with heart rate monitor.",
                "Versatile tablet for entertainment and productivity on the go."
            ]
        }
        self.product_data = pd.DataFrame(products_data)

        # Simulate loading customer review data
        reviews_data = {
            "ReviewID": ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11", "R12"],
            "ProductID": ["P1", "P1", "P2", "P3", "P3", "P4", "P5", "P1", "P2", "P4", "P5", "P2"],
            "UserID": ["U1", "U2", "U3", "U4", "U5", "U6", "U7", "U8", "U9", "U10", "U11", "U12"],
            "Review Text": [
                "This laptop is amazing, super fast!",
                "Screen quality is good but a bit heavy.",
                "Love this phone, camera is incredible.",
                "Sound is okay, not the best bass.",
                "Very comfortable and great for daily use.",
                "Battery life is impressive for a smartwatch.",
                "Good tablet for the price, a bit slow sometimes.",
                "Best laptop I've ever owned. Highly recommend.",
                "Phone is fast, but the price is too high.",
                "Stylish watch, but software is a bit buggy.",
                "Perfect for reading and browsing. Portable.",
                "Excellent display and performance."
            ],
            "Star Rating": [5, 4, 5, 3, 4, 5, 3, 5, 3, 2, 4, 5]
        }
        self.review_data = pd.DataFrame(reviews_data)
        self.product_ids = self.product_data["ProductID"].tolist()

    def train_preprocessing_pipelines(self):
        # Numerical and Categorical features for products
        numerical_features = ["Price", "Rating"]
        categorical_features = ["Category"]

        numerical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, numerical_features),
                ("cat", categorical_transformer, categorical_features)
            ],
            remainder="passthrough"
        )

        # Fit on product data (excluding text for now, as text embeddings are handled separately)
        self.preprocessor.fit(self.product_data[numerical_features + categorical_features])

        # Prepare review texts for sentiment analysis (TF-IDF)
        self.review_data["Processed_Review_Text"] = self.review_data["Review Text"].apply(preprocess_text)
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000) # Limit features for simplicity
        self.tfidf_vectorizer.fit(self.review_data["Processed_Review_Text"])

    def train_sentiment_model(self):
        if self.tfidf_vectorizer is None or self.review_data is None:
            raise ValueError("Preprocessing pipelines not trained. Call train_preprocessing_pipelines first.")

        X_sentiment = self.tfidf_vectorizer.transform(self.review_data["Processed_Review_Text"])
        # Map Star Rating to a simple sentiment (e.g., >3 positive, <=3 negative/neutral)
        y_sentiment = self.review_data["Star Rating"].apply(lambda x: 1 if x > 3 else 0)

        self.sentiment_model = LogisticRegression(max_iter=1000, random_state=42)
        self.sentiment_model.fit(X_sentiment, y_sentiment)

    def generate_product_feature_vectors(self):
        if self.preprocessor is None or self.sentiment_model is None or self.product_data is None or self.review_data is None:
            raise ValueError("System not fully trained/initialized.")

        # 1. Numerical and Categorical Features
        product_num_cat_features = self.preprocessor.transform(self.product_data)

        # 2. Product Description Embeddings (SentenceTransformer)
        product_descriptions_processed = self.product_data["Description"].apply(preprocess_text).tolist()
        product_description_embeddings = self.sentence_transformer_model.encode(product_descriptions_processed, show_progress_bar=False)

        # 3. Aggregated Review Sentiment per Product
        review_tfidf_features = self.tfidf_vectorizer.transform(self.review_data["Processed_Review_Text"])
        review_sentiments = self.sentiment_model.predict_proba(review_tfidf_features)[:, 1] # Probability of positive sentiment
        self.review_data["Sentiment_Score"] = review_sentiments

        # Group sentiments by product and calculate mean
        aggregated_sentiment = self.review_data.groupby("ProductID")["Sentiment_Score"].mean().reset_index()
        # Merge back to product_data, fill NaN for products with no reviews
        product_sentiment_df = self.product_data[["ProductID"]].merge(aggregated_sentiment, on="ProductID", how="left")
        product_sentiment_df["Sentiment_Score"] = product_sentiment_df["Sentiment_Score"].fillna(0.5) # Neutral if no reviews

        # Combine all features
        # Convert product_num_cat_features to DataFrame for easier merging
        product_num_cat_df = pd.DataFrame(product_num_cat_features, index=self.product_data.index)
        
        # Align ProductID for embeddings and sentiment before concatenation
        product_embeddings_df = pd.DataFrame(product_description_embeddings, index=self.product_data.index)
        
        # Need to ensure the order of products in all feature sets is the same as self.product_data
        # The preprocessor and sentence_transformer maintain original order for dataframes passed directly
        # For sentiment, ensure product_sentiment_df is aligned by ProductID
        final_product_features_list = []
        for i, product_id in enumerate(self.product_data["ProductID"]):
            num_cat_feat = product_num_cat_features[i]
            desc_emb = product_description_embeddings[i]
            sentiment_score = product_sentiment_df[product_sentiment_df["ProductID"] == product_id]["Sentiment_Score"].iloc[0]
            final_product_features_list.append(np.concatenate([num_cat_feat, desc_emb, [sentiment_score]]))

        self.product_features_matrix = np.array(final_product_features_list)


    def get_sentiment(self, review_text):
        if self.sentiment_model is None or self.tfidf_vectorizer is None:
            raise ValueError("Sentiment model not trained or vectorizer not initialized.")
        processed_text = preprocess_text(review_text)
        if not processed_text:
            return "Neutral"
        text_tfidf = self.tfidf_vectorizer.transform([processed_text])
        sentiment_proba = self.sentiment_model.predict_proba(text_tfidf)[0]
        if sentiment_proba[1] > 0.6: # Threshold for positive
            return "Positive"
        elif sentiment_proba[0] > 0.6: # Threshold for negative
            return "Negative"
        else:
            return "Neutral"

    def recommend_products(self, product_id, top_n=3):
        if self.product_features_matrix is None or self.product_ids is None:
            raise ValueError("Product features not generated. Call generate_product_feature_vectors first.")

        try:
            product_idx = self.product_ids.index(product_id)
        except ValueError:
            return f"Product ID {product_id} not found."

        target_product_features = self.product_features_matrix[product_idx].reshape(1, -1)
        similarities = cosine_similarity(target_product_features, self.product_features_matrix).flatten()

        # Exclude the product itself and sort by similarity
        similar_indices = similarities.argsort()[-top_n-1:][::-1]
        similar_indices = [idx for idx in similar_indices if self.product_ids[idx] != product_id]

        recommended_products = []
        for idx in similar_indices[:top_n]:
            recommended_products.append(self.product_data.loc[self.product_data["ProductID"] == self.product_ids[idx]].to_dict(orient="records")[0])

        return recommended_products


if __name__ == "__main__":
    system = ECommerceMLSystem()
    print("Loading data...")
    system.load_data()
    print("Training preprocessing pipelines...")
    system.train_preprocessing_pipelines()
    print("Training sentiment model...")
    system.train_sentiment_model()
    print("Generating product feature vectors for recommendation...")
    system.generate_product_feature_vectors()

    print("\n--- Sentiment Analysis Examples ---")
    test_reviews = [
        "This product is fantastic, absolutely love it!",
        "Terrible quality, broke after one week.",
        "It's okay, nothing special.",
        "Very good value for money."
    ]
    for review in test_reviews:
        sentiment = system.get_sentiment(review)
        print(f"Review: '{review}' -> Sentiment: {sentiment}")

    print("\n--- Product Recommendation Examples ---")
    product_to_recommend_for = "P1" # Laptop X
    print(f"\nRecommendations for Product {product_to_recommend_for} ({system.product_data[system.product_data['ProductID']==product_to_recommend_for]['Name'].iloc[0]}):")
    recommendations = system.recommend_products(product_to_recommend_for)
    if isinstance(recommendations, str):
        print(recommendations)
    else:
        for rec in recommendations:
            print(f"  - {rec['Name']} ({rec['Category']}) - Price: ${rec['Price']:.2f}, Rating: {rec['Rating']}")

    product_to_recommend_for = "P3" # Headphones Z
    print(f"\nRecommendations for Product {product_to_recommend_for} ({system.product_data[system.product_data['ProductID']==product_to_recommend_for]['Name'].iloc[0]}):")
    recommendations = system.recommend_products(product_to_recommend_for)
    if isinstance(recommendations, str):
        print(recommendations)
    else:
        for rec in recommendations:
            print(f"  - {rec['Name']} ({rec['Category']}) - Price: ${rec['Price']:.2f}, Rating: {rec['Rating']}")

    product_to_recommend_for = "P5" # Tablet B
    print(f"\nRecommendations for Product {product_to_recommend_for} ({system.product_data[system.product_data['ProductID']==product_to_recommend_for]['Name'].iloc[0]}):")
    recommendations = system.recommend_products(product_to_recommend_for)
    if isinstance(recommendations, str):
        print(recommendations)
    else:
        for rec in recommendations:
            print(f"  - {rec['Name']} ({rec['Category']}) - Price: ${rec['Price']:.2f}, Rating: {rec['Rating']}")
