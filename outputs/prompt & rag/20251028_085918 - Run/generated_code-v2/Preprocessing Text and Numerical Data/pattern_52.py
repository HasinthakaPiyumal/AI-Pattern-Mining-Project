import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize


class TextPreprocessor:
    def __init__(self, use_stemming=True):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer() if use_stemming else None
        self.lemmatizer = WordNetLemmatizer() if not use_stemming else None
        self.vectorizer = TfidfVectorizer(max_features=5000)

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word not in self.stop_words]
        if self.stemmer:
            processed_tokens = [self.stemmer.stem(word) for word in filtered_tokens]
        elif self.lemmatizer:
            processed_tokens = [self.lemmatizer.lemmatize(word) for word in filtered_tokens]
        else:
            processed_tokens = filtered_tokens
        return " ".join(processed_tokens)

    def fit_vectorizer(self, texts):
        processed_texts = [self.preprocess_text(text) for text in texts]
        self.vectorizer.fit(processed_texts)

    def transform_texts(self, texts):
        processed_texts = [self.preprocess_text(text) for text in texts]
        return self.vectorizer.transform(processed_texts)


class NumericalPreprocessor:
    def __init__(self, numerical_features, categorical_features):
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        self.categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        self.preprocessor = None

    def fit(self, df):
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', self.numeric_transformer, self.numerical_features),
                ('cat', self.categorical_transformer, self.categorical_features)
            ])
        self.preprocessor.fit(df)

    def transform(self, df):
        if self.preprocessor is None:
            raise RuntimeError("NumericalPreprocessor has not been fitted yet.")
        return self.preprocessor.transform(df)


class SentimentModel:
    def __init__(self, model=LogisticRegression(max_iter=1000, solver='liblinear')):
        self.model = model

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)


class RecommendationEngine:
    def __init__(self, products_df):
        self.products_df = products_df.copy()
        self.similarity_matrix = None

    def build_similarity_matrix(self, preprocessed_features):
        self.similarity_matrix = cosine_similarity(preprocessed_features)

    def recommend_products(self, product_id, n_recommendations=5):
        if self.similarity_matrix is None:
            raise RuntimeError("Similarity matrix has not been built yet.")

        product_idx = self.products_df[self.products_df['product_id'] == product_id].index[0]
        similar_products_indices = self.similarity_matrix[product_idx].argsort()[-n_recommendations-1:-1][::-1]
        
        recommended_product_ids = self.products_df.iloc[similar_products_indices]['product_id'].tolist()
        return recommended_product_ids


if __name__ == "__main__":
    
    try:
        stopwords.words('english')
    except LookupError:
        import nltk
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')

    
    raw_reviews_data = {
        'review_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'product_id': [101, 102, 101, 103, 102, 104, 101, 103, 105, 104],
        'review_text': [
            "This product is amazing! I love it.",
            "Disappointed with the quality, broke quickly.",
            "Good value for money, highly recommend.",
            "It's okay, nothing special.",
            "Fantastic purchase, very happy!",
            "Terrible experience, complete waste of money.",
            "Solid product, performs as expected.",
            "Could be better, a bit flimsy.",
            "Excellent! Will buy again.",
            "Not worth the price, look elsewhere."
        ],
        'sentiment_label': ['positive', 'negative', 'positive', 'neutral', 'positive', 'negative', 'positive', 'negative', 'positive', 'negative']
    }
    raw_products_data = {
        'product_id': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        'product_name': ['Laptop Pro', 'Mouse Ergonomic', 'Keyboard RGB', 'Monitor 4K', 'Webcam HD', 'Headphones ANC', 'Microphone USB', 'Speaker Portable', 'Router WiFi', 'Smartwatch X'],
        'price': [1200, 35, 80, 450, 60, 150, 75, 90, 110, 200],
        'category': ['Electronics', 'Electronics', 'Electronics', 'Electronics', 'Electronics', 'Electronics', 'Electronics', 'Electronics', 'Electronics', 'Wearable'],
        'avg_rating': [4.5, 2.8, 4.2, 3.9, 4.7, 3.5, 4.1, 3.0, 4.8, 3.2],
        'stock_available': [True, True, False, True, True, False, True, True, True, False]
    }

    reviews_df = pd.DataFrame(raw_reviews_data)
    products_df = pd.DataFrame(raw_products_data)

    print("--- Raw Data ---")
    print("Reviews Sample:\n", reviews_df.head())
    print("Products Sample:\n", products_df.head())
    print("\n")

    
    text_preprocessor = TextPreprocessor(use_stemming=True)
    text_preprocessor.fit_vectorizer(reviews_df['review_text'])
    X_text_features = text_preprocessor.transform_texts(reviews_df['review_text'])

    
    numerical_features_prod = ['price', 'avg_rating']
    categorical_features_prod = ['category', 'stock_available']
    numerical_preprocessor = NumericalPreprocessor(numerical_features_prod, categorical_features_prod)
    numerical_preprocessor.fit(products_df)
    X_numerical_features_prod = numerical_preprocessor.transform(products_df)

    print("--- Preprocessed Data ---")
    print("Text Features (TF-IDF shape):", X_text_features.shape)
    print("Numerical Product Features (shape):", X_numerical_features_prod.shape)
    print("\n")

    
    X_sentiment = X_text_features 
    y_sentiment = reviews_df['sentiment_label'].map({'positive': 1, 'negative': 0, 'neutral': 0.5}).values 

    X_train, X_test, y_train, y_test = train_test_split(X_sentiment, y_sentiment, test_size=0.2, random_state=42)

    sentiment_model = SentimentModel()
    sentiment_model.train(X_train, y_train)
    sentiment_predictions = sentiment_model.predict(X_test)
    sentiment_proba = sentiment_model.predict_proba(X_test)

    print("--- Sentiment Analysis Results ---")
    print("Sample Test Reviews:")
    for i, review_idx in enumerate(X_test.nonzero()[0]): 
        if i >= 3: break
        print(f"  Review: {reviews_df.loc[X_test[review_idx].nonzero()[1][0] // text_preprocessor.vectorizer.max_features + reviews_df.index[0], 'review_text']}")
        print(f"  Predicted Sentiment: {'Positive' if sentiment_predictions[i] == 1 else 'Negative' if sentiment_predictions[i] == 0 else 'Neutral'} (Proba: {sentiment_proba[i][1]:.2f})")
    
    review_to_predict = "This is a decent product but I expected more. Not bad, not great."
    processed_review = text_preprocessor.transform_texts([review_to_predict])
    predicted_sentiment = sentiment_model.predict(processed_review)[0]
    predicted_sentiment_proba = sentiment_model.predict_proba(processed_review)[0]
    sentiment_label = 'Positive' if predicted_sentiment == 1 else 'Negative' if predicted_sentiment == 0 else 'Neutral'
    print(f"\nPrediction for new review '{review_to_predict}': {sentiment_label} (Positive Proba: {predicted_sentiment_proba[1]:.2f})")
    print("\n")

    
    recommendation_engine = RecommendationEngine(products_df)
    recommendation_engine.build_similarity_matrix(X_numerical_features_prod)

    sample_product_id = 101 
    recommended_products = recommendation_engine.recommend_products(sample_product_id)

    print("--- Product Recommendation Results ---")
    print(f"If you liked Product ID {sample_product_id} ({products_df[products_df['product_id'] == sample_product_id]['product_name'].iloc[0]}), you might also like:")
    for rec_id in recommended_products:
        prod_name = products_df[products_df['product_id'] == rec_id]['product_name'].iloc[0]
        print(f"  - Product ID {rec_id}: {prod_name}")