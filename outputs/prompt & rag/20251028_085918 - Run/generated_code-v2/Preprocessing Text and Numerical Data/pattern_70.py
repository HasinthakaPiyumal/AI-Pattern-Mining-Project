import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Ensure NLTK data is downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')


class DataIngestion:
    def ingest_data(self, file_path):
        try:
            df = pd.read_csv(file_path)
            return df
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            return pd.DataFrame()

class TextPreprocessor:
    def __init__(self, stem_or_lem="stem"):
        self.stop_words = set(stopwords.words('english'))
        if stem_or_lem == "stem":
            self.tokenizer = PorterStemmer()
            self.process_word = self.tokenizer.stem
        else:
            self.tokenizer = WordNetLemmatizer()
            self.process_word = self.tokenizer.lemmatize
        self.vectorizer = TfidfVectorizer(max_features=5000)

    def preprocess_text(self, text):
        tokens = nltk.word_tokenize(text.lower())
        filtered_tokens = [self.process_word(word) for word in tokens if word.isalpha() and word not in self.stop_words]
        return " ".join(filtered_tokens)

    def fit_transform(self, series):
        processed_series = series.apply(self.preprocess_text)
        return self.vectorizer.fit_transform(processed_series)
    
    def transform(self, series):
        processed_series = series.apply(self.preprocess_text)
        return self.vectorizer.transform(processed_series)


class NumericalPreprocessor:
    def __init__(self, numerical_cols, categorical_cols):
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        
        self.numerical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        
        self.categorical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', self.numerical_pipeline, self.numerical_cols),
                ('cat', self.categorical_pipeline, self.categorical_cols)
            ], 
            remainder='passthrough'
        )

    def fit_transform(self, df):
        return self.preprocessor.fit_transform(df)

    def transform(self, df):
        return self.preprocessor.transform(df)


class SentimentAnalyzer:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)

    def train_model(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        return self.model, accuracy, report

    def predict(self, X):
        return self.model.predict(X)


class PricePredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def train_model(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return self.model, mse, r2

    def predict(self, X):
        return self.model.predict(X)


def main():
    # --- 1. Data Ingestion ---
    data_ingestor = DataIngestion()
    
    # Create a dummy dataset for demonstration
    data = {
        'product_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'product_name': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Webcam', 'Headphones', 'Speaker', 'Microphone', 'Router', 'Printer'],
        'review': [
            'This laptop is excellent, fast and powerful!',
            'Mouse is okay, but sometimes lags.',
            'Great keyboard for gaming, very responsive.',
            'Monitor has great display quality, a bit expensive.',
            'Webcam is clear but low light performance is bad.',
            'Headphones have amazing sound and comfort.',
            'Speaker quality is decent for its price.',
            'Microphone picks up too much background noise.',
            'Router setup was easy, good signal strength.',
            'Printer is slow but reliable.'
        ],
        'price': [1200.00, 25.50, 75.00, 300.00, 45.00, 150.00, 80.00, 60.00, 90.00, 180.00],
        'rating': [5, 3, 4, 4, 2, 5, 3, 2, 4, 3],
        'sales_volume': [1500, 5000, 2000, 800, 3000, 1000, 2500, 1200, 1800, 700],
        'category': ['Electronics', 'Electronics', 'Electronics', 'Electronics', 'Electronics', 'Electronics', 'Electronics', 'Electronics', 'Electronics', 'Electronics'],
        'sentiment_label': [1, 0, 1, 1, 0, 1, 0, 0, 1, 0] # 1 for positive, 0 for negative
    }
    df = pd.DataFrame(data)
    
    # In a real scenario, you'd load from a file:
    # df = data_ingestor.ingest_data("your_product_data.csv")
    if df.empty:
        return

    print("--- Data Ingested ---")
    print(df.head())
    print("\n")

    # --- 2. Text Preprocessing ---
    text_preprocessor = TextPreprocessor(stem_or_lem="lemmatize") # Using lemmatization
    text_features = text_preprocessor.fit_transform(df['review'])
    
    # Convert sparse matrix to DataFrame for concatenation
    text_feature_df = pd.DataFrame(text_features.toarray(), columns=[f"text_feature_{i}" for i in range(text_features.shape[1])])
    text_feature_df.index = df.index # Align indices

    print("--- Text Preprocessing Complete (TF-IDF Vectorized) ---")
    print(text_feature_df.head())
    print("\n")

    # --- 3. Numerical Preprocessing ---
    numerical_cols = ['price', 'rating', 'sales_volume']
    categorical_cols = ['category']
    
    numerical_preprocessor = NumericalPreprocessor(numerical_cols, categorical_cols)
    
    # Create a temporary DataFrame for numerical/categorical processing
    # to avoid issues with sparse text features during ColumnTransformer fit
    temp_df_numerical = df[numerical_cols + categorical_cols].copy()
    processed_numerical_features = numerical_preprocessor.fit_transform(temp_df_numerical)
    
    # Get feature names after one-hot encoding for categorical columns
    onehot_feature_names = numerical_preprocessor.preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_cols)
    all_numerical_feature_names = numerical_cols + list(onehot_feature_names)
    
    processed_numerical_df = pd.DataFrame(processed_numerical_features, columns=all_numerical_feature_names, index=df.index)

    print("--- Numerical Preprocessing Complete ---")
    print(processed_numerical_df.head())
    print("\n")

    # Combine all features
    combined_features_df = pd.concat([processed_numerical_df, text_feature_df], axis=1)
    print("--- Combined Features ---")
    print(combined_features_df.head())
    print("\n")

    # --- 4. Sentiment Analysis Model Training ---
    print("--- Training Sentiment Analysis Model ---")
    sentiment_analyzer = SentimentAnalyzer()
    sentiment_model, sentiment_accuracy, sentiment_report = sentiment_analyzer.train_model(combined_features_df, df['sentiment_label'])
    
    print(f"Sentiment Model Accuracy: {sentiment_accuracy:.4f}")
    print("Sentiment Classification Report:\n", sentiment_report)
    print("\n")

    # Predict sentiment for the entire dataset to use as a feature for price prediction
    df['predicted_sentiment'] = sentiment_analyzer.predict(combined_features_df)
    
    print("--- Predicted Sentiments Added to DataFrame ---")
    print(df[['review', 'sentiment_label', 'predicted_sentiment']].head())
    print("\n")

    # --- 5. Price Prediction Model Training ---
    print("--- Training Price Prediction Model ---")
    
    # Features for price prediction now include original numerical features, encoded categories, TF-IDF, and predicted sentiment
    price_prediction_features_df = pd.concat([processed_numerical_df, text_feature_df, df[['predicted_sentiment']]], axis=1)
    
    price_predictor = PricePredictor()
    price_model, price_mse, price_r2 = price_predictor.train_model(price_prediction_features_df, df['price'])

    print(f"Price Prediction Model MSE: {price_mse:.4f}")
    print(f"Price Prediction Model R2 Score: {price_r2:.4f}")
    print("\n")

    # --- Demonstrate Predictions ---
    print("--- Demonstrating Predictions ---")
    sample_data_for_prediction = df.sample(n=3, random_state=1)
    print("Original Sample Data:\n", sample_data_for_prediction[['review', 'price', 'rating', 'sentiment_label']])
    
    # Preprocess sample data for prediction
    sample_text_features = text_preprocessor.transform(sample_data_for_prediction['review'])
    sample_text_feature_df = pd.DataFrame(sample_text_features.toarray(), columns=[f"text_feature_{i}" for i in range(sample_text_features.shape[1])], index=sample_data_for_prediction.index)
    
    sample_numerical_df = sample_data_for_prediction[numerical_cols + categorical_cols].copy()
    sample_processed_numerical_features = numerical_preprocessor.transform(sample_numerical_df)
    sample_processed_numerical_df = pd.DataFrame(sample_processed_numerical_features, columns=all_numerical_feature_names, index=sample_data_for_prediction.index)
    
    sample_combined_features_df = pd.concat([sample_processed_numerical_df, sample_text_feature_df], axis=1)
    
    sample_predicted_sentiment = sentiment_analyzer.predict(sample_combined_features_df)
    print("\nSample Predicted Sentiments:", sample_predicted_sentiment)

    sample_price_prediction_features_df = pd.concat([sample_processed_numerical_df, sample_text_feature_df, pd.DataFrame(sample_predicted_sentiment, columns=['predicted_sentiment'], index=sample_data_for_prediction.index)], axis=1)
    
    sample_predicted_prices = price_predictor.predict(sample_price_prediction_features_df)
    
    sample_results = sample_data_for_prediction[['price', 'review', 'sentiment_label']].copy()
    sample_results['predicted_sentiment'] = sample_predicted_sentiment
    sample_results['predicted_price'] = sample_predicted_prices
    
    print("\nSample Prediction Results:")
    print(sample_results)

if __name__ == "__main__":
    main()