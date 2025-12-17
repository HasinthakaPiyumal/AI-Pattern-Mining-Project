import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

try:
    nltk.data.find("corpora/stopwords")
except nltk.downloader.DownloadError:
    nltk.download("stopwords")
try:
    nltk.data.find("corpora/wordnet")
except nltk.downloader.DownloadError:
    nltk.download("wordnet")
try:
    nltk.data.find("tokenizers/punkt")
except nltk.downloader.DownloadError:
    nltk.download("punkt")

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(max_features=1000)

    def fit(self, X, y=None):
        processed_text = X.apply(self._preprocess_text)
        self.vectorizer.fit(processed_text)
        return self

    def transform(self, X, y=None):
        processed_text = X.apply(self._preprocess_text)
        return self.vectorizer.transform(processed_text)

    def _preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text) # Remove punctuation and numbers
        tokens = nltk.word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        return " ".join(tokens)

class NumericalPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.imputer = SimpleImputer(strategy='mean')
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.imputer.fit(X[self.columns])
        self.scaler.fit(self.imputer.transform(X[self.columns]))
        return self

    def transform(self, X, y=None):
        imputed_data = self.imputer.transform(X[self.columns])
        scaled_data = self.scaler.transform(imputed_data)
        return scaled_data

class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_frame):
        return data_frame[self.key]

class SupportTicketSystem:
    def __init__(self):
        self.text_features = 'description'
        self.numerical_features = ['customer_sentiment_score', 'ticket_age_hours']
        self.categorization_model = RandomForestClassifier(random_state=42)
        self.prioritization_model = RandomForestRegressor(random_state=42)
        self.preprocessing_pipeline = self._build_preprocessing_pipeline()

    def _build_preprocessing_pipeline(self):
        return Pipeline([
            ('features', FeatureUnion([
                ('text_features', Pipeline([
                    ('selector', ItemSelector(key=self.text_features)),
                    ('preprocessor', TextPreprocessor())
                ])),
                ('numerical_features', Pipeline([
                    ('selector', ItemSelector(key=self.numerical_features)),
                    ('preprocessor', NumericalPreprocessor(columns=self.numerical_features))
                ]))
            ]))
        ])

    def train(self, X_train, y_train_category, y_train_priority):
        print("Fitting preprocessing pipeline...")
        X_train_processed = self.preprocessing_pipeline.fit_transform(X_train)

        print("Training categorization model...")
        self.categorization_model.fit(X_train_processed, y_train_category)

        print("Training prioritization model...")
        self.prioritization_model.fit(X_train_processed, y_train_priority)
        print("Training complete.")

    def predict(self, X_new):
        print("Preprocessing new data...")
        X_new_processed = self.preprocessing_pipeline.transform(X_new)

        print("Predicting category...")
        category_predictions = self.categorization_model.predict(X_new_processed)

        print("Predicting priority...")
        priority_predictions = self.prioritization_model.predict(X_new_processed)

        return category_predictions, priority_predictions

if __name__ == "__main__":
    # Simulate Data Ingestion
    data = {
        'description': [
            'Printer not working, urgent issue!',
            'Cannot log in to my account, password reset needed.',
            'Question about my latest bill, it seems incorrect.',
            'Slow internet connection, browsing is difficult.',
            'Feature request: add dark mode to the application.',
            'My computer crashed, blue screen of death.',
            'I forgot my username, please help.',
            'Billing inquiry for subscription renewal.',
            'Network down, cannot access internal tools.',
            'General feedback about user interface improvements.'
        ],
        'customer_sentiment_score': [0.8, 0.2, 0.5, 0.1, 0.9, 0.0, 0.3, 0.6, 0.0, 0.7],
        'ticket_age_hours': [2, 1, 5, 3, 10, 0.5, 4, 7, 1.5, 8],
        'category': ['Technical', 'Account', 'Billing', 'Technical', 'Feature Request', 'Technical', 'Account', 'Billing', 'Technical', 'General Inquiry'],
        'priority': [4, 3, 2, 4, 1, 5, 2, 3, 5, 1]
    }
    df = pd.DataFrame(data)

    # Split data for training and testing
    X = df[['description', 'customer_sentiment_score', 'ticket_age_hours']]
    y_category = df['category']
    y_priority = df['priority']

    X_train, X_test, y_train_category, y_test_category, y_train_priority, y_test_priority = train_test_split(
        X, y_category, y_priority, test_size=0.2, random_state=42
    )

    # Initialize and train the system
    system = SupportTicketSystem()
    system.train(X_train, y_train_category, y_train_priority)

    # Simulate new incoming tickets for prediction
    new_tickets_data = {
        'description': [
            'My internet is completely down, I cannot work!',
            'I need to update my payment method.',
            'The app is crashing every time I open it.',
            'Can I get a refund for my last purchase?'
        ],
        'customer_sentiment_score': [0.0, 0.6, 0.1, 0.4],
        'ticket_age_hours': [0.1, 0.5, 0.2, 1.0]
    }
    new_tickets_df = pd.DataFrame(new_tickets_data)

    # Make predictions on new tickets
    predicted_categories, predicted_priorities = system.predict(new_tickets_df)

    print("\n--- New Ticket Predictions ---")
    for i in range(len(new_tickets_df)):
        print(f"Ticket: '{new_tickets_df.iloc[i]['description']}'")
        print(f"  Predicted Category: {predicted_categories[i]}")
        print(f"  Predicted Priority: {int(round(predicted_priorities[i]))}") # Round priority to nearest integer
        print("------------------------------")