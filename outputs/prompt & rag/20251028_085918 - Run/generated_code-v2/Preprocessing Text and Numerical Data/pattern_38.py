import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import joblib


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

class TicketProcessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = set(stopwords.words('english'))
        self.category_model = None
        self.sentiment_model = None
        self.preprocessor = None

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = nltk.word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stopwords]
        return " ".join(tokens)

    def train(self, df, text_col, numerical_cols, category_col, sentiment_col):
        df['cleaned_text'] = df[text_col].apply(self.clean_text)

        text_features = 'cleaned_text'

        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('text', TfidfVectorizer(), text_features)
            ])

        X = df[[text_features] + numerical_cols]
        y_category = df[category_col]
        y_sentiment = df[sentiment_col]

        X_processed = self.preprocessor.fit_transform(X)

        self.category_model = LogisticRegression(random_state=42, max_iter=1000)
        self.category_model.fit(X_processed, y_category)

        self.sentiment_model = LogisticRegression(random_state=42, max_iter=1000)
        self.sentiment_model.fit(X_processed, y_sentiment)

    def predict(self, df, text_col, numerical_cols):
        df['cleaned_text'] = df[text_col].apply(self.clean_text)
        X = df[['cleaned_text'] + numerical_cols]
        X_processed = self.preprocessor.transform(X)

        category_predictions = self.category_model.predict(X_processed)
        sentiment_predictions = self.sentiment_model.predict(X_processed)

        return category_predictions, sentiment_predictions


if __name__ == "__main__":
    data = {
        'ticket_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'ticket_text': [
            "My order #12345 hasn't arrived. Where is it?",
            "The product I received is damaged. I need a refund.",
            "How do I return an item? The website isn't clear.",
            "Great service! My issue was resolved quickly.",
            "I want to change my shipping address for order #67890.",
            "The app keeps crashing. Very frustrating experience.",
            "Payment failed for my recent purchase.",
            "Can I track my order #54321 online?",
            "This is the worst product ever! Completely broken.",
            "Fantastic support from your team regarding my query."
        ],
        'priority_score': [8, 9, 6, 4, 7, 9, 8, 5, 10, 3],
        'category': [
            'Shipping', 'Product Issue', 'Returns', 'Service Feedback', 'Shipping',
            'Technical Issue', 'Billing', 'Shipping', 'Product Issue', 'Service Feedback'
        ],
        'sentiment': [
            'Negative', 'Negative', 'Neutral', 'Positive', 'Neutral',
            'Negative', 'Negative', 'Neutral', 'Negative', 'Positive'
        ]
    }
    df = pd.DataFrame(data)

    processor = TicketProcessor()
    
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

    text_col = 'ticket_text'
    numerical_cols = ['priority_score']
    category_col = 'category'
    sentiment_col = 'sentiment'

    processor.train(train_df, text_col, numerical_cols, category_col, sentiment_col)

    joblib.dump(processor, 'ticket_processor_pipeline.pkl')

    loaded_processor = joblib.load('ticket_processor_pipeline.pkl')

    new_tickets_data = {
        'ticket_id': [11, 12, 13],
        'ticket_text': [
            "My recent purchase arrived broken, this is unacceptable!",
            "I need assistance with my account login. Forgotten password.",
            "Thank you for the quick delivery, very happy with my order!"
        ],
        'priority_score': [9, 7, 4]
    }
    new_tickets_df = pd.DataFrame(new_tickets_data)

    predicted_categories, predicted_sentiments = loaded_processor.predict(new_tickets_df, text_col, numerical_cols)

    new_tickets_df['predicted_category'] = predicted_categories
    new_tickets_df['predicted_sentiment'] = predicted_sentiments

    print(new_tickets_df[['ticket_text', 'priority_score', 'predicted_category', 'predicted_sentiment']])