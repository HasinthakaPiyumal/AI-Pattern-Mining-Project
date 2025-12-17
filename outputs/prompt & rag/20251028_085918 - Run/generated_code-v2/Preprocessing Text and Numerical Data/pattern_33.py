import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from scipy.sparse import hstack

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

class TicketProcessor:
    def __init__(self):
        self.numerical_imputer = None
        self.numerical_scaler = None
        self.tfidf_vectorizer = None
        self.priority_model = None
        self.department_model = None
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text):
        text = text.lower()
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in self.stop_words]
        return " ".join(tokens)

    def fit(self, df_tickets, numerical_cols, text_col, priority_col, department_col):
        # Preprocess numerical data
        numerical_data = df_tickets[numerical_cols]
        self.numerical_imputer = SimpleImputer(strategy='mean')
        numerical_data_imputed = self.numerical_imputer.fit_transform(numerical_data)

        self.numerical_scaler = StandardScaler()
        numerical_data_scaled = self.numerical_scaler.fit_transform(numerical_data_imputed)

        # Preprocess text data
        df_tickets['cleaned_description'] = df_tickets[text_col].apply(self.preprocess_text)
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000) 
        text_vectors = self.tfidf_vectorizer.fit_transform(df_tickets['cleaned_description'])

        # Combine features
        combined_features = hstack([numerical_data_scaled, text_vectors])

        # Prepare targets
        y_priority = df_tickets[priority_col]
        y_department = df_tickets[department_col]

        # Split data for training
        X_train, X_test, y_priority_train, y_priority_test, y_department_train, y_department_test = \
            train_test_split(combined_features, y_priority, y_department, test_size=0.2, random_state=42)

        # Train Priority Model
        self.priority_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.priority_model.fit(X_train, y_priority_train)

        # Train Department Model
        self.department_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.department_model.fit(X_train, y_department_train)

        return X_test, y_priority_test, y_department_test

    def predict(self, new_ticket_data, numerical_cols, text_col):
        # Preprocess new numerical data
        new_numerical_data = pd.DataFrame([new_ticket_data[col] for col in numerical_cols]).T
        new_numerical_data.columns = numerical_cols # Ensure columns are correctly set

        new_numerical_data_imputed = self.numerical_imputer.transform(new_numerical_data)
        new_numerical_data_scaled = self.numerical_scaler.transform(new_numerical_data_imputed)

        # Preprocess new text data
        new_cleaned_description = self.preprocess_text(new_ticket_data[text_col])
        new_text_vector = self.tfidf_vectorizer.transform([new_cleaned_description])

        # Combine features
        new_combined_features = hstack([new_numerical_data_scaled, new_text_vector])

        # Predict
        predicted_priority = self.priority_model.predict(new_combined_features)[0]
        predicted_department = self.department_model.predict(new_combined_features)[0]

        return predicted_priority, predicted_department


# --- Main Execution --- #
if __name__ == "__main__":
    # 1. Simulate Data Ingestion (dummy data)
    data = {
        'customer_id': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115],
        'product_id': [1, 2, 1, 3, 2, 1, 3, 2, 1, 3, 2, 1, 3, 2, 1],
        'urgency_score': [5, 3, 4, 5, 2, 4, 5, 3, 4, 5, 2, 4, 5, 3, 4],
        'creation_timestamp': pd.to_datetime(['2023-01-01 10:00', '2023-01-01 11:30', '2023-01-01 12:00',
                                                 '2023-01-02 09:00', '2023-01-02 14:00', '2023-01-03 10:00',
                                                 '2023-01-03 11:00', '2023-01-03 13:00', '2023-01-04 09:30',
                                                 '2023-01-04 15:00', '2023-01-05 10:00', '2023-01-05 11:00',
                                                 '2023-01-05 12:00', '2023-01-06 09:00', '2023-01-06 10:00']),
        'description': [
            "My internet is not working at all. This is urgent!",
            "I have a question about my last bill. Can someone help?",
            "Software bug: application crashes when opening file X. Need immediate fix.",
            "My payment didn't go through, what happened?",
            "Need to reset my password, can't log in.",
            "Feature request: Add dark mode to the mobile app.",
            "Database connection error, all services are down. High priority!",
            "Question about changing my subscription plan.",
            "Printer not connecting to my laptop. Please assist.",
            "I received a wrong item in my order. What should I do?",
            "Website is loading very slowly for me.",
            "Issue with my account settings, can't update profile.",
            "Server is unresponsive, major outage affecting customers. Critical!",
            "Enquiry about new product features.",
            "My refund hasn't arrived yet."
        ],
        'priority': ['High', 'Medium', 'High', 'Medium', 'Low', 'Low', 'High', 'Medium', 'Medium', 'High', 'Low', 'Medium', 'High', 'Low', 'Medium'],
        'department': ['Technical', 'Billing', 'Technical', 'Billing', 'General', 'Product', 'Technical', 'Billing', 'Technical', 'Shipping', 'Technical', 'General', 'Technical', 'Product', 'Billing']
    }
    df_tickets = pd.DataFrame(data)

    # Add some missing values for demonstration
    df_tickets.loc[2, 'urgency_score'] = np.nan
    df_tickets.loc[7, 'product_id'] = np.nan

    # Define columns
    numerical_features = ['customer_id', 'product_id', 'urgency_score']
    text_feature = 'description'
    priority_target = 'priority'
    department_target = 'department'

    # Initialize and fit the processor
    processor = TicketProcessor()
    X_test, y_priority_test, y_department_test = processor.fit(
        df_tickets, numerical_features, text_feature, priority_target, department_target
    )

    # Evaluate the models
    print("\n--- Evaluating Priority Model ---")
    y_priority_pred = processor.priority_model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_priority_test, y_priority_pred):.2f}")
    print(classification_report(y_priority_test, y_priority_pred))

    print("\n--- Evaluating Department Model ---")
    y_department_pred = processor.department_model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_department_test, y_department_pred):.2f}")
    print(classification_report(y_department_test, y_department_pred))

    # Demonstrate prediction on a new ticket
    print("\n--- Predicting for a New Ticket ---")
    new_ticket = {
        'customer_id': 201,
        'product_id': 1,
        'urgency_score': 5,
        'creation_timestamp': pd.to_datetime('2023-01-07 10:00'),
        'description': "My account is locked and I can't access anything. This is critical!"
    }

    predicted_priority, predicted_department = processor.predict(new_ticket, numerical_features, text_feature)
    print(f"New Ticket Priority: {predicted_priority}")
    print(f"New Ticket Department: {predicted_department}")

    new_ticket_2 = {
        'customer_id': 202,
        'product_id': 3,
        'urgency_score': 2,
        'creation_timestamp': pd.to_datetime('2023-01-07 11:00'),
        'description': "I just want to ask about the features of your new product. Not urgent."
    }
    predicted_priority_2, predicted_department_2 = processor.predict(new_ticket_2, numerical_features, text_feature)
    print(f"\nNew Ticket 2 Priority: {predicted_priority_2}")
    print(f"New Ticket 2 Department: {predicted_department_2}")