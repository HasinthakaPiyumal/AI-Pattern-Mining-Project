import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from scipy.sparse import hstack

nltk.download('stopwords')
nltk.download('wordnet')

class SupportTicketSystem:
    def __init__(self):
        self.text_pipeline = None
        self.numerical_pipeline = None
        self.category_model = None
        self.priority_model = None
        self.tfidf_vectorizer = TfidfVectorizer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def _preprocess_text(self, text):
        tokens = nltk.word_tokenize(text.lower())
        filtered_tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in self.stop_words]
        return " ".join(filtered_tokens)

    def train(self, tickets_df):
        # 1. Data Ingestion (simulated with input DataFrame)
        X_text = tickets_df['ticket_description']
        X_numerical = tickets_df[['customer_tenure', 'previous_interactions']]
        y_category = tickets_df['category']
        y_priority = tickets_df['priority']

        # Split data for training (demonstration purposes)
        X_train_text, X_test_text, X_train_num, X_test_num, y_cat_train, y_cat_test, y_pri_train, y_pri_test = \
            train_test_split(X_text, X_numerical, y_category, y_priority, test_size=0.2, random_state=42)

        # 2. Preprocessing Module
        # Text Preprocessor (fit and transform on training data)
        processed_train_text = X_train_text.apply(self._preprocess_text)
        self.tfidf_vectorizer.fit(processed_train_text)
        X_train_text_vec = self.tfidf_vectorizer.transform(processed_train_text)

        # Numerical Preprocessor (fit on training data)
        self.numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        X_train_num_processed = self.numerical_pipeline.fit_transform(X_train_num)

        # 3. Feature Combination Module
        X_train_combined = hstack([X_train_text_vec, X_train_num_processed])

        # 4. Machine Learning Models
        # Ticket Categorization Model
        self.category_model = RandomForestClassifier(random_state=42)
        self.category_model.fit(X_train_combined, y_cat_train)

        # Priority Assignment Model
        self.priority_model = RandomForestClassifier(random_state=42)
        self.priority_model.fit(X_train_combined, y_pri_train)

        print("Models trained successfully!")

    def predict(self, raw_ticket_description, raw_numerical_metadata):
        # Preprocess new text data
        processed_text = self._preprocess_text(raw_ticket_description)
        text_vec = self.tfidf_vectorizer.transform([processed_text])

        # Preprocess new numerical data
        numerical_df = pd.DataFrame([raw_numerical_metadata], columns=['customer_tenure', 'previous_interactions'])
        numerical_processed = self.numerical_pipeline.transform(numerical_df)

        # Combine features
        combined_features = hstack([text_vec, numerical_processed])

        # Predict category and priority
        predicted_category = self.category_model.predict(combined_features)[0]
        predicted_priority = self.priority_model.predict(combined_features)[0]

        return {"category": predicted_category, "priority": predicted_priority}

# --- Example Usage ---
if __name__ == "__main__":
    # Generate dummy data
    data = {
        'ticket_description': [
            "My internet is not working at all, can't connect to anything.",
            "I want to change my billing address, please help me with this.",
            "The new software update broke my application, it crashes on startup.",
            "How can I upgrade my subscription plan?",
            "I have a question about my last invoice, it seems incorrect.",
            "My password reset link is not arriving in my email.",
            "Need technical assistance with setting up my new device.",
            "Can you explain the charges on my bill for October?",
            "Website is very slow, taking ages to load pages.",
            "I need to cancel my service, what's the procedure?"
        ],
        'customer_tenure': [12, 3, 24, 6, 18, 9, 30, 5, 15, 7],
        'previous_interactions': [5, 1, 10, 2, 7, 3, 12, 1, 6, 4],
        'category': [
            'Technical', 'Billing', 'Technical', 'Billing', 'Billing',
            'Technical', 'Technical', 'Billing', 'Technical', 'Billing'
        ],
        'priority': [
            'High', 'Low', 'High', 'Medium', 'Medium',
            'Medium', 'High', 'Low', 'High', 'Low'
        ]
    }
    df = pd.DataFrame(data)

    # Simulate some missing numerical values for imputation
    df.loc[2, 'customer_tenure'] = np.nan
    df.loc[7, 'previous_interactions'] = np.nan

    system = SupportTicketSystem()
    system.train(df)

    # Make a prediction for a new ticket
    new_ticket_desc = "My payment failed and I can't access premium features."
    new_numerical_meta = {'customer_tenure': 8, 'previous_interactions': 2}

    prediction = system.predict(new_ticket_desc, new_numerical_meta)
    print(f"\nNew ticket prediction: {prediction}")

    new_ticket_desc_2 = "My internet is constantly disconnecting, very frustrating."
    new_numerical_meta_2 = {'customer_tenure': 20, 'previous_interactions': 15}

    prediction_2 = system.predict(new_ticket_desc_2, new_numerical_meta_2)
    print(f"New ticket prediction 2: {prediction_2}")

    new_ticket_desc_3 = "I need to update my credit card information for my subscription."
    new_numerical_meta_3 = {'customer_tenure': 1, 'previous_interactions': 0}

    prediction_3 = system.predict(new_ticket_desc_3, new_numerical_meta_3)
    print(f"New ticket prediction 3: {prediction_3}")