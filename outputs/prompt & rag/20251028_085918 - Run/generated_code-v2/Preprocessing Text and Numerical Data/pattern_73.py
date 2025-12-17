import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_absolute_error

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class TicketPreprocessingPipeline:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = set(stopwords.words('english'))

        self.numerical_features = ['customer_id', 'previous_interactions']
        self.categorical_features = ['product_id']
        self.text_features = 'ticket_description'

        self.numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        self.categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', self.numeric_transformer, self.numerical_features),
                ('cat', self.categorical_transformer, self.categorical_features),
                ('text', TfidfVectorizer(preprocessor=self._preprocess_text, max_features=5000), self.text_features)
            ],
            remainder='passthrough'
        )

    def _preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = nltk.word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stopwords]
        return ' '.join(tokens)

    def fit(self, X, y=None):
        self.preprocessor.fit(X)
        return self

    def transform(self, X):
        return self.preprocessor.transform(X)

class TicketCategorizationPrioritizationModel:
    def __init__(self, random_state=42):
        self.category_model = RandomForestClassifier(random_state=random_state)
        self.priority_model = RandomForestRegressor(random_state=random_state)

    def fit(self, X_transformed, y_category, y_priority):
        self.category_model.fit(X_transformed, y_category)
        self.priority_model.fit(X_transformed, y_priority)
        return self

    def predict(self, X_transformed):
        category_predictions = self.category_model.predict(X_transformed)
        priority_predictions = self.priority_model.predict(X_transformed)
        return category_predictions, priority_predictions

if __name__ == "__main__":
    data = {
        'ticket_description': [
            'My internet is not working at all since yesterday. Please help!',
            'I have a billing question regarding my last invoice.',
            'Feature request: can you add a dark mode option to the app?',
            'Account locked out, cannot log in. Urgent assistance needed.',
            'Problem with product X, it keeps crashing. Need support.',
            'General inquiry about service plans.',
            'My data plan seems incorrect. Check usage.',
            'Improve UI for settings page.',
            'Printer not connecting to network. Troubleshooting help.',
            'Want to upgrade my subscription.',
            np.nan, # Missing text
            'Another internet issue, intermittent connection.'
        ],
        'customer_id': [
            101, 102, 103, 104, 105, 101, 106, 107, 108, 109, 110, 111
        ],
        'product_id': [
            'P1', 'P2', 'P3', 'P1', 'P2', 'P3', 'P1', 'P2', 'P3', 'P1', 'P2', 'P3'
        ],
        'previous_interactions': [
            2, 0, 1, 5, 3, np.nan, 1, 0, 2, 0, 4, 1
        ],
        'category': [
            'Technical Issue', 'Billing Query', 'Feature Request', 'Technical Issue', 
            'Technical Issue', 'General Inquiry', 'Billing Query', 'Feature Request', 
            'Technical Issue', 'General Inquiry', 'Technical Issue', 'Technical Issue'
        ],
        'priority_score': [
            0.9, 0.4, 0.2, 0.95, 0.8, 0.3, 0.6, 0.1, 0.7, 0.25, 0.85, 0.75
        ]
    }
    df = pd.DataFrame(data)

    # Handle NaN in text by filling with an empty string before splitting for consistency
    df['ticket_description'] = df['ticket_description'].fillna('')

    X = df[['ticket_description', 'customer_id', 'product_id', 'previous_interactions']]
    y_category = df['category']
    y_priority = df['priority_score']

    X_train, X_test, y_category_train, y_category_test, y_priority_train, y_priority_test = train_test_split(
        X, y_category, y_priority, test_size=0.2, random_state=42, stratify=y_category
    )

    # Preprocessing
    preprocessor = TicketPreprocessingPipeline()
    X_train_transformed = preprocessor.fit(X_train).transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Model Training and Prediction
    model_system = TicketCategorizationPrioritizationModel()
    model_system.fit(X_train_transformed, y_category_train, y_priority_train)
    category_preds, priority_preds = model_system.predict(X_test_transformed)

    # Evaluation
    print("\n--- Categorization Model Evaluation ---")
    print(classification_report(y_category_test, category_preds))

    print("\n--- Prioritization Model Evaluation ---")
    print(f"Mean Absolute Error: {mean_absolute_error(y_priority_test, priority_preds):.4f}")

    print("\n--- Example Predictions on Test Set ---")
    for i in range(len(X_test)):
        print(f"Ticket: {X_test.iloc[i]['ticket_description'][:50]}...")
        print(f"Actual Category: {y_category_test.iloc[i]}, Predicted: {category_preds[i]}")
        print(f"Actual Priority: {y_priority_test.iloc[i]:.2f}, Predicted: {priority_preds[i]:.2f}")
        print("-" * 20)
