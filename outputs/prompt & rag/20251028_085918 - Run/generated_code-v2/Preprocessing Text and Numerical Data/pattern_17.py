import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Ensure NLTK resources are downloaded
try:
    stopwords.words('english')
    WordNetLemmatizer()
    word_tokenize("test")
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('omw-1.4') # Open Multilingual Wordnet for WordNetLemmatizer


# 1. Data Ingestion & Simulation
def simulate_customer_tickets(n_samples=1000):
    data = {
        'ticket_id': [f'T{i:04d}' for i in range(n_samples)],
        'description': [
            "My internet is not working. It's been down for 3 days and I'm very frustrated.",
            "I need help setting up my new router. The instructions are unclear.",
            "My billing statement seems incorrect. I was overcharged this month.",
            "The app crashes frequently on my Android device. It's very annoying.",
            "I want to upgrade my subscription to the premium plan. Excellent service!",
            "Lost my password, can't access my account. Please help quickly.",
            "The delivery of my product was delayed. Where is my package?",
            "Having trouble connecting my smart home device. Support docs are confusing.",
            "I love your new features! Keep up the good work!",
            "My laptop repair is taking too long. Status update please.",
            "My internet speed is very slow, tried restarting the modem multiple times. This is unacceptable.",
            "I need to change my payment method. How can I do that?",
            "Your customer service representative was incredibly helpful, thank you!",
            "My account was unexpectedly suspended. Why?",
            "I have a general question about product specifications.",
            "The product arrived damaged. I demand a refund immediately.",
            "How do I cancel my service? I no longer need it.",
            "I am satisfied with the recent update to the software.",
            "My credit card was charged twice for the same purchase.",
            "Can you tell me about your holiday discounts?",
        ] * (n_samples // 20 + 1), # Repeat to fill samples
        'priority': np.random.choice(['low', 'medium', 'high'], n_samples, p=[0.5, 0.3, 0.2]),
        'customer_age': np.random.randint(18, 70, n_samples),
        'product_id': np.random.choice(['P101', 'P102', 'P103', 'P104', 'P105', 'P106', np.nan], n_samples, p=[0.2, 0.2, 0.15, 0.15, 0.1, 0.1, 0.1]),
        'sentiment': np.random.choice(['negative', 'neutral', 'positive'], n_samples, p=[0.4, 0.3, 0.3]),
        'category': np.random.choice(['technical', 'billing', 'account', 'product', 'general'], n_samples, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
    }
    df = pd.DataFrame(data).head(n_samples)

    # Introduce some missing customer_age values
    missing_age_indices = np.random.choice(df.index, int(n_samples * 0.05), replace=False)
    df.loc[missing_age_indices, 'customer_age'] = np.nan

    # Align sentiment and category a bit more with descriptions
    df.loc[df['description'].str.contains('frustrated|down|unacceptable|annoying|crashes|damaged|demand refund|suspended|incorrect|overcharged', case=False, na=False), 'sentiment'] = 'negative'
    df.loc[df['description'].str.contains('excellent service|love your|good work|satisfied|helpful|thank you', case=False, na=False), 'sentiment'] = 'positive'
    df.loc[df['description'].str.contains('internet|router|app crashes|smart home device|laptop repair|speed', case=False, na=False), 'category'] = 'technical'
    df.loc[df['description'].str.contains('billing|charged|refund|payment method|overcharged', case=False, na=False), 'category'] = 'billing'
    df.loc[df['description'].str.contains('password|account|subscription|cancel service', case=False, na=False), 'category'] = 'account'
    df.loc[df['description'].str.contains('product|features|specifications|delivery|package', case=False, na=False), 'category'] = 'product'
    df.loc[df['description'].str.contains('general question|holiday discounts', case=False, na=False), 'category'] = 'general'

    return df


# 2. Text Preprocessing (NLTK for TfidfVectorizer)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def text_preprocessor_for_vectorizer(text):
    text = text.lower() # Lowercasing
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text) # Punctuation removal
    tokens = word_tokenize(text) # Tokenization
    filtered_tokens = [word for word in tokens if word not in stop_words] # Stop-word removal
    lemmas = [lemmatizer.lemmatize(word) for word in filtered_tokens] # Lemmatization
    return " ".join(lemmas) # Return a string for TfidfVectorizer

# 3. Define the ColumnTransformer for preprocessing
numerical_features = ['customer_age']
categorical_features = ['priority', 'product_id']
text_features = 'description' # Single text column

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

text_transformer = TfidfVectorizer(preprocessor=text_preprocessor_for_vectorizer, max_features=5000)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features),
        ('txt', text_transformer, text_features)
    ],
    remainder='drop' # Drop any columns not specified
)

# 4. Machine Learning Models and Training
def train_models(df):
    X = df.drop(columns=['ticket_id', 'sentiment', 'category'])
    y_sentiment = df['sentiment']
    y_category = df['category']

    # Split data
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X, y_sentiment, test_size=0.2, random_state=42, stratify=y_sentiment)
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_category, test_size=0.2, random_state=42, stratify=y_category)

    # Sentiment Model Pipeline
    sentiment_model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    sentiment_model_pipeline.fit(X_train_s, y_train_s)
    s_accuracy = sentiment_model_pipeline.score(X_test_s, y_test_s)
    print(f"Sentiment Model Accuracy: {s_accuracy:.4f}")

    # Category Model Pipeline
    category_model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    category_model_pipeline.fit(X_train_c, y_train_c)
    c_accuracy = category_model_pipeline.score(X_test_c, y_test_c)
    print(f"Category Model Accuracy: {c_accuracy:.4f}")

    return sentiment_model_pipeline, category_model_pipeline

# 5. Prediction and Smart Routing Logic
def smart_route_ticket(ticket_data_series, sentiment_model, category_model):
    # Ensure ticket_data_series is a DataFrame for prediction
    ticket_df = pd.DataFrame([ticket_data_series])

    predicted_sentiment = sentiment_model.predict(ticket_df)[0]
    predicted_category = category_model.predict(ticket_df)[0]
    ticket_priority = ticket_data_series['priority']

    routing_department = "General Support"
    escalation_flag = False
    notes = []

    if predicted_sentiment == 'negative':
        notes.append("Negative customer sentiment detected.")
        if ticket_priority == 'high':
            routing_department = "Escalated Support - Senior Agent"
            escalation_flag = True
            notes.append("High priority ticket with negative sentiment, immediate escalation required.")
        elif predicted_category in ['technical', 'billing']:
            routing_department = f"Specialized {predicted_category.capitalize()} Support"
            notes.append(f"Negative sentiment in {predicted_category} category.")
        else:
            routing_department = "Customer Relations Team"
    elif predicted_sentiment == 'positive':
        notes.append("Positive customer sentiment detected.")
        routing_department = "Customer Success Team" if predicted_category == 'general' else f"Feedback Team - {predicted_category.capitalize()}"
    else: # Neutral sentiment
        notes.append("Neutral customer sentiment detected.")

    if ticket_priority == 'high' and not escalation_flag:
        routing_department = f"Priority {routing_department}"
        notes.append("High priority ticket.")
        escalation_flag = True # Even if not negative, high priority needs attention

    # Refine routing based on category
    if predicted_category == 'technical':
        routing_department = "Technical Support L1" if 'Senior Agent' not in routing_department else routing_department
        notes.append("Categorized as Technical issue.")
    elif predicted_category == 'billing':
        routing_department = "Billing Department" if 'Senior Agent' not in routing_department else routing_department
        notes.append("Categorized as Billing issue.")
    elif predicted_category == 'account':
        routing_department = "Account Management" if 'Senior Agent' not in routing_department else routing_department
        notes.append("Categorized as Account issue.")
    elif predicted_category == 'product':
        routing_department = "Product Support" if 'Senior Agent' not in routing_department else routing_department
        notes.append("Categorized as Product issue.")
    elif predicted_category == 'general':
        routing_department = "General Inquiry Support" if 'Senior Agent' not in routing_department else routing_department
        notes.append("Categorized as General inquiry.")

    return {
        'predicted_sentiment': predicted_sentiment,
        'predicted_category': predicted_category,
        'routing_department': routing_department,
        'escalation_flag': escalation_flag,
        'notes': notes
    }

# Main execution flow
if __name__ == "__main__":
    print("Simulating customer support tickets...")
    df_tickets = simulate_customer_tickets(n_samples=500)
    print("Simulated Data Head:")
    print(df_tickets.head())
    print("\nSimulated Data Info:")
    df_tickets.info()

    print("\nTraining sentiment and category models...")
    sentiment_model, category_model = train_models(df_tickets)
    print("Models trained successfully.")

    # --- Demonstrate smart routing with new example tickets ---
    print("\n--- Demonstrating Smart Routing ---")

    # Example 1: High priority, negative sentiment, technical issue
    new_ticket_1 = pd.Series({
        'ticket_id': 'T9001',
        'description': "My internet has been completely dead for 2 days! I'm losing business because of this! Unacceptable!",
        'priority': 'high',
        'customer_age': 45,
        'product_id': 'P101'
    })
    print(f"\nNew Ticket 1: {new_ticket_1['description']}")
    routing_result_1 = smart_route_ticket(new_ticket_1, sentiment_model, category_model)
    print(f"Routing Result 1: {routing_result_1}")

    # Example 2: Medium priority, neutral/positive sentiment, billing issue
    new_ticket_2 = pd.Series({
        'ticket_id': 'T9002',
        'description': "I just received my bill and it seems a bit higher than expected. Could you review the charges?",
        'priority': 'medium',
        'customer_age': 30,
        'product_id': 'P103'
    })
    print(f"\nNew Ticket 2: {new_ticket_2['description']}")
    routing_result_2 = smart_route_ticket(new_ticket_2, sentiment_model, category_model)
    print(f"Routing Result 2: {routing_result_2}")

    # Example 3: Low priority, positive sentiment, general inquiry
    new_ticket_3 = pd.Series({
        'ticket_id': 'T9003',
        'description': "Just wanted to say I love your new product update! It's fantastic!",
        'priority': 'low',
        'customer_age': 28,
        'product_id': 'P105'
    })
    print(f"\nNew Ticket 3: {new_ticket_3['description']}")
    routing_result_3 = smart_route_ticket(new_ticket_3, sentiment_model, category_model)
    print(f"Routing Result 3: {routing_result_3}")

    # Example 4: Missing customer age, medium priority, technical
    new_ticket_4 = pd.Series({
        'ticket_id': 'T9004',
        'description': "My software keeps crashing after the latest patch. I need a fix.",
        'priority': 'medium',
        'customer_age': np.nan, # Missing age
        'product_id': 'P102'
    })
    print(f"\nNew Ticket 4: {new_ticket_4['description']}")
    routing_result_4 = smart_route_ticket(new_ticket_4, sentiment_model, category_model)
    print(f"Routing Result 4: {routing_result_4}")
