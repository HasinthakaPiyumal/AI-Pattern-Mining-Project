import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


data_reviews = {
    "customer_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "review_text": [
        "This product is amazing! Highly recommend.",
        "Terrible experience, very slow delivery and broken item.",
        "Good value for money, but sizing is off.",
        "Love it, perfect fit and quality.",
        "Not bad, average purchase.",
        "Disappointed with the customer service.",
        "Fantastic product, exceeded my expectations.",
        "Awful, never buying again from this store.",
        "Decent product, quick shipping.",
        "So happy with my new gadget!"
    ],
    "sentiment_label": [1, 0, 1, 1, 0, 0, 1, 0, 1, 1]
}
customer_reviews_df = pd.DataFrame(data_reviews)


data_numerical = {
    "customer_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "age": [30, 45, 22, 38, 55, 29, np.nan, 41, 33, 27],
    "purchase_count": [10, 3, 7, 15, 2, 8, 12, 1, 9, 11],
    "avg_order_value": [50.5, 120.0, 35.7, 75.2, 200.0, 40.0, 60.0, 150.0, 80.0, 45.0],
    "customer_segment": ["Gold", "Silver", "Bronze", "Gold", "Platinum", "Silver", "Gold", "Bronze", "Silver", "Gold"],
    "churn_label": [0, 1, 0, 0, 1, 1, 0, 1, 0, 0]
}
customer_numerical_df = pd.DataFrame(data_numerical)


stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

customer_reviews_df["processed_review_text"] = customer_reviews_df["review_text"].apply(preprocess_text)

tfidf_vectorizer = TfidfVectorizer(max_features=100)
tfidf_features = tfidf_vectorizer.fit_transform(customer_reviews_df["processed_review_text"])
tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
tfidf_df["customer_id"] = customer_reviews_df["customer_id"]


numerical_features = ["age", "purchase_count", "avg_order_value"]
categorical_features = ["customer_segment"]

numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ],
    remainder="passthrough"
)

numerical_processed = preprocessor.fit_transform(customer_numerical_df)

transformed_numerical_features = numerical_features
transformed_categorical_features = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
all_transformed_features = list(transformed_numerical_features) + list(transformed_categorical_features)

numerical_processed_df = pd.DataFrame(numerical_processed, columns=all_transformed_features)
numerical_processed_df["customer_id"] = customer_numerical_df["customer_id"]


customer_reviews_df["review_length"] = customer_reviews_df["review_text"].apply(len)
review_length_df = customer_reviews_df[["customer_id", "review_length"]]


combined_features_df = pd.merge(numerical_processed_df, tfidf_df, on="customer_id", how="left")
combined_features_df = pd.merge(combined_features_df, review_length_df, on="customer_id", how="left")

combined_features_df = combined_features_df.fillna(0)


X_sentiment = tfidf_features
y_sentiment = customer_reviews_df["sentiment_label"]

X_train_sent, X_test_sent, y_train_sent, y_test_sent = train_test_split(X_sentiment, y_sentiment, test_size=0.3, random_state=42)

sentiment_model = LogisticRegression(max_iter=1000)
sentiment_model.fit(X_train_sent, y_train_sent)


y_churn = pd.merge(combined_features_df[["customer_id"]], customer_numerical_df[["customer_id", "churn_label"]], on="customer_id", how="left")["churn_label"]
X_churn = combined_features_df.drop("customer_id", axis=1)

X_train_churn, X_test_churn, y_train_churn, y_test_churn = train_test_split(X_churn, y_churn, test_size=0.3, random_state=42)

churn_model = RandomForestClassifier(random_state=42)
churn_model.fit(X_train_churn, y_train_churn)


y_pred_sent = sentiment_model.predict(X_test_sent)
print("Sentiment Analysis Model Performance:")
print(f"Accuracy: {accuracy_score(y_test_sent, y_pred_sent):.4f}")
print("Classification Report:")
print(classification_report(y_test_sent, y_pred_sent))
print("\n")

y_pred_churn = churn_model.predict(X_test_churn)
y_prob_churn = churn_model.predict_proba(X_test_churn)[:, 1]

print("Churn Prediction Model Performance:")
print(f"Accuracy: {accuracy_score(y_test_churn, y_pred_churn):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test_churn, y_prob_churn):.4f}")
print("Classification Report:")
print(classification_report(y_test_churn, y_pred_churn))