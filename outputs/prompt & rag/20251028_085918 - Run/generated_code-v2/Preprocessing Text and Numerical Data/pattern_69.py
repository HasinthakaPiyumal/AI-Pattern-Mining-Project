import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.sparse import hstack

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)


def preprocess_numerical_data(df, numerical_cols):
    imputer = SimpleImputer(strategy="mean")
    df[numerical_cols] = imputer.fit_transform(df[numerical_cols])
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df


def preprocess_text_data(text_series):
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    processed_texts = []
    for text in text_series:
        tokens = nltk.word_tokenize(text.lower())
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in stop_words]
        processed_texts.append(" ".join(tokens))

    vectorizer = TfidfVectorizer(max_features=1000)
    text_features = vectorizer.fit_transform(processed_texts)
    return text_features, vectorizer


def train_evaluate_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(solver="liblinear", random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return accuracy, precision, recall, f1


if __name__ == "__main__":
    np.random.seed(42)
    num_samples = 100

    data = {
        "Age": np.random.randint(20, 80, num_samples),
        "Blood_Pressure_Sys": np.random.randint(90, 180, num_samples),
        "Blood_Pressure_Dia": np.random.randint(60, 120, num_samples),
        "Cholesterol": np.random.randint(150, 250, num_samples),
        "Patient_Notes": [
            "Patient presented with mild fever and sore throat. No other significant symptoms.",
            "Experiencing severe chest pain for the last 2 hours. History of heart disease.",
            "Routine check-up, no complaints. Healthy lifestyle mentioned.",
            "Diabetic patient with high blood sugar levels. Needs medication adjustment.",
            "Headache and nausea. Suspected viral infection.",
            "Follow-up appointment. Blood pressure is stable.",
            "New onset of joint pain and fatigue. Under investigation.",
            "Minor cut on hand, dressed it. No infection signs.",
            "Asthma exacerbation, prescribed inhaler. Advised rest.",
            "Annual physical exam. All vitals within normal range."
        ] * (num_samples // 10),
        "Disease_Present": np.random.randint(0, 2, num_samples) # 0 for No Disease, 1 for Disease
    }
    df = pd.DataFrame(data)

    df.loc[::10, "Blood_Pressure_Sys"] = np.nan # Introduce some missing numerical values
    df.loc[::5, "Cholesterol"] = np.nan

    numerical_cols = ["Age", "Blood_Pressure_Sys", "Blood_Pressure_Dia", "Cholesterol"]
    text_col = "Patient_Notes"
    target_col = "Disease_Present"

    df_numerical_preprocessed = preprocess_numerical_data(df.copy(), numerical_cols)
    text_features_sparse, _ = preprocess_text_data(df[text_col])

    numerical_features_dense = df_numerical_preprocessed[numerical_cols].values

    combined_features = hstack([numerical_features_dense, text_features_sparse])

    X = combined_features
    y = df[target_col]

    accuracy, precision, recall, f1 = train_evaluate_model(X, y)

    print(f"Model Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")