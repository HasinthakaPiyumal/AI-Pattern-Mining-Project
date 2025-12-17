import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

data = {
    "Patient_ID": [1, 2, 3, 4, 5, 6, 7],
    "Age": [30, 45, 60, None, 25, 50, 70],
    "Lab_Result_A": [10.5, 12.1, 8.9, 11.2, None, 13.0, 9.5],
    "Vital_Sign_B": [70, 85, 65, 72, 80, 78, None],
    "Medication_Type": ["TypeA", "TypeB", "TypeA", "TypeC", "TypeB", "TypeA", "TypeC"],
    "Doctor_Notes": [
        "Patient reports mild headache and fatigue. No fever.",
        "High blood pressure observed. Advised diet change.",
        "Chronic cough and shortness of breath. History of smoking.",
        "Minor cut on hand. Prescribed antibiotics.",
        "Sudden severe abdominal pain. Referred to specialist.",
        "Routine check-up, all normal. No complaints.",
        "Fever, body aches, and sore throat. Suspected flu."
    ],
    "Diagnosis": ["Headache", "Hypertension", "COPD", "Injury", "Abdominal Pain", "Healthy", "Flu"]
}
df = pd.DataFrame(data)

X = df.drop("Diagnosis", axis=1)
y = df["Diagnosis"]

numerical_features = ["Age", "Lab_Result_A", "Vital_Sign_B"]
categorical_features = ["Medication_Type"]
text_features = ["Doctor_Notes"]

numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

stopwords_set = set(stopwords.words("english"))
stemmer = PorterStemmer()

def custom_tokenizer(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [stemmer.stem(word) for word in tokens if word.isalpha() and word not in stopwords_set]
    return tokens

text_transformer = TfidfVectorizer(
    tokenizer=custom_tokenizer,
    preprocessor=None,
    stop_words=None,
    max_features=100
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features),
        ("text", text_transformer, text_features[0])
    ],
    remainder="drop"
)

X_preprocessed = preprocessor.fit_transform(X)

print("Original X shape:", X.shape)
print("Preprocessed X shape:", X_preprocessed.shape)
print("\nExample of preprocessed data (first 5 rows and a few columns):\n", X_preprocessed[:5, :])

print("\nPreprocessing complete. Data is ready for ML model training.")