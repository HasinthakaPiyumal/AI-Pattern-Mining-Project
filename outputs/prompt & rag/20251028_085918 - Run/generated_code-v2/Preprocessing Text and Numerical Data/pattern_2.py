import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

def preprocess_numerical_data(df, numerical_cols):
    imputer = SimpleImputer(strategy="mean")
    df[numerical_cols] = imputer.fit_transform(df[numerical_cols])
    
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df

def preprocess_text_data(df, text_col):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    
    def clean_text(text):
        text = str(text).lower()  # Convert to string and lowercase
        text = re.sub(r"[^a-z]", " ", text) # Remove non-alphabetic characters
        tokens = nltk.word_tokenize(text)
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        return " ".join(tokens)
    
    df[text_col + "_cleaned"] = df[text_col].apply(clean_text)
    
    tfidf_vectorizer = TfidfVectorizer(max_features=5000) # Limit features for simplicity
    tfidf_features = tfidf_vectorizer.fit_transform(df[text_col + "_cleaned"])
    
    tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=tfidf_vectorizer.get_feature_names_out(), index=df.index)
    
    return df, tfidf_df

def main():
    # Simulate a dataset
    data = {
        "patient_id": range(100),
        "age": np.random.randint(20, 90, 100),
        "blood_pressure_systolic": np.random.randint(90, 180, 100),
        "heart_rate": np.random.randint(60, 120, 100),
        "lab_result_glucose": np.random.uniform(70, 200, 100),
        "doctor_notes": [
            "Patient presented with chest pain and shortness of breath. History of hypertension.",
            "Follow-up visit. Patient is stable, advised medication adherence.",
            "Emergency admission due to severe headache. Suspected stroke, undergoing tests.",
            "Routine check-up. No significant changes. Advised lifestyle modifications.",
            "Discharge summary: patient recovered well. Instructions provided for home care."
        ] * 20, # Repeat notes to have enough data
        "readmission_risk": np.random.randint(0, 2, 100) # 0: Low, 1: High
    }
    
    # Introduce some missing values for demonstration
    for col in ["age", "blood_pressure_systolic", "lab_result_glucose"]:
        data[col][np.random.choice(100, 10, replace=False)] = np.nan
        
    df = pd.DataFrame(data)
    
    print("Original Data Head:")
    print(df.head())
    print("\nMissing values before imputation:")
    print(df.isnull().sum())
    
    # Numerical Preprocessing
    numerical_cols = ["age", "blood_pressure_systolic", "heart_rate", "lab_result_glucose"]
    df_processed_numerical = preprocess_numerical_data(df.copy(), numerical_cols)
    
    print("\nNumerical Data Head after Preprocessing:")
    print(df_processed_numerical[numerical_cols].head())
    print("\nMissing values after numerical preprocessing:")
    print(df_processed_numerical.isnull().sum())
    
    # Text Preprocessing
    df_temp, df_tfidf = preprocess_text_data(df.copy(), "doctor_notes")
    
    print("\nCleaned Doctor Notes Head:")
    print(df_temp[["doctor_notes", "doctor_notes_cleaned"]].head())
    print("\nTF-IDF Features Shape:", df_tfidf.shape)
    print("TF-IDF Features Head (Sample):")
    print(df_tfidf.iloc[:, :5].head()) # Print first 5 TF-IDF columns
    
    # Combine all features for model training
    # Drop original text column and the temporary cleaned column before merging TF-IDF
    df_combined = df_processed_numerical.drop(columns=["doctor_notes"])
    df_combined = pd.concat([df_combined.reset_index(drop=True), df_tfidf.reset_index(drop=True)], axis=1)
    
    print("\nCombined Features Head (Sample):")
    print(df_combined.head())
    print("Combined Features Shape:", df_combined.shape)
    
    # Prepare data for model training
    X = df_combined.drop(columns=["patient_id", "readmission_risk"])
    y = df_combined["readmission_risk"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    print("\nTraining a Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print("\nModel Performance (Logistic Regression):")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()