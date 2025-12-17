import pandas as pd
import numpy as np
import re
import nltk
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_numerical_data(df, numerical_cols, categorical_cols_for_ohe):
    imputer = SimpleImputer(strategy='mean')
    df[numerical_cols] = imputer.fit_transform(df[numerical_cols])

    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    if categorical_cols_for_ohe:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoded_features = encoder.fit_transform(df[categorical_cols_for_ohe])
        encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols_for_ohe), index=df.index)
        df = pd.concat([df.drop(columns=categorical_cols_for_ohe), encoded_df], axis=1)
    
    return df

def preprocess_text_data(df, text_col):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    def clean_text(text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)  # Remove punctuation and numbers
        return text

    def tokenize_and_stem(text):
        tokens = nltk.word_tokenize(text)
        filtered_tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
        return " ".join(filtered_tokens)
    
    df[text_col] = df[text_col].apply(clean_text)
    df[text_col] = df[text_col].apply(tokenize_and_stem)

    vectorizer = TfidfVectorizer(max_features=1000) # Limit features for demonstration
    tfidf_matrix = vectorizer.fit_transform(df[text_col]).toarray()
    tfidf_df = pd.DataFrame(tfidf_matrix, columns=vectorizer.get_feature_names_out(), index=df.index)
    
    return tfidf_df


if __name__ == "__main__":
    # 1. Mock Data Ingestion
    data = {
        'patient_id': range(1, 11),
        'age': [35, 67, 45, np.nan, 72, 58, 60, 49, 81, 53],
        'weight': [70, 85, 65, 78, 92, np.nan, 73, 68, 88, 75],
        'blood_pressure_systolic': [120, 145, 130, 118, 150, 135, 128, 122, 160, 130],
        'gender': ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F'],
        'hospital_id': ['HospA', 'HospB', 'HospA', 'HospC', 'HospB', 'HospA', 'HospC', 'HospB', 'HospA', 'HospC'],
        'doctor_notes': [
            "Patient presented with mild fever and cough. Discharged after observation.",
            "Elderly patient with heart condition, prescribed new medication. Follow-up in 2 weeks.",
            "Routine check-up, no major concerns. Advised on diet.",
            "Severe headache, admitted for tests. Results pending.",
            "Diabetic patient, insulin dosage adjusted. Monitored for 3 days.",
            "Broken arm, cast applied. Physiotherapy recommended.",
            "Cold symptoms, advised rest and fluids. No severe issues.",
            "High blood pressure, lifestyle changes suggested. Regular monitoring.",
            "Stroke patient, recovering well. Discharge planning initiated.",
            "Minor injury, wound cleaned and dressed. Sent home."
        ],
        'readmitted': [0, 1, 0, 1, 1, 0, 0, 1, 1, 0] # Target variable (for demonstration)
    }
    raw_df = pd.DataFrame(data)

    numerical_features = ['age', 'weight', 'blood_pressure_systolic']
    categorical_ohe_features = ['gender', 'hospital_id']
    text_feature = 'doctor_notes'
    target_feature = 'readmitted'

    # Separate target for later use
    X = raw_df.drop(columns=[target_feature, 'patient_id'])
    y = raw_df[target_feature]

    # Make copies for independent preprocessing
    numerical_df = X[numerical_features + categorical_ohe_features].copy()
    text_df = X[[text_feature]].copy()

    print("\n--- Original Data ---")
    print(raw_df.head())

    # 2. Numerical Data Preprocessing
    print("\n--- Preprocessing Numerical Data ---")
    processed_numerical_df = preprocess_numerical_data(numerical_df, numerical_features, categorical_ohe_features)
    print("Processed Numerical Features Shape:", processed_numerical_df.shape)
    print(processed_numerical_df.head())

    # 3. Text Data Preprocessing
    print("\n--- Preprocessing Text Data ---")
    processed_text_df = preprocess_text_data(text_df, text_feature)
    print("Processed Text Features Shape:", processed_text_df.shape)
    print(processed_text_df.head())

    # 4. Feature Combination
    print("\n--- Combining Features ---")
    final_features_df = pd.concat([processed_numerical_df, processed_text_df], axis=1)
    print("Final Combined Features Shape:", final_features_df.shape)
    print(final_features_df.head())

    # 5. Predictive Model (Placeholder)
    print("\n--- Predictive Model Placeholder ---")
    print("Combined features are ready for training a machine learning model.")
    print("Example: model = RandomForestClassifier()")
    print("         model.fit(final_features_df, y)")
    print("         predictions = model.predict(test_data_features)")