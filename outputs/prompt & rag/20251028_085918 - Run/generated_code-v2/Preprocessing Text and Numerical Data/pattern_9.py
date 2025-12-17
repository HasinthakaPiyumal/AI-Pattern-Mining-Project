import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class PatientDataPreprocessor:
    def __init__(self):
        self.numerical_imputer = SimpleImputer(strategy='mean')
        self.numerical_scaler = StandardScaler()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000)
        self.stop_words = set(stopwords.words('english'))
        self.porter_stemmer = PorterStemmer()
        self.wordnet_lemmatizer = WordNetLemmatizer()

    def ingest_data(self, numerical_data_path, text_data_path=None):
        numerical_df = pd.read_csv(numerical_data_path)
        text_data = []
        if text_data_path:
            with open(text_data_path, 'r', encoding='utf-8') as f:
                text_data = [line.strip() for line in f.readlines()]
        return numerical_df, text_data

    def preprocess_numerical_data(self, numerical_df):
        numerical_features = numerical_df.select_dtypes(include=np.number)
        imputed_data = self.numerical_imputer.fit_transform(numerical_features)
        scaled_data = self.numerical_scaler.fit_transform(imputed_data)
        return pd.DataFrame(scaled_data, columns=numerical_features.columns)

    def preprocess_text_data(self, text_list):
        processed_texts = []
        for text in text_list:
            tokens = word_tokenize(text.lower())
            filtered_tokens = [token for token in tokens if token.isalnum() and token not in self.stop_words]
            lemmas = [self.wordnet_lemmatizer.lemmatize(token) for token in filtered_tokens]
            processed_texts.append(' '.join(lemmas))
        return self.tfidf_vectorizer.fit_transform(processed_texts)

    def integrate_features(self, numerical_features_df, text_features_matrix):
        # Convert sparse matrix to dense DataFrame for concatenation if text features exist
        if text_features_matrix is not None and text_features_matrix.shape[0] > 0:
            text_features_df = pd.DataFrame(text_features_matrix.toarray(), 
                                            columns=[f'text_feature_{i}' for i in range(text_features_matrix.shape[1])])
            # Ensure numerical and text dataframes have compatible index/length for concatenation
            # Assuming a 1:1 correspondence for simplicity, or handle complex joins if needed
            # For this example, we'll assume they correspond row-wise.
            # If numerical_features_df might be shorter due to dropped rows during imputation etc.,
            # then resampling or more complex alignment would be needed.
            min_rows = min(len(numerical_features_df), len(text_features_df))
            integrated_df = pd.concat([numerical_features_df.iloc[:min_rows].reset_index(drop=True),
                                       text_features_df.iloc[:min_rows].reset_index(drop=True)], axis=1)
        else:
            integrated_df = numerical_features_df.copy()
        return integrated_df

    def run_preprocessing_pipeline(self, numerical_data_path, text_data_path=None):
        numerical_df, text_data = self.ingest_data(numerical_data_path, text_data_path)
        
        preprocessed_numerical_df = self.preprocess_numerical_data(numerical_df)
        
        text_features_matrix = None
        if text_data and len(text_data) > 0:
            text_features_matrix = self.preprocess_text_data(text_data)
        
        final_dataset = self.integrate_features(preprocessed_numerical_df, text_features_matrix)
        return final_dataset

if __name__ == '__main__':
    # Example Usage:
    # Create dummy data files for demonstration
    with open('sample_numerical_data.csv', 'w') as f:
        f.write('age,weight,blood_pressure_sys,blood_pressure_dia,cholesterol\n')
        f.write('65,70,120,80,200\n')
        f.write('45,85,140,90,240\n')
        f.write('70,60,110,70,180\n')
        f.write('50,75,130,85,220\n')
        f.write('60,NaN,125,82,210\n')

    with open('sample_text_data.txt', 'w') as f:
        f.write('Patient complains of severe headache and mild fever.\n')
        f.write('Regular checkup. No significant issues found. Advised healthy diet.\n')
        f.write('History of hypertension. Medication prescribed. Follow up in 3 months.\n')
        f.write('Mild allergic reaction to new medication. Switched to alternative.\n')

    preprocessor = PatientDataPreprocessor()
    processed_data = preprocessor.run_preprocessing_pipeline(
        numerical_data_path='sample_numerical_data.csv',
        text_data_path='sample_text_data.txt'
    )

    print("\n--- Preprocessed Data ---")
    print(processed_data.head())
    print("\nShape of preprocessed data:", processed_data.shape)

    # Example with only numerical data
    preprocessor_numerical_only = PatientDataPreprocessor()
    processed_numerical_only = preprocessor_numerical_only.run_preprocessing_pipeline(
        numerical_data_path='sample_numerical_data.csv'
    )
    print("\n--- Preprocessed Numerical-Only Data ---")
    print(processed_numerical_only.head())
    print("\nShape of preprocessed numerical-only data:", processed_numerical_only.shape)
