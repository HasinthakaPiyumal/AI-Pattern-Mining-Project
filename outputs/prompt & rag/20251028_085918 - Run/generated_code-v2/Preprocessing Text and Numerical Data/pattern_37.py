import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Download NLTK data (if not already downloaded)
# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

class PreprocessingModule:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        self.numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        self.categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        self.preprocessor = None

    def _preprocess_text(self, text):
        if not isinstance(text, str):
            return ""
        tokens = word_tokenize(text.lower())
        tokens = [word for word in tokens if word.isalpha() and word not in self.stop_words]
        lemmatized_tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        return " ".join(lemmatized_tokens)

    def fit_text_vectorizer(self, text_series):
        processed_text = text_series.apply(self._preprocess_text)
        self.tfidf_vectorizer.fit(processed_text)

    def transform_text(self, text_series):
        processed_text = text_series.apply(self._preprocess_text)
        return self.tfidf_vectorizer.transform(processed_text)

    def fit_numerical_preprocessor(self, df, numerical_cols, categorical_cols):
        transformers = []
        if numerical_cols:
            transformers.append(('num', self.numerical_transformer, numerical_cols))
        if categorical_cols:
            transformers.append(('cat', self.categorical_transformer, categorical_cols))
            
        if transformers:
            self.preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')
            self.preprocessor.fit(df)
        else:
            self.preprocessor = None

    def transform_numerical_data(self, df):
        if self.preprocessor:
            return self.preprocessor.transform(df)
        else:
            return df

    def full_preprocessing_pipeline(self, text_data, numerical_df, text_col, numerical_cols, categorical_cols):
        # Text Preprocessing
        print("Fitting and transforming text data...")
        self.fit_text_vectorizer(text_data[text_col])
        text_features = self.transform_text(text_data[text_col])
        text_feature_names = self.tfidf_vectorizer.get_feature_names_out()
        text_df = pd.DataFrame(text_features.toarray(), columns=text_feature_names, index=numerical_df.index)

        # Numerical Preprocessing
        print("Fitting and transforming numerical data...")
        self.fit_numerical_preprocessor(numerical_df, numerical_cols, categorical_cols)
        transformed_numerical_array = self.transform_numerical_data(numerical_df)

        # Get feature names for numerical data after transformation
        numerical_feature_names = []
        if self.preprocessor:
            for name, transformer, cols in self.preprocessor.transformers_:
                if name == 'num':
                    numerical_feature_names.extend(cols)
                elif name == 'cat':
                    numerical_feature_names.extend(self.preprocessor.named_transformers_['cat'].get_feature_names_out(cols))
            
            # Handle remainder columns if any
            remainder_cols_indices = self.preprocessor.transformers_[-1][2]
            if remainder_cols_indices and isinstance(remainder_cols_indices, np.ndarray):
                original_cols = numerical_df.columns.tolist()
                remainder_col_names = [original_cols[i] for i in remainder_cols_indices]
                numerical_feature_names.extend(remainder_col_names)
        else:
            numerical_feature_names = numerical_df.columns.tolist()

        numerical_df_transformed = pd.DataFrame(transformed_numerical_array, columns=numerical_feature_names, index=numerical_df.index)
        
        # Combine features
        print("Combining preprocessed features...")
        combined_features_df = pd.concat([text_df, numerical_df_transformed], axis=1)
        
        print("Preprocessing complete.")
        return combined_features_df


if __name__ == '__main__':
    # Example Usage
    # Sample Data
    review_data = {
        'review_id': [1, 2, 3, 4, 5],
        'text_review': [
            "This product is amazing! I love it so much. Highly recommend.",
            "It's okay, not great, not bad. Just average quality.",
            "Terrible experience. The item broke after one use.",
            "Good value for money. Fast delivery and nice packaging.",
            "Absolutely fantastic! Will buy again soon."
        ]
    }
    product_data = {
        'product_id': [101, 102, 103, 104, 105],
        'rating': [5, 3, 1, 4, 5],
        'price': [25.99, 12.50, 5.00, 30.00, 28.00],
        'sales_volume': [1200, 500, 50, 800, 1500],
        'category': ['Electronics', 'Home', 'Electronics', 'Books', 'Electronics'],
        'stock_level': [100, 200, np.nan, 150, 120]
    }

    text_df = pd.DataFrame(review_data)
    numerical_df = pd.DataFrame(product_data)

    # Initialize the Preprocessing Module
    preprocessor = PreprocessingModule()

    # Define columns
    text_column = 'text_review'
    numerical_columns = ['rating', 'price', 'sales_volume', 'stock_level']
    categorical_columns = ['category']

    # Run the full preprocessing pipeline
    processed_data = preprocessor.full_preprocessing_pipeline(text_df, numerical_df, text_column, numerical_columns, categorical_columns)

    print("\nShape of processed data:", processed_data.shape)
    print("\nFirst 5 rows of processed data:\n", processed_data.head())
    print("\nColumns of processed data:\n", processed_data.columns.tolist())
    print("\nMissing values in processed data:\n", processed_data.isnull().sum().sum())
