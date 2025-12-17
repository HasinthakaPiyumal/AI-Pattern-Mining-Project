import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_numerical_data(df: pd.DataFrame):
    """
    Applies preprocessing transformations to numerical and categorical data.

    Args:
        df (pd.DataFrame): The input DataFrame containing numerical and categorical columns.

    Returns:
        pd.DataFrame: The DataFrame with preprocessed numerical and one-hot encoded categorical features.
    """

    # Identify numerical and categorical columns
    numerical_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    # Create preprocessing pipelines for numerical and categorical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Create a preprocessor object using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ], 
        remainder='passthrough' # Keep other columns not specified (if any)
    )

    # Fit and transform the data
    transformed_data = preprocessor.fit_transform(df)

    # Get feature names after one-hot encoding for categorical columns
    onehot_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_cols)
    all_feature_names = list(numerical_cols) + list(onehot_feature_names)

    # Convert the transformed array back to a DataFrame
    # Note: ColumnTransformer output is a NumPy array, so we need to reconstruct DataFrame
    # If 'remainder' is 'passthrough', then the remaining columns will be at the end of the array.
    # For simplicity and clarity in this example, we assume all relevant columns are handled.
    # For a robust solution, careful mapping of column names from `remainder='passthrough'` is needed.
    
    # If there are columns in 'remainder', their names should also be collected.
    # Here we are assuming that all columns are either numerical_cols or categorical_cols for simplicity of name reconstruction.

    processed_df = pd.DataFrame(transformed_data, columns=all_feature_names, index=df.index)
    
    return processed_df

if __name__ == '__main__':
    # Example Usage:
    data = {
        'Age': [30, 45, 22, 55, None, 38],
        'Weight': [70, 85, 60, None, 75, 90],
        'Lab_Result_A': [10.2, 15.5, 8.1, 12.0, 9.5, 18.0],
        'Diagnosis_Category': ['Cardio', 'Neuro', 'Ortho', 'Cardio', 'Neuro', None],
        'Gender': ['Male', 'Female', 'Female', 'Male', 'Female', 'Male']
    }
    df = pd.DataFrame(data)

    print("Original DataFrame:")
    print(df)
    print("\nMissing values before preprocessing:")
    print(df.isnull().sum())

    processed_df = preprocess_numerical_data(df)

    print("\nProcessed DataFrame (numerical and categorical):")
    print(processed_df.head())
    print("\nShape of processed DataFrame:", processed_df.shape)
    print("\nMissing values after preprocessing:")
    print(processed_df.isnull().sum())
