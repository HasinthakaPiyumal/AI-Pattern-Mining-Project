import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay

# --- 1. Data Loading and Preprocessing Module ---

def load_and_preprocess_data():
    # Simulate a credit risk dataset
    np.random.seed(42)
    n_samples = 1000
    data = {
        'Age': np.random.randint(20, 70, n_samples),
        'Income': np.random.randint(30000, 150000, n_samples),
        'Loan_Amount': np.random.randint(5000, 100000, n_samples),
        'Credit_Score': np.random.randint(300, 850, n_samples),
        'Employment_Status': np.random.choice(['Employed', 'Unemployed', 'Self-Employed', 'Retired'], n_samples, p=[0.6, 0.1, 0.2, 0.1]),
        'Loan_Term': np.random.choice([12, 24, 36, 48, 60], n_samples),
        'Default': np.random.randint(0, 2, n_samples) # 0 for No Default, 1 for Default
    }
    df = pd.DataFrame(data)

    # Define target and features
    X = df.drop('Default', axis=1)
    y = df['Default']

    # Identify numerical and categorical features
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include='object').columns.tolist()

    # Create preprocessing pipelines for numerical and categorical features
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Create a column transformer to apply different transformations to different columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Data loaded and preprocessed successfully.")
    return X_train, X_test, y_train, y_test, preprocessor, numerical_features, categorical_features

# --- 2. Credit Risk Model Training Module ---

def train_credit_risk_model(X_train, y_train, preprocessor):
    # Create a pipeline that first preprocesses and then trains the model
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
    ])

    # Train the model
    model_pipeline.fit(X_train, y_train)
    
    print("Credit risk model trained successfully.")
    return model_pipeline

# --- 3. Partial Dependence Plot (PDP) Generation & 4. Visualization Module ---

def generate_and_visualize_pdps(model, X_test, features_to_plot, numerical_features, categorical_features):
    # Get feature names after one-hot encoding
    ohe_feature_names = model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
    all_feature_names = numerical_features + list(ohe_feature_names)

    # Ensure the features_to_plot are present in the transformed feature names
    # For simplicity, we will plot only original features for PDPs.
    # If plotting one-hot encoded features, they need to be handled carefully.
    
    print("Generating Partial Dependence Plots...")

    # Plot single-feature PDPs
    for feature in features_to_plot:
        if feature in numerical_features or feature in categorical_features:
            fig, ax = plt.subplots(figsize=(8, 6))
            PartialDependenceDisplay.from_estimator(
                model,
                X_test,
                features=[feature],
                feature_names=all_feature_names, # Use all_feature_names for consistent indexing
                target=1, # Plot PDP for the 'Default' class (1)
                grid_resolution=50,
                ax=ax
            )
            ax.set_title(f"Partial Dependence Plot for {feature}")
            ax.set_ylabel("Partial Dependence (Probability of Default)")
            plt.tight_layout()
            plt.show()
        else:
            print(f"Warning: Feature '{feature}' not found in original dataset for PDP generation.")

    # Example of plotting a two-feature PDP (if applicable and desired)
    # For this synthetic data, let's pick two numerical features for a 2D PDP
    if 'Age' in numerical_features and 'Income' in numerical_features:
        print("Generating 2D Partial Dependence Plot for Age and Income...")
        fig, ax = plt.subplots(figsize=(10, 8))
        PartialDependenceDisplay.from_estimator(
            model,
            X_test,
            features=[('Age', 'Income')],
            feature_names=all_feature_names, # Use all_feature_names for consistent indexing
            target=1, # Plot PDP for the 'Default' class (1)
            grid_resolution=50,
            ax=ax
        )
        ax.set_title("2D Partial Dependence Plot for Age and Income")
        plt.tight_layout()
        plt.show()
    else:
        print("Cannot generate 2D PDP for Age and Income as they are not both numerical features.")

    print("Partial Dependence Plots generated and displayed.")


if __name__ == '__main__':
    # Flow of Execution
    X_train, X_test, y_train, y_test, preprocessor, numerical_features, categorical_features = load_and_preprocess_data()
    model_pipeline = train_credit_risk_model(X_train, y_train, preprocessor)

    # Define features for which to generate PDPs
    # This list can be customized based on features of interest
    features_to_interpret = ['Age', 'Income', 'Credit_Score', 'Employment_Status', 'Loan_Amount']
    
    generate_and_visualize_pdps(model_pipeline, X_test, features_to_interpret, numerical_features, categorical_features)
