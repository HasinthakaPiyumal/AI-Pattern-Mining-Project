import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
import joblib

def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)
    age = np.random.randint(20, 90, num_samples)
    num_chronic_conditions = np.random.randint(0, 5, num_samples)
    length_of_stay = np.random.randint(1, 30, num_samples)
    discharge_destinations = ['Home', 'Rehab Facility', 'Skilled Nursing Facility', 'Other']
    discharge_destination = np.random.choice(discharge_destinations, num_samples)

    # Simulate readmission risk based on features
    # Higher age, more chronic conditions, longer stay, and certain discharge destinations increase risk
    readmitted_prob = (age * 0.005 + num_chronic_conditions * 0.1 + length_of_stay * 0.01 + 
                       (pd.Series(discharge_destination).map({'Home': 0.1, 'Rehab Facility': 0.3, 'Skilled Nursing Facility': 0.5, 'Other': 0.2}))).clip(0.1, 0.9)

    readmitted = (np.random.rand(num_samples) < readmitted_prob).astype(int)

    data = pd.DataFrame({
        'age': age,
        'num_chronic_conditions': num_chronic_conditions,
        'length_of_stay': length_of_stay,
        'discharge_destination': discharge_destination,
        'readmitted': readmitted
    })
    return data

def train_model(X_train, y_train, preprocessor):
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
    ])
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"Model ROC AUC: {roc_auc:.4f}")
    return roc_auc

def plot_pdps(model, X_processed, features_to_plot):
    fig, ax = plt.subplots(figsize=(15, 5), ncols=len(features_to_plot))
    PartialDependenceDisplay.from_estimator(model, X_processed, features_to_plot, 
                                            kind='average', ax=ax, 
                                            feature_names=X_processed.columns, 
                                            target=1)
    fig.suptitle('Partial Dependence Plots for Readmission Risk Prediction', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    print("Generating synthetic patient data...")
    data = generate_synthetic_data(num_samples=2000)
    print("Data generated successfully. Sample head:")
    print(data.head())

    X = data.drop('readmitted', axis=1)
    y = data['readmitted']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Define preprocessing steps
    categorical_features = ['discharge_destination']
    numerical_features = ['age', 'num_chronic_conditions', 'length_of_stay']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    print("Training RandomForestClassifier model...")
    model = train_model(X_train, y_train, preprocessor)
    print("Model training complete.")

    print("Evaluating model performance...")
    evaluate_model(model, X_test, y_test)

    # Save the trained model
    joblib.dump(model, 'readmission_risk_model.joblib')
    print("Model saved as 'readmission_risk_model.joblib'")

    # For PDP, we need the preprocessed X_train to get feature names correctly
    # and then apply the preprocessor to the original X to ensure consistency with the model
    X_train_processed_df = pd.DataFrame(model.named_steps['preprocessor'].transform(X_train),
                                      columns=numerical_features + 
                                      list(model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)))
    
    print("Generating Partial Dependence Plots...")
    features_to_plot = ['age', 'num_chronic_conditions', 'length_of_stay', 'discharge_destination_Rehab Facility']
    
    # Ensure features_to_plot exist in the processed dataframe
    valid_features_to_plot = [f for f in features_to_plot if f in X_train_processed_df.columns]
    if len(valid_features_to_plot) < len(features_to_plot):
        print(f"Warning: Some requested PDP features were not found in processed data: {list(set(features_to_plot) - set(valid_features_to_plot))}")

    if valid_features_to_plot:
        plot_pdps(model.named_steps['classifier'], X_train_processed_df, valid_features_to_plot)
    else:
        print("No valid features to plot PDPs.")
    print("PDP generation complete.")