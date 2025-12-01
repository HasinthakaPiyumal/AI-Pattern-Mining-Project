import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

def generate_synthetic_data(num_samples=1000):
    """Generates synthetic patient data for demonstration."""
    np.random.seed(42)
    data = {
        "age": np.random.randint(18, 90, num_samples),
        "num_chronic_conditions": np.random.randint(0, 5, num_samples),
        "length_of_stay": np.random.randint(1, 30, num_samples),
        "insurance_type": np.random.choice(["Private", "Medicare", "Medicaid", "None"], num_samples, p=[0.4, 0.3, 0.2, 0.1]),
        "gender": np.random.choice(["Male", "Female"], num_samples),
        "readmitted": np.random.randint(0, 2, num_samples) # Target variable
    }
    df = pd.DataFrame(data)

    # Introduce some correlations for a more realistic scenario
    df['readmitted'] = df.apply(
        lambda row: 1 if (row['num_chronic_conditions'] > 2 and row['age'] > 60) or (row['length_of_stay'] > 15 and row['insurance_type'] == 'Medicaid') else row['readmitted'],
        axis=1
    )
    df['readmitted'] = df['readmitted'].apply(lambda x: 1 if np.random.rand() < 0.3 else x) # Add some random noise

    return df

def preprocess_data(df):
    """Simple preprocessing for categorical features using one-hot encoding."""
    df_processed = pd.get_dummies(df, columns=["insurance_type", "gender"], drop_first=True)
    return df_processed

def train_model(X_train, y_train):
    """Trains a Gradient Boosting Classifier as a black-box model."""
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    return model

def plot_partial_dependence(model, X, features_to_plot, title_suffix=""):
    """Generates and plots Partial Dependence Plots."""
    print(f"Generating PDPs for features: {features_to_plot}")
    n_cols = 3 # Number of columns for subplots
    n_rows = int(np.ceil(len(features_to_plot) / n_cols))

    fig, ax = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    ax = ax.flatten() if n_rows > 1 or n_cols > 1 else [ax]

    # Use PartialDependenceDisplay for plotting
    display = PartialDependenceDisplay.from_estimator(
        model, X, features=features_to_plot, kind="average", ax=ax
    )

    plt.suptitle(f"Partial Dependence Plots for Patient Readmission Risk {title_suffix}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    return display

if __name__ == "__main__":
    print("Starting Patient Readmission Risk Predictor Interpretability script...")

    # 1. Generate Synthetic Data
    df = generate_synthetic_data(num_samples=1000)
    print("\nSynthetic Data Head:")
    print(df.head())

    # 2. Preprocess Data
    X = df.drop("readmitted", axis=1)
    y = df["readmitted"]
    X_processed = preprocess_data(X)

    # Ensure all columns are numeric after one-hot encoding
    # Align columns between training features and features for PDP
    # This step is crucial for `PartialDependenceDisplay`
    
    # 3. Train a Black-Box Model
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    print(f"\nModel trained. Accuracy on test set: {model.score(X_test, y_test):.4f}")

    # 4. Define features for PDP and plot
    # Ensure these features exist in the processed dataframe
    features_to_plot_single = [
        "age",
        "num_chronic_conditions",
        "length_of_stay",
        "insurance_type_Private", # Example of one-hot encoded feature
        "gender_Male" # Example of one-hot encoded feature
    ]
    # Filter to ensure features exist in X_processed.columns
    features_to_plot_single = [f for f in features_to_plot_single if f in X_processed.columns]

    if features_to_plot_single:
        plot_partial_dependence(model, X_processed, features_to_plot_single, title_suffix="(Single Features)")

    # Plot two-feature PDPs (interactions)
    features_to_plot_pairs = [
        ("age", "num_chronic_conditions"),
        ("length_of_stay", "insurance_type_Medicaid") # Example pair with one-hot encoded
    ]
    # Filter to ensure all features in pairs exist
    filtered_pairs = []
    for f1, f2 in features_to_plot_pairs:
        if f1 in X_processed.columns and f2 in X_processed.columns:
            filtered_pairs.append((f1, f2))
    
    if filtered_pairs:
        plot_partial_dependence(model, X_processed, filtered_pairs, title_suffix="(Interactions)")

    print("\nPartial Dependence Plots generated and displayed. This provides global insights into the model's predictions.")