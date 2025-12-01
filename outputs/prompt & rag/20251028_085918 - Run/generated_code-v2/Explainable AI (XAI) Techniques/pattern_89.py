import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_dummy_model(X, y):
    """Trains a dummy RandomForestClassifier."""
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model

def generate_ice_data(model, patient_instance: pd.Series, feature_to_vary: str, feature_range: np.ndarray) -> pd.DataFrame:
    """Generates data points for an Individual Conditional Expectation (ICE) plot.

    Args:
        model: The trained machine learning model with a .predict_proba method.
        patient_instance: A pandas Series representing a single patient's features.
        feature_to_vary: The name of the feature to vary for the ICE plot.
        feature_range: A numpy array of values to test for the feature_to_vary.

    Returns:
        A pandas DataFrame with 'feature_value' and 'predicted_probability' columns.
    """
    ice_data = []
    original_features = patient_instance.drop(feature_to_vary, errors='ignore').to_dict()

    for value in feature_range:
        # Create a modified instance with the varied feature value
        modified_instance_dict = {**original_features, feature_to_vary: value}
        modified_instance_df = pd.DataFrame([modified_instance_dict], columns=model.feature_names_in_)

        # Predict probability for the positive class
        prediction_proba = model.predict_proba(modified_instance_df)[:, 1][0]
        ice_data.append({'feature_value': value, 'predicted_probability': prediction_proba})

    return pd.DataFrame(ice_data)

if __name__ == "__main__":
    # 1. Create a dummy dataset (simulating patient data)
    np.random.seed(42)
    data = {
        'Age': np.random.randint(20, 80, 100),
        'BloodPressure': np.random.randint(90, 180, 100),
        'Cholesterol': np.random.randint(150, 250, 100),
        'Dosage': np.random.uniform(10, 100, 100),
        'Response': np.random.randint(0, 2, 100) # 0: No Response, 1: Response
    }
    df = pd.DataFrame(data)

    X = df.drop('Response', axis=1)
    y = df['Response']

    # Split data for training (optional, but good practice)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Train a dummy model
    model = train_dummy_model(X_train, y_train)

    # 3. Select a patient instance from the test set for explanation
    patient_idx = 0
    patient_instance = X_test.iloc[patient_idx]
    original_prediction = model.predict_proba(patient_instance.to_frame().T)[:, 1][0]
    print(f"Original patient instance:\n{patient_instance}")
    print(f"Original predicted response probability: {original_prediction:.3f}\n")

    # 4. Define the feature to vary and its range
    feature_to_explain = 'Dosage'
    # Determine a relevant range for the feature based on the dataset
    min_val = X[feature_to_explain].min()
    max_val = X[feature_to_explain].max()
    feature_range = np.linspace(min_val, max_val, 50) # 50 points for the plot

    # 5. Generate ICE plot data for the selected patient and feature
    ice_plot_df = generate_ice_data(model, patient_instance, feature_to_explain, feature_range)

    print(f"ICE plot data for patient {patient_idx} varying '{feature_to_explain}':\n")
    print(ice_plot_df.head())
    print("...")
    print(ice_plot_df.tail())

    # In a real application, this 'ice_plot_df' would be sent to the frontend for visualization.