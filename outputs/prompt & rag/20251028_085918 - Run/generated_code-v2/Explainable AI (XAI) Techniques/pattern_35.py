import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class TreatmentPredictor:
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        self.features = None

    def train(self, X, y):
        self.features = X.columns.tolist()
        self.model.fit(X, y)
        print("Model trained successfully.")

    def predict(self, X):
        return self.model.predict(X)

    def generate_ice_plot_data(self, patient_instance, feature_name, num_steps=50):
        if feature_name not in self.features:
            raise ValueError(f"Feature \'{feature_name}\' not in model features.")

        # Create a copy of the patient instance to vary the feature
        ice_data_df = pd.DataFrame([patient_instance.copy()])

        # Determine the range for the chosen feature
        feature_min = self.get_feature_range(feature_name)['min']
        feature_max = self.get_feature_range(feature_name)['max']

        feature_values = np.linspace(feature_min, feature_max, num_steps)
        predictions = []

        for val in feature_values:
            temp_instance = patient_instance.copy()
            temp_instance[feature_name] = val
            predictions.append(self.predict(pd.DataFrame([temp_instance]))) # Ensure input is DataFrame

        return feature_values, np.array(predictions).flatten()

    def get_feature_range(self, feature_name):
        # For a real application, this would come from the training data or domain knowledge
        # For simulation, we'll use a simple heuristic or predefined ranges
        if feature_name == 'age':
            return {'min': 20, 'max': 80}
        elif feature_name == 'drug_dosage':
            return {'min': 50, 'max': 500}
        elif feature_name == 'biomarker_level':
            return {'min': 0.1, 'max': 5.0}
        elif feature_name == 'comorbidity_score':
            return {'min': 0, 'max': 10}
        else:
            # Default for other numerical features
            return {'min': 0, 'max': 100}

def simulate_patient_data(num_patients=1000):
    np.random.seed(42)
    data = {
        'age': np.random.randint(25, 75, num_patients),
        'gender': np.random.choice(['Male', 'Female'], num_patients),
        'drug_dosage': np.random.uniform(100, 400, num_patients),
        'biomarker_level': np.random.uniform(0.5, 4.5, num_patients),
        'comorbidity_score': np.random.randint(0, 8, num_patients),
        'ethnicity': np.random.choice(['Caucasian', 'Asian', 'African', 'Other'], num_patients),
    }
    df = pd.DataFrame(data)

    # Simulate treatment response with some interaction and non-linearity
    df['treatment_response'] = (
        10 + 0.3 * df['age']
        + 0.05 * df['drug_dosage']
        + 2 * df['biomarker_level']
        - 0.8 * df['comorbidity_score']
        + (df['age'] * df['drug_dosage'] * 0.001) # Interaction term
        + np.random.normal(0, 5, num_patients)
    )
    # Introduce some heterogeneity based on gender for instance
    df.loc[df['gender'] == 'Female', 'treatment_response'] += 5 * df.loc[df['gender'] == 'Female', 'biomarker_level']

    return df

def preprocess_data(df):
    df_encoded = pd.get_dummies(df, columns=['gender', 'ethnicity'], drop_first=True)
    return df_encoded

def plot_ice(feature_values, predictions, patient_id, feature_name, actual_prediction=None):
    plt.figure(figsize=(10, 6))
    plt.plot(feature_values, predictions, color='blue', alpha=0.8, label=f'Patient {patient_id} ICE Plot')
    plt.xlabel(f'{feature_name} Value')
    plt.ylabel('Predicted Treatment Response')
    plt.title(f'Individual Conditional Expectation (ICE) Plot for Patient {patient_id} (Feature: {feature_name})')
    if actual_prediction is not None:
        plt.axhline(y=actual_prediction, color='red', linestyle='--', label=f'Actual Prediction ({actual_prediction:.2f})')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # 1. Data Preparation
    print("Simulating patient data...")
    patient_df = simulate_patient_data(num_patients=1000)
    patient_df_processed = preprocess_data(patient_df.drop(columns=['treatment_response']))
    target = patient_df['treatment_response']

    X_train, X_test, y_train, y_test = train_test_split(patient_df_processed, target, test_size=0.2, random_state=42)

    # 2. Machine Learning Model Layer
    print("Training treatment predictor model...")
    predictor = TreatmentPredictor()
    predictor.train(X_train, y_train)

    # Evaluate the model (optional)
    y_pred = predictor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Model Mean Squared Error on test set: {mse:.2f}")

    # 5. User Interface (Command-line Demo)
    print("\n--- ICE Plot Generation Demo ---")
    while True:
        try:
            patient_idx = int(input(f"Enter patient ID (0 to {len(patient_df) - 1}) to analyze (or -1 to exit): "))
            if patient_idx == -1:
                break
            if not (0 <= patient_idx < len(patient_df)):
                print("Invalid patient ID. Please try again.")
                continue

            available_features = [col for col in X_train.columns if 'gender_' not in col and 'ethnicity_' not in col] # Exclude one-hot encoded
            print(f"Available features for ICE plots: {', '.join(available_features)}")
            feature_to_explore = input("Enter the feature name to explore (e.g., age, drug_dosage, biomarker_level): ")

            if feature_to_explore not in available_features:
                print("Invalid feature name or feature is one-hot encoded. Please choose from the list.")
                continue

            selected_patient_original = patient_df.iloc[patient_idx]
            selected_patient_processed = patient_df_processed.iloc[patient_idx]

            # Get the actual prediction for the selected patient
            actual_prediction = predictor.predict(pd.DataFrame([selected_patient_processed]))[0]
            print(f"\nPredicted treatment response for Patient {patient_idx}: {actual_prediction:.2f}")

            # 3. Explanations Layer (ICE Plot Generator) & 4. Visualization Layer
            print(f"Generating ICE plot for Patient {patient_idx} and feature \'{feature_to_explore}\'...")
            feature_values, ice_predictions = predictor.generate_ice_plot_data(
                selected_patient_processed,
                feature_to_explore
            )
            plot_ice(feature_values, ice_predictions, patient_idx, feature_to_explore, actual_prediction)

        except ValueError:
            print("Invalid input. Please enter a number for patient ID.")
        except Exception as e:
            print(f"An error occurred: {e}")

    print("Exiting ICE Plot Predictor.")
