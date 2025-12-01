import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

class BlackBoxClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(random_state=42)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

class LACEExplainer:
    def __init__(self, black_box_model, training_data_raw, training_data_encoded, feature_names_raw, feature_names_encoded, categorical_features_raw, numerical_features_raw, target_name):
        self.black_box_model = black_box_model
        self.training_data_raw = training_data_raw
        self.training_data_encoded = training_data_encoded
        self.feature_names_raw = feature_names_raw
        self.feature_names_encoded = feature_names_encoded
        self.categorical_features_raw = categorical_features_raw
        self.numerical_features_raw = numerical_features_raw
        self.target_name = target_name
        self.nn_model = NearestNeighbors(n_neighbors=5, algorithm='auto') # k will be tuned later
        self.nn_model.fit(self.training_data_encoded)

    def _get_local_neighborhood(self, instance_encoded, k):
        distances, indices = self.nn_model.kneighbors(instance_encoded.reshape(1, -1), n_neighbors=k)
        local_X_encoded = self.training_data_encoded.iloc[indices[0]]
        local_X_raw = self.training_data_raw.iloc[indices[0]]
        local_y_proba = self.black_box_model.predict_proba(local_X_encoded)
        return local_X_raw, local_X_encoded, local_y_proba

    def _train_local_surrogate(self, local_X_encoded, local_y_proba):
        # Use the predicted probabilities of the black-box model as labels for the surrogate
        surrogate_model = DecisionTreeClassifier(max_depth=3, random_state=42)
        surrogate_model.fit(local_X_encoded, np.argmax(local_y_proba, axis=1))
        return surrogate_model

    def _extract_local_rules(self, surrogate_model, feature_names_encoded):
        tree_rules = export_text(surrogate_model, feature_names=feature_names_encoded)
        rules = [line.strip() for line in tree_rules.split('\n') if '|---' in line or 'class:' in line]
        return rules

    def _quantify_prediction_difference(self, instance_encoded, instance_raw, target_class_idx, feature_names_raw):
        base_prediction = self.black_box_model.predict_proba(instance_encoded.reshape(1, -1))[0, target_class_idx]
        contributions = {}

        # Single attribute contributions
        for i, feature_name in enumerate(feature_names_raw):
            perturbed_instance_encoded = instance_encoded.copy()
            original_value_raw = instance_raw[feature_name]

            if feature_name in self.numerical_features_raw:
                # Replace with mean for numerical features
                mean_val = self.training_data_raw[feature_name].mean()
                # Need to re-encode after perturbation if it affects OHE features
                # For simplicity, we directly modify the encoded instance if possible, or re-encode from raw
                # This part is complex due to OHE. A simpler approach for this example:
                # Perturb raw, then re-encode the single instance
                perturbed_raw = instance_raw.copy()
                perturbed_raw[feature_name] = mean_val
                
                # Re-encode the single perturbed instance using the fitted encoders
                temp_df = pd.DataFrame([perturbed_raw], columns=self.training_data_raw.columns)
                temp_df_encoded = self._encode_instance(temp_df, self.ohe, self.scaler)
                perturbed_instance_encoded = temp_df_encoded.iloc[0].values

            elif feature_name in self.categorical_features_raw:
                # Replace with mode for categorical features
                mode_val = self.training_data_raw[feature_name].mode()[0]
                perturbed_raw = instance_raw.copy()
                perturbed_raw[feature_name] = mode_val

                temp_df = pd.DataFrame([perturbed_raw], columns=self.training_data_raw.columns)
                temp_df_encoded = self._encode_instance(temp_df, self.ohe, self.scaler)
                perturbed_instance_encoded = temp_df_encoded.iloc[0].values

            else: # Should not happen if all features are categorized
                continue

            perturbed_prediction = self.black_box_model.predict_proba(perturbed_instance_encoded.reshape(1, -1))[0, target_class_idx]
            contributions[feature_name] = base_prediction - perturbed_prediction

        # Note: Interaction contributions (attribute-value conjunctions) would require more complex perturbation
        # and combination generation, which is beyond the scope of this simplified example for brevity.
        return contributions

    def _encode_instance(self, instance_df, ohe, scaler):
        # Apply One-Hot Encoding
        instance_categorical_encoded = ohe.transform(instance_df[self.categorical_features_raw])
        instance_categorical_df = pd.DataFrame(instance_categorical_encoded, columns=ohe.get_feature_names_out(self.categorical_features_raw))

        # Apply Standard Scaling
        instance_numerical_scaled = scaler.transform(instance_df[self.numerical_features_raw])
        instance_numerical_df = pd.DataFrame(instance_numerical_scaled, columns=self.numerical_features_raw)

        # Combine encoded and scaled features
        instance_encoded_df = pd.concat([instance_numerical_df, instance_categorical_df], axis=1)
        return instance_encoded_df

    def explain_instance(self, instance_raw, k=None):
        # Determine target class for explanation
        instance_df = pd.DataFrame([instance_raw], columns=self.feature_names_raw)
        instance_encoded = self._encode_instance(instance_df, self.ohe, self.scaler).iloc[0].values
        
        black_box_prediction_proba = self.black_box_model.predict_proba(instance_encoded.reshape(1, -1))[0]
        predicted_class_idx = np.argmax(black_box_prediction_proba)
        predicted_class_proba = black_box_prediction_proba[predicted_class_idx]

        if k is None:
            # Simple dynamic k: e.g., square root of neighborhood size
            k = int(np.sqrt(len(self.training_data_encoded)))
            if k < 2: k = 2 # Ensure at least 2 neighbors

        local_X_raw, local_X_encoded, local_y_proba = self._get_local_neighborhood(instance_encoded, k)
        local_surrogate = self._train_local_surrogate(local_X_encoded, local_y_proba)
        local_rules = self._extract_local_rules(local_surrogate, self.feature_names_encoded)
        prediction_contributions = self._quantify_prediction_difference(instance_encoded, instance_raw, predicted_class_idx, self.feature_names_raw)

        explanation = {
            "instance_raw": instance_raw,
            "black_box_prediction_proba": black_box_prediction_proba.tolist(),
            "predicted_class_index": int(predicted_class_idx),
            "predicted_class_probability": float(predicted_class_proba),
            "k_neighbors_used": k,
            "local_rules": local_rules,
            "feature_contributions": prediction_contributions,
        }
        return explanation

# --- Example Usage --- 

# 1. Generate some synthetic healthcare data
def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)
    data = {
        "Age": np.random.randint(20, 80, num_samples),
        "Gender": np.random.choice(["Male", "Female"], num_samples),
        "BMI": np.random.normal(25, 5, num_samples),
        "Blood_Pressure": np.random.randint(90, 180, num_samples),
        "Cholesterol": np.random.normal(200, 30, num_samples),
        "Smoker": np.random.choice(["Yes", "No"], num_samples),
        "Family_History": np.random.choice(["Yes", "No"], num_samples),
        "Exercise": np.random.randint(0, 7, num_samples), # days per week
        "Diagnosis": np.random.choice(["No Disease", "Heart Disease", "Diabetes"], num_samples, p=[0.7, 0.2, 0.1])
    }
    df = pd.DataFrame(data)

    # Introduce some correlations for disease
    df.loc[(df["Age"] > 50) & (df["Smoker"] == "Yes"), "Diagnosis"] = "Heart Disease"
    df.loc[(df["BMI"] > 30) & (df["Family_History"] == "Yes"), "Diagnosis"] = "Diabetes"

    return df

df_raw = generate_synthetic_data(1000)

# Define features and target
target_name = "Diagnosis"
feature_names_raw = [col for col in df_raw.columns if col != target_name]
categorical_features_raw = ["Gender", "Smoker", "Family_History"]
numerical_features_raw = ["Age", "BMI", "Blood_Pressure", "Cholesterol", "Exercise"]

X_raw = df_raw[feature_names_raw]
y = df_raw[target_name]

# Preprocessing: One-Hot Encoding for categorical, Standard Scaling for numerical
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
scaler = StandardScaler()

X_categorical_encoded = ohe.fit_transform(X_raw[categorical_features_raw])
X_categorical_df = pd.DataFrame(X_categorical_encoded, columns=ohe.get_feature_names_out(categorical_features_raw))

X_numerical_scaled = scaler.fit_transform(X_raw[numerical_features_raw])
X_numerical_df = pd.DataFrame(X_numerical_scaled, columns=numerical_features_raw)

X_encoded = pd.concat([X_numerical_df, X_categorical_df], axis=1)
feature_names_encoded = X_encoded.columns.tolist()

# Train the black-box model
black_box_model = BlackBoxClassifier()
black_box_model.fit(X_encoded, y)

# Create LACE Explainer
lace_explainer = LACEExplainer(black_box_model, X_raw, X_encoded, feature_names_raw, feature_names_encoded, categorical_features_raw, numerical_features_raw, target_name)
lace_explainer.ohe = ohe # Store encoders for instance re-encoding
lace_explainer.scaler = scaler

# Select an instance to explain
instance_to_explain_raw = X_raw.iloc[5]
print(f"\nInstance to explain (Raw):\n{instance_to_explain_raw}\n")

# Get explanation
explanation = lace_explainer.explain_instance(instance_to_explain_raw)

print("\n--- LACE Explanation ---")
print(f"Black-box prediction probabilities: {explanation['black_box_prediction_proba']}")
print(f"Predicted class index: {explanation['predicted_class_index']}")
print(f"Predicted class probability: {explanation['predicted_class_probability']}")
print(f"K neighbors used: {explanation['k_neighbors_used']}")

print("\nLocal Rules (from surrogate model):")
for rule in explanation["local_rules"]:
    print(f"- {rule}")

print("\nFeature Contributions (Prediction Difference):")
for feature, contribution in explanation["feature_contributions"].items():
    print(f"- {feature}: {contribution:.4f}")
