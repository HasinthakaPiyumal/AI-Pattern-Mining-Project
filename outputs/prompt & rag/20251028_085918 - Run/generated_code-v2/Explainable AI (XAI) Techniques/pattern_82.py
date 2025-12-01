import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier, export_text
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, request, jsonify

app = Flask(__name__)

# --- 1. Mock Black-Box Medical Diagnosis Model and Data --- #
def generate_synthetic_ehr_data(num_samples=1000, num_features=10):
    np.random.seed(42)
    data = {
        f"feature_{i}": np.random.rand(num_samples) * 100 for i in range(num_features - 2)
    }
    data["age"] = np.random.randint(20, 80, num_samples)
    data["gender"] = np.random.choice([0, 1], num_samples) # 0 for female, 1 for male
    df = pd.DataFrame(data)
    
    # Create a target variable (e.g., disease presence) with some correlation
    df["disease_target"] = (df["feature_0"] * 0.1 + df["age"] * 0.05 + df["gender"] * 5 + np.random.randn(num_samples) * 5 > 10).astype(int)
    return df


class BlackBoxModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.feature_names = None

    def train(self, X_train, y_train):
        self.feature_names = X_train.columns.tolist()
        self.model.fit(X_train, y_train)

    def predict_proba(self, X):
        if isinstance(X, pd.Series): # Handle single instance
            X = X.to_frame().T
        elif isinstance(X, np.ndarray):
            if X.ndim == 1: # Handle single instance numpy array
                X = X.reshape(1, -1)
            X = pd.DataFrame(X, columns=self.feature_names)
        return self.model.predict_proba(X)

# Generate training data and train the black-box model
synthetic_data = generate_synthetic_ehr_data()
X_train_global = synthetic_data.drop("disease_target", axis=1)
y_train_global = synthetic_data["disease_target"]

black_box_model = BlackBoxModel()
black_box_model.train(X_train_global, y_train_global)

# --- LACE Explainer Implementation --- #
class LACEExplainer:
    def __init__(self, black_box_model, training_data_X, feature_names):
        self.black_box_model = black_box_model
        self.training_data_X = training_data_X
        self.feature_names = feature_names
        self.nn_model = NearestNeighbors(metric='euclidean')
        self.nn_model.fit(training_data_X)

    def _find_local_neighborhood(self, instance, k):
        distances, indices = self.nn_model.kneighbors(instance.to_frame().T, n_neighbors=k)
        neighborhood_indices = indices[0]
        local_X = self.training_data_X.iloc[neighborhood_indices]
        return local_X

    def _train_local_surrogate(self, local_X, local_predictions_proba):
        # For simplicity, we use DecisionTreeClassifier as a rule-based surrogate
        # L3 would involve more sophisticated rule extraction
        local_y_binary = (local_predictions_proba[:, 1] > 0.5).astype(int) # Convert probabilities to binary labels
        surrogate_model = DecisionTreeClassifier(max_depth=3, random_state=42)
        surrogate_model.fit(local_X, local_y_binary)
        return surrogate_model

    def _calculate_prediction_difference(self, instance, original_prediction_proba, local_X_neighborhood):
        contributions = {}
        original_proba_positive = original_prediction_proba[0, 1]

        # Iterate through each feature to calculate individual contributions
        for feature in self.feature_names:
            instance_perturbed = instance.copy()
            
            # Simulate feature omission by replacing with neighborhood mean/median
            # For simplicity, using mean from the local neighborhood as perturbation
            if pd.api.types.is_numeric_dtype(local_X_neighborhood[feature]):
                perturbed_value = local_X_neighborhood[feature].mean()
            else:
                # For categorical features, use the mode or a specific 'unknown' value
                perturbed_value = local_X_neighborhood[feature].mode()[0]
            
            instance_perturbed[feature] = perturbed_value
            
            perturbed_proba = self.black_box_model.predict_proba(instance_perturbed)
            perturbed_proba_positive = perturbed_proba[0, 1]
            
            # Prediction difference: P(original) - P(omitted)
            contributions[feature] = original_proba_positive - perturbed_proba_positive
            
        # This section would be extended for interaction terms (patterns)
        # For this example, we focus on individual features.
        # Extracting patterns from a Decision Tree can be done by traversing it,
        # but quantifying their specific 'prediction difference' requires careful marginalization.
        
        return contributions

    def _tune_k(self, instance, initial_k=10, max_k=50, step=5, stability_threshold=0.05):
        # A simple K-tuning mechanism: find a K where local predictions are stable
        best_k = initial_k
        previous_prediction = None

        for k_val in range(initial_k, max_k + 1, step):
            local_X = self._find_local_neighborhood(instance, k_val)
            
            if len(local_X) < k_val: # Not enough neighbors, stop increasing k
                break
            
            # Get black-box predictions for the local neighborhood
            local_predictions_proba = self.black_box_model.predict_proba(local_X)
            
            # Train a temporary surrogate to get a local prediction trend
            temp_surrogate = self._train_local_surrogate(local_X, local_predictions_proba)
            current_prediction_proba = temp_surrogate.predict_proba(instance.to_frame().T)[0, 1]
            
            if previous_prediction is not None and abs(current_prediction_proba - previous_prediction) < stability_threshold:
                best_k = k_val
                break
            previous_prediction = current_prediction_proba
            best_k = k_val # Keep increasing if not stable enough

        return best_k


    def explain_instance(self, instance_df, k=None):
        instance = instance_df.squeeze() # Ensure it's a Series

        # 1. Get black-box model's prediction for the instance
        original_prediction_proba = self.black_box_model.predict_proba(instance)
        predicted_class_proba = original_prediction_proba[0, 1]
        predicted_class = self.black_box_model.model.predict(instance.to_frame().T)[0]

        # 2. Tune K if not provided
        if k is None:
            k = self._tune_k(instance)
        print(f"Using K = {k} for explanation.")

        # 3. Generate local neighborhood
        local_X_neighborhood = self._find_local_neighborhood(instance, k)
        local_predictions_proba = self.black_box_model.predict_proba(local_X_neighborhood)

        # 4. Train local surrogate model
        surrogate_model = self._train_local_surrogate(local_X_neighborhood, local_predictions_proba)

        # 5. Extract rules (qualitative explanation)
        rules_text = export_text(surrogate_model, feature_names=self.feature_names)

        # 6. Calculate prediction difference (quantitative explanation)
        feature_contributions = self._calculate_prediction_difference(instance, original_prediction_proba, local_X_neighborhood)
        
        return {
            "instance_prediction_proba": predicted_class_proba,
            "instance_predicted_class": predicted_class,
            "k_used": k,
            "local_rules": rules_text,
            "feature_contributions": feature_contributions
        }

    def plot_contributions(self, contributions, instance_prediction_proba):
        features = list(contributions.keys())
        values = list(contributions.values())

        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = np.arange(len(features))
        
        ax.barh(y_pos, values, align='center', color=['green' if v > 0 else 'red' for v in values])
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()  # features with higher contribution on top
        ax.set_xlabel('Prediction Difference (Change in P(Disease))')
        ax.set_title(f'Feature Contributions to Prediction (P(Disease) = {instance_prediction_proba:.2f})')
        ax.axvline(0, color='grey', linewidth=0.8)

        # Save plot to a bytes buffer
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')


# Initialize the LACE Explainer with the trained black-box model and global training data
lace_explainer = LACEExplainer(black_box_model, X_train_global, black_box_model.feature_names)

# --- API Endpoints --- #
@app.route("/explain", methods=["POST"])
def explain_diagnosis():
    data = request.json
    if not data:
        return jsonify({"error": "No patient data provided."}), 400

    try:
        patient_data_df = pd.DataFrame([data], columns=black_box_model.feature_names)
        explanation_results = lace_explainer.explain_instance(patient_data_df)

        # Generate visualization
        plot_base64 = lace_explainer.plot_contributions(
            explanation_results["feature_contributions"],
            explanation_results["instance_prediction_proba"]
        )
        explanation_results["contribution_plot_base64"] = plot_base64

        return jsonify(explanation_results), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def health_check():
    return "Medical Diagnosis Explainer is running!"

if __name__ == '__main__':
    # Example of how to run the explainer directly without the API
    # Select a random instance from the training data for explanation
    # patient_instance = X_train_global.sample(1, random_state=10).iloc[0]
    # print(f"\nExplaining instance:\n{patient_instance.to_dict()}")
    # explanation = lace_explainer.explain_instance(patient_instance.to_frame().T)
    # print("\n--- Explanation Results ---")
    # print(f"Black-box prediction (P(Disease)): {explanation['instance_prediction_proba']:.4f}")
    # print(f"Predicted Class: {explanation['instance_predicted_class']}")
    # print(f"K used: {explanation['k_used']}")
    # print("\nLocal Rules (from Surrogate Model):\n", explanation['local_rules'])
    # print("\nFeature Contributions (Prediction Difference):\n", explanation['feature_contributions'])

    # # Plotting for direct run (uncomment to save locally)
    # plot_base64_str = lace_explainer.plot_contributions(
    #     explanation["feature_contributions"],
    #     explanation["instance_prediction_proba"]
    # )
    # with open("contribution_plot.png", "wb") as f:
    #     f.write(base64.b64decode(plot_base64_str))
    # print("Contribution plot saved as contribution_plot.png")

    print("Starting Flask API...")
    app.run(debug=True, host='0.0.0.0', port=5000)
