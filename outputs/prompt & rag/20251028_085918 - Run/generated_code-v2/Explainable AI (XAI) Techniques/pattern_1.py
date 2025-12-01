
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.pipeline import Pipeline
import joblib

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.numerical_cols = []
        self.categorical_cols = []
        self.feature_names_out = []

    def fit(self, X):
        self.numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
        self.categorical_cols = X.select_dtypes(include='object').columns.tolist()

        if self.numerical_cols:
            self.scaler.fit(X[self.numerical_cols])
        if self.categorical_cols:
            self.onehot_encoder.fit(X[self.categorical_cols])
        
        self._set_feature_names_out(X)
        return self

    def transform(self, X):
        X_processed_parts = []

        if self.numerical_cols:
            X_scaled = self.scaler.transform(X[self.numerical_cols])
            X_processed_parts.append(pd.DataFrame(X_scaled, columns=self.numerical_cols, index=X.index))
        
        if self.categorical_cols:
            X_onehot = self.onehot_encoder.transform(X[self.categorical_cols])
            onehot_feature_names = self.onehot_encoder.get_feature_names_out(self.categorical_cols)
            X_processed_parts.append(pd.DataFrame(X_onehot, columns=onehot_feature_names, index=X.index))
        
        X_processed = pd.concat(X_processed_parts, axis=1)
        return X_processed

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def _set_feature_names_out(self, X):
        if self.numerical_cols:
            self.feature_names_out.extend(self.numerical_cols)
        if self.categorical_cols:
            self.feature_names_out.extend(self.onehot_encoder.get_feature_names_out(self.categorical_cols))

    def inverse_transform_instance(self, preprocessed_instance_df, original_instance_template):
        # Assumes preprocessed_instance_df is a single row DataFrame of preprocessed values
        # original_instance_template is a single row DataFrame with original feature names and types
        
        decoded_features = {}

        # Inverse transform numerical features
        if self.numerical_cols:
            scaled_numerical_features = preprocessed_instance_df[self.numerical_cols]
            original_numerical_values = self.scaler.inverse_transform(scaled_numerical_features)
            for i, col in enumerate(self.numerical_cols):
                decoded_features[col] = original_numerical_values[0, i]
        
        # Inverse transform categorical features
        if self.categorical_cols:
            # Create a dummy array for one-hot decoder
            onehot_data = np.zeros((1, len(self.onehot_encoder.get_feature_names_out())))
            onehot_feature_names = self.onehot_encoder.get_feature_names_out(self.categorical_cols)
            
            for i, feature_name in enumerate(onehot_feature_names):
                if feature_name in preprocessed_instance_df.columns:
                    onehot_data[0, i] = preprocessed_instance_df[feature_name].iloc[0]
            
            # Decode the one-hot encoded features
            original_categorical_values = self.onehot_encoder.inverse_transform(onehot_data)
            for i, col in enumerate(self.categorical_cols):
                decoded_features[col] = original_categorical_values[0, i]
                
        return pd.DataFrame([decoded_features], columns=original_instance_template.columns)

def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)
    data = {
        'age': np.random.randint(20, 80, num_samples),
        'gender': np.random.choice(['Male', 'Female'], num_samples),
        'blood_pressure': np.random.normal(120, 15, num_samples).astype(int),
        'cholesterol': np.random.normal(200, 30, num_samples).astype(int),
        'diabetes': np.random.choice([0, 1], num_samples, p=[0.85, 0.15]),
        'smoker': np.random.choice([0, 1], num_samples, p=[0.7, 0.3]),
    }
    df = pd.DataFrame(data)

    # Generate disease_risk with some correlation
    df['disease_risk'] = 0
    df.loc[(df['age'] > 60) | (df['blood_pressure'] > 140) | (df['cholesterol'] > 240) | (df['diabetes'] == 1) | (df['smoker'] == 1), 'disease_risk'] = 1
    df.loc[(df['age'] > 70) & (df['blood_pressure'] > 150) & (df['cholesterol'] > 260), 'disease_risk'] = 1
    df.loc[(df['gender'] == 'Female') & (df['age'] < 40) & (df['smoker'] == 0), 'disease_risk'] = 0
    df['disease_risk'] = df['disease_risk'].apply(lambda x: 1 if np.random.rand() < 0.6 else 0 if x == 1 else (1 if np.random.rand() < 0.1 else 0))
    
    return df

def train_black_box_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    return model

def predict_proba(model, X_instance):
    if isinstance(X_instance, pd.Series):
        X_instance = X_instance.to_frame().T
    return model.predict_proba(X_instance)[:, 1]

def save_model(model, filename):
    joblib.dump(model, filename)

def load_model(filename):
    return joblib.load(filename)

class LACEExplainer:
    def __init__(self, black_box_model, data_preprocessor, feature_names):
        self.black_box_model = black_box_model
        self.data_preprocessor = data_preprocessor
        self.feature_names = feature_names

    def find_k_nearest_neighbors(self, instance, training_data, k=10):
        nn_model = NearestNeighbors(n_neighbors=k, algorithm='auto')
        nn_model.fit(training_data)
        distances, indices = nn_model.kneighbors(instance.values.reshape(1, -1))
        return training_data.iloc[indices[0]]

    def train_local_surrogate(self, local_neighbors_X, local_neighbors_y):
        surrogate_model = DecisionTreeClassifier(max_depth=3, random_state=42)
        surrogate_model.fit(local_neighbors_X, local_neighbors_y)
        return surrogate_model

    def _extract_rules(self, tree_model, feature_names):
        tree_rules = export_text(tree_model, feature_names=feature_names)
        return tree_rules

    def _calculate_prediction_difference(self, original_preprocessed_instance, original_prediction, X_train_preprocessed):
        contributions = {}
        # Iterate over original feature names (before one-hot encoding for clarity in explanation)
        original_cols = self.data_preprocessor.numerical_cols + self.data_preprocessor.categorical_cols

        for feature in original_cols:
            perturbed_instance = original_preprocessed_instance.copy()
            
            # Create a reference for marginalization (e.g., mean for numerical, mode for categorical)
            # For simplicity, we'll use the mean/mode from the *entire* training set for perturbation.
            # A more sophisticated LACE would use the local neighborhood's mean/mode.
            if feature in self.data_preprocessor.numerical_cols:
                mean_val = self.data_preprocessor.scaler.mean_[self.data_preprocessor.numerical_cols.index(feature)]
                # Need to map back to the scaled value
                scaled_mean_val = (mean_val - self.data_preprocessor.scaler.mean_[self.data_preprocessor.numerical_cols.index(feature)]) / self.data_preprocessor.scaler.scale_[self.data_preprocessor.numerical_cols.index(feature)]
                perturbed_instance[feature] = scaled_mean_val
            elif feature in self.data_preprocessor.categorical_cols:
                # Find the one-hot encoded columns corresponding to this categorical feature
                onehot_prefix = f"{feature}_"
                onehot_cols_for_feature = [col for col in self.feature_names if col.startswith(onehot_prefix)]
                
                # Set all one-hot columns for this feature to 0 in the perturbed instance
                for col in onehot_cols_for_feature:
                    if col in perturbed_instance.columns:
                        perturbed_instance[col] = 0.0
                
                # Find the mode of the original categorical feature in the training data
                # This requires access to the original training data or a stored mode
                # For simplification, we'll assume we can use the most frequent category's one-hot encoding
                # A more robust solution would require inverse transforming a slice of X_train and finding mode
                # Here, we'll use a simplified approach: just setting the feature to 'missing' or 'average effect'
                # For now, simply setting to 0 in one-hot acts as a form of marginalization.
                # A better approach would be to average predictions over all possible values of the categorical feature.
                # For demonstration, setting to 0 is a basic omission.
                pass # The loop above already sets all to 0
            
            perturbed_prediction = predict_proba(self.black_box_model, perturbed_instance)[0]
            contributions[feature] = original_prediction - perturbed_prediction

        return contributions

    def explain_instance(self, original_instance_df, X_train_original, k=10):
        # 1. Preprocess the instance and training data
        preprocessed_instance = self.data_preprocessor.transform(original_instance_df)
        X_train_preprocessed = self.data_preprocessor.transform(X_train_original)

        # 2. Get black-box prediction for the instance
        original_prediction = predict_proba(self.black_box_model, preprocessed_instance)[0]

        # 3. Find K-nearest neighbors in the preprocessed training data
        local_neighbors_preprocessed = self.find_k_nearest_neighbors(preprocessed_instance, X_train_preprocessed, k=k)
        
        # Get corresponding original training data instances for local_neighbors_preprocessed (for target labels)
        # Map back to original indices to get y_train_original
        local_neighbors_indices = local_neighbors_preprocessed.index
        local_neighbors_y = pd.Series([predict_proba(self.black_box_model, X_train_preprocessed.loc[[idx]])[0] for idx in local_neighbors_indices])
        
        # 4. Train a local surrogate model on the neighborhood
        surrogate_model = self.train_local_surrogate(local_neighbors_preprocessed, local_neighbors_y)

        # 5. Extract qualitative rules from the local surrogate model
        # Use the preprocessed feature names for the tree rules initially
        rules = self._extract_rules(surrogate_model, self.feature_names)

        # 6. Calculate quantitative prediction differences for individual features
        contributions = self._calculate_prediction_difference(preprocessed_instance, original_prediction, X_train_preprocessed)

        return {"prediction": original_prediction, "rules": rules, "contributions": contributions}

if __name__ == "__main__":
    # 1. Data Layer: Generate and Preprocess Data
    print("Generating synthetic data...")
    df_original = generate_synthetic_data(num_samples=1000)
    X = df_original.drop('disease_risk', axis=1)
    y = df_original['disease_risk']

    # Split data for training the black-box model
    X_train_original, X_test_original, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    data_preprocessor = DataPreprocessor()
    X_train_preprocessed = data_preprocessor.fit_transform(X_train_original)
    X_test_preprocessed = data_preprocessor.transform(X_test_original)

    # 2. Black-Box Model Layer: Train the Model
    print("Training black-box model (RandomForestClassifier)...")
    black_box_model = train_black_box_model(X_train_preprocessed, y_train)
    print(f"Black-box model accuracy on test set: {black_box_model.score(X_test_preprocessed, y_test):.4f}")

    # 3. LACE Explainer Layer: Explain an instance
    print("Initializing LACE Explainer...")
    lace_explainer = LACEExplainer(black_box_model, data_preprocessor, data_preprocessor.feature_names_out)

    # Select an instance to explain (e.g., the first instance from the test set)
    instance_to_explain_original = X_test_original.iloc[[0]]
    print(f"\nExplaining instance: {instance_to_explain_original.to_dict('records')[0]}")

    explanation = lace_explainer.explain_instance(instance_to_explain_original, X_train_original, k=15)

    print(f"\nBlack-box model predicted disease risk for this instance: {explanation['prediction']:.4f}")

    print("\n--- Local Rules (Qualitative Explanation) ---")
    print(explanation['rules'])

    print("\n--- Feature Contributions (Quantitative Explanation) ---")
    sorted_contributions = sorted(explanation['contributions'].items(), key=lambda item: abs(item[1]), reverse=True)
    for feature, contribution in sorted_contributions:
        print(f"  {feature}: {contribution:.4f}")

    # Demonstrate with another instance (e.g., one with higher risk)
    print("\n\n--- Explaining another instance with potentially higher risk ---")
    high_risk_instance_data = {
        'age': [75],
        'gender': ['Male'],
        'blood_pressure': [160],
        'cholesterol': [280],
        'diabetes': [1],
        'smoker': [1],
    }
    high_risk_instance_df = pd.DataFrame(high_risk_instance_data)
    print(f"\nExplaining instance: {high_risk_instance_df.to_dict('records')[0]}")

    explanation_high_risk = lace_explainer.explain_instance(high_risk_instance_df, X_train_original, k=15)
    print(f"\nBlack-box model predicted disease risk for this instance: {explanation_high_risk['prediction']:.4f}")

    print("\n--- Local Rules (Qualitative Explanation) ---")
    print(explanation_high_risk['rules'])

    print("\n--- Feature Contributions (Quantitative Explanation) ---")
    sorted_contributions_high_risk = sorted(explanation_high_risk['contributions'].items(), key=lambda item: abs(item[1]), reverse=True)
    for feature, contribution in sorted_contributions_high_risk:
        print(f"  {feature}: {contribution:.4f}")

