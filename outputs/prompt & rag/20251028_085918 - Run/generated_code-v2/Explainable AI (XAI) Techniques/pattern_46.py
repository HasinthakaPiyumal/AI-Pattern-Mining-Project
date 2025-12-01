import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score

# 1. Data Preprocessing Layer
class DataPreprocessor:
    def __init__(self, numerical_cols, categorical_cols):
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.feature_names_out = None

    def fit(self, X):
        if self.numerical_cols:
            self.scaler.fit(X[self.numerical_cols])
        if self.categorical_cols:
            self.encoder.fit(X[self.categorical_cols])
        self.feature_names_out = self._get_feature_names(X)

    def transform(self, X):
        X_processed = X.copy()
        if self.numerical_cols:
            X_processed[self.numerical_cols] = self.scaler.transform(X[self.numerical_cols])
        if self.categorical_cols:
            encoded_features = self.encoder.transform(X[self.categorical_cols])
            encoded_df = pd.DataFrame(encoded_features, columns=self.encoder.get_feature_names_out(self.categorical_cols), index=X.index)
            X_processed = X_processed.drop(columns=self.categorical_cols)
            X_processed = pd.concat([X_processed, encoded_df], axis=1)
        return X_processed

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def _get_feature_names(self, X):
        num_features = self.numerical_cols if self.numerical_cols else []
        cat_features = list(self.encoder.get_feature_names_out(self.categorical_cols)) if self.categorical_cols else []
        return num_features + cat_features

# 2. Black-box Predictive Model Layer (Placeholder)
class BlackBoxModel:
    def __init__(self):
        self.model = RandomForestClassifier(random_state=42)
        self.is_fitted = False

    def fit(self, X, y):
        self.model.fit(X, y)
        self.is_fitted = True

    def predict_proba(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model.predict_proba(X)

# 3. LACE Explainer Core Layer
class LACEExplainer:
    def __init__(self, black_box_model, preprocessor, X_train_original, X_train_preprocessed, feature_names):
        self.black_box_model = black_box_model
        self.preprocessor = preprocessor
        self.X_train_original = X_train_original
        self.X_train_preprocessed = X_train_preprocessed
        self.feature_names = feature_names
        self.knn = NearestNeighbors(n_neighbors=5, algorithm='auto')
        self.knn.fit(X_train_preprocessed)

    def _generate_local_neighborhood(self, instance_preprocessed, k):
        distances, indices = self.knn.kneighbors(instance_preprocessed.reshape(1, -1), n_neighbors=k)
        local_data_indices = indices[0]
        return self.X_train_original.iloc[local_data_indices], self.X_train_preprocessed.iloc[local_data_indices]

    def _train_local_surrogate(self, local_X_preprocessed, black_box_predictions, target_class_idx):
        # For LACE, the local surrogate aims to extract rules, not just predict probabilities
        # A shallow Decision Tree can approximate this for explanation
        surrogate_model = DecisionTreeClassifier(max_depth=3, random_state=42)
        surrogate_model.fit(local_X_preprocessed, black_box_predictions[:, target_class_idx] > 0.5)
        return surrogate_model

    def _calculate_prediction_difference(self, instance_original, instance_preprocessed, target_class_idx, k_neighbors):
        contributions = {}
        original_prediction_proba = self.black_box_model.predict_proba(instance_preprocessed.reshape(1, -1))[0, target_class_idx]

        # Get local neighborhood for marginalization context
        local_X_original, _ = self._generate_local_neighborhood(instance_preprocessed, k_neighbors)

        for i, feature_name in enumerate(self.feature_names):
            temp_instance_original = instance_original.copy()

            # For numerical features, replace with mean of local neighborhood
            if feature_name in self.preprocessor.numerical_cols:
                mean_val = local_X_original[feature_name].mean()
                temp_instance_original[feature_name] = mean_val
            # For categorical features, replace with mode of local neighborhood
            elif any(cat_col_prefix in feature_name for cat_col_prefix in self.preprocessor.categorical_cols):
                original_cat_col = None
                for cat_col in self.preprocessor.categorical_cols:
                    if cat_col in feature_name:
                        original_cat_col = cat_col
                        break
                if original_cat_col:
                    mode_val = local_X_original[original_cat_col].mode()[0]
                    temp_instance_original[original_cat_col] = mode_val
                else:
                    # Fallback for features that might not directly map if one-hot encoded
                    continue
            else: # If a feature name doesn't directly map, skip or handle as needed
                continue

            # Preprocess the perturbed instance
            perturbed_instance_preprocessed = self.preprocessor.transform(pd.DataFrame([temp_instance_original], columns=instance_original.index))
            # Ensure columns match training data, add missing and reorder
            perturbed_instance_preprocessed = perturbed_instance_preprocessed.reindex(columns=self.feature_names, fill_value=0)

            # Get prediction for perturbed instance
            perturbed_prediction_proba = self.black_box_model.predict_proba(perturbed_instance_preprocessed)[0, target_class_idx]

            # Contribution is the difference in probability upon omission (marginalization)
            # LACE uses 'prediction difference' by approximating marginalization.
            # We calculate: Original P - P_with_feature_omitted_or_perturbed
            contributions[feature_name] = original_prediction_proba - perturbed_prediction_proba

        return contributions

    def explain_instance(self, instance_original, k=10):
        # 1. Preprocess the instance
        instance_preprocessed = self.preprocessor.transform(pd.DataFrame([instance_original], columns=instance_original.index))
        instance_preprocessed = instance_preprocessed.reindex(columns=self.feature_names, fill_value=0).iloc[0]

        # 2. Get black-box model prediction for the instance
        bb_prediction_proba = self.black_box_model.predict_proba(instance_preprocessed.reshape(1, -1))[0]
        predicted_class_idx = np.argmax(bb_prediction_proba)
        predicted_class_proba = bb_prediction_proba[predicted_class_idx]

        # 3. Generate local neighborhood
        local_X_original, local_X_preprocessed = self._generate_local_neighborhood(instance_preprocessed, k)

        # 4. Train local surrogate model and get rules
        # Get black-box predictions for the local neighborhood
        local_bb_predictions = self.black_box_model.predict_proba(local_X_preprocessed)
        local_surrogate_model = self._train_local_surrogate(local_X_preprocessed, local_bb_predictions, predicted_class_idx)
        surrogate_rules = export_text(local_surrogate_model, feature_names=self.feature_names)

        # 5. Calculate prediction differences (contributions)
        feature_contributions = self._calculate_prediction_difference(instance_original, instance_preprocessed, predicted_class_idx, k)

        explanation = {
            "predicted_class_idx": predicted_class_idx,
            "predicted_class_proba": predicted_class_proba,
            "feature_contributions": feature_contributions,
            "qualitative_rules": surrogate_rules,
        }
        return explanation

# 4. Explanation Visualization Layer
def visualize_explanation(contributions, title="Feature Contributions"):
    if not contributions:
        st.write("No contributions to visualize.")
        return
    df_contrib = pd.DataFrame(list(contributions.items()), columns=['Feature', 'Contribution'])
    df_contrib = df_contrib.sort_values(by='Contribution', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Contribution', y='Feature', data=df_contrib, palette='viridis', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Prediction Difference (Contribution to Predicted Class Probability)')
    ax.set_ylabel('Feature')
    st.pyplot(fig)

# 5. API/User Interface Layer (Streamlit App)

def main():
    st.title("Healthcare Predictive Model Explainer (LACE)")
    st.write("Upload patient data (CSV) or use a sample to get interpretable explanations for disease risk predictions.")

    # Sample Data Generation (for demonstration)
    def generate_sample_data(num_samples=100):
        data = {
            'Age': np.random.randint(20, 80, num_samples),
            'BMI': np.random.uniform(18, 40, num_samples),
            'Glucose': np.random.uniform(70, 200, num_samples),
            'BloodPressure': np.random.uniform(80, 180, num_samples),
            'Insulin': np.random.uniform(10, 300, num_samples),
            'FamilyHistory': np.random.choice(['Yes', 'No'], num_samples),
            'Smoking': np.random.choice(['Yes', 'No'], num_samples),
            'PhysicalActivity': np.random.uniform(0, 7, num_samples), # days per week
            'Diabetes': np.random.randint(0, 2, num_samples) # Target variable
        }
        df = pd.DataFrame(data)
        # Make diabetes prediction somewhat correlated
        df.loc[((df['BMI'] > 30) | (df['Glucose'] > 140)) & (df['Age'] > 45), 'Diabetes'] = np.random.choice([0, 1], p=[0.2, 0.8], size=len(df[((df['BMI'] > 30) | (df['Glucose'] > 140)) & (df['Age'] > 45)]))
        return df

    # Initialize data and models
    if 'df_data' not in st.session_state:
        st.session_state.df_data = generate_sample_data(200)
        st.session_state.target_column = 'Diabetes'

        numerical_cols = ['Age', 'BMI', 'Glucose', 'BloodPressure', 'Insulin', 'PhysicalActivity']
        categorical_cols = ['FamilyHistory', 'Smoking']

        X = st.session_state.df_data.drop(columns=[st.session_state.target_column])
        y = st.session_state.df_data[st.session_state.target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        preprocessor = DataPreprocessor(numerical_cols, categorical_cols)
        X_train_preprocessed = preprocessor.fit_transform(X_train)
        # X_test_preprocessed = preprocessor.transform(X_test) # Not needed for LACE directly

        black_box_model = BlackBoxModel()
        black_box_model.fit(X_train_preprocessed, y_train)

        feature_names = preprocessor.feature_names_out

        st.session_state.preprocessor = preprocessor
        st.session_state.black_box_model = black_box_model
        st.session_state.X_train_original = X_train
        st.session_state.X_train_preprocessed = X_train_preprocessed
        st.session_state.feature_names = feature_names

        st.session_state.lace_explainer = LACEExplainer(
            st.session_state.black_box_model,
            st.session_state.preprocessor,
            st.session_state.X_train_original,
            st.session_state.X_train_preprocessed,
            st.session_state.feature_names
        )

    st.sidebar.header("Patient Data Input")
    with st.sidebar.form("patient_input_form"):
        st.write("Enter patient details:")
        age = st.number_input('Age', min_value=1, max_value=120, value=45)
        bmi = st.number_input('BMI', min_value=10.0, max_value=60.0, value=28.5)
        glucose = st.number_input('Glucose', min_value=50.0, max_value=300.0, value=120.0)
        blood_pressure = st.number_input('Blood Pressure', min_value=60.0, max_value=200.0, value=120.0)
        insulin = st.number_input('Insulin', min_value=10.0, max_value=500.0, value=100.0)
        family_history = st.selectbox('Family History of Diabetes', ['Yes', 'No'])
        smoking = st.selectbox('Smoking Habit', ['Yes', 'No'])
        physical_activity = st.number_input('Physical Activity (days/week)', min_value=0.0, max_value=7.0, value=3.0)

        submitted = st.form_submit_button("Get Explanation")

    if submitted:
        input_data = pd.Series({
            'Age': age,
            'BMI': bmi,
            'Glucose': glucose,
            'BloodPressure': blood_pressure,
            'Insulin': insulin,
            'FamilyHistory': family_history,
            'Smoking': smoking,
            'PhysicalActivity': physical_activity
        })

        st.subheader("Black-box Model Prediction")
        # Preprocess input data for prediction
        instance_preprocessed_for_prediction = st.session_state.preprocessor.transform(pd.DataFrame([input_data], columns=input_data.index))
        instance_preprocessed_for_prediction = instance_preprocessed_for_prediction.reindex(columns=st.session_state.feature_names, fill_value=0)

        prediction_proba = st.session_state.black_box_model.predict_proba(instance_preprocessed_for_prediction)[0]
        predicted_class = np.argmax(prediction_proba)
        st.write(f"Predicted Class (0=No Diabetes, 1=Diabetes): **{predicted_class}**")
        st.write(f"Probability of Class 0: {prediction_proba[0]:.2f}")
        st.write(f"Probability of Class 1: {prediction_proba[1]:.2f}")

        st.subheader("LACE Explanation")
        explanation = st.session_state.lace_explainer.explain_instance(input_data)

        st.write(f"Explanation for predicting Class **{explanation['predicted_class_idx']}** with probability **{explanation['predicted_class_proba']:.2f}**:")

        st.markdown("### Qualitative Rules (Local Surrogate Model)")
        st.text(explanation['qualitative_rules'])

        st.markdown("### Quantitative Feature Contributions (Prediction Difference)")
        visualize_explanation(explanation['feature_contributions'])

if __name__ == "__main__":
    main()