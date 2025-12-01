import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class BlackBoxDiagnosticModel:
    def __init__(self):
        self.model = RandomForestClassifier(random_state=42)
        self.scaler = StandardScaler()
        self.features = None
        self.is_fitted = False

    def fit(self, X, y, feature_names):
        self.features = feature_names
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted yet.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted yet.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def get_feature_importances(self):
        if self.is_fitted and self.model.feature_importances_ is not None:
            return dict(zip(self.features, self.model.feature_importances_))
        return {}


class LACEExplainer:
    def __init__(self, model):
        self.model = model

    def explain_instance(self, instance_df, top_features=3):
        feature_importances = self.model.get_feature_importances()
        if not feature_importances:
            return "No explanation available. Model not trained or feature importances not accessible."

        explanation_lines = []
        feature_impacts = {}
        
        for feature, importance in feature_importances.items():
            if feature in instance_df.columns:
                feature_value = instance_df[feature].iloc[0]
                feature_impacts[feature] = importance * feature_value

        sorted_impacts = sorted(feature_impacts.items(), key=lambda item: abs(item[1]), reverse=True)

        explanation_lines.append("Key factors influencing this prediction:")
        for feature, impact in sorted_impacts[:top_features]:
            value = instance_df[feature].iloc[0]
            sign = "positive" if impact >= 0 else "negative"
            explanation_lines.append(f"- **{feature}**: {value:.2f} (Estimated {sign} impact on prediction)")
        
        return "\n".join(explanation_lines)

    def compare_explanations(self, explanation1_text, explanation2_text):
        st.subheader("Explanation Comparison")
        st.markdown("---")
        st.write("### Explanation 1")
        st.write(explanation1_text)
        st.write("### Explanation 2")
        st.write(explanation2_text)
        st.markdown("---")
        st.write("Note: In a real system, a more sophisticated comparison would analyze rule sets, feature contributions, etc.")


def generate_synthetic_data(num_samples=1000, n_features=10):
    np.random.seed(42)
    
    features = [f"feature_{i+1}" for i in range(n_features)]
    X = pd.DataFrame(np.random.rand(num_samples, n_features) * 100, columns=features)
    
    y = ((X["feature_1"] * 0.5 + X["feature_2"] * 0.7 - X["feature_3"] * 0.3 + np.random.randn(num_samples) * 10) > 70).astype(int)
    
    return X, y, features

def preprocess_data(df):
    return df


st.set_page_config(layout="wide")
st.title("MediXplain: Interactive Diagnostic Explanation System")
st.markdown("An interactive human-in-the-loop framework for understanding and debugging black-box AI diagnostic models.")

if "model" not in st.session_state:
    st.session_state.model = BlackBoxDiagnosticModel()
    X_raw, y, feature_names = generate_synthetic_data()
    st.session_state.feature_names = feature_names
    X_train, X_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42)
    st.session_state.model.fit(X_train, y_train, feature_names)
    st.session_state.explainer = LACEExplainer(st.session_state.model)
    st.session_state.X_test = X_test
    st.session_state.y_test = y_test
    st.session_state.predictions_test = st.session_state.model.predict(X_test)

if "explanations" not in st.session_state:
    st.session_state.explanations = {}

model = st.session_state.model
explainer = st.session_state.explainer
feature_names = st.session_state.feature_names
X_test = st.session_state.X_test
y_test = st.session_state.y_test
predictions_test = st.session_state.predictions_test

st.sidebar.header("Instance Selection & Input")

instance_source = st.sidebar.radio("Select instance source:", ("From Test Data", "Manual Input"))

current_instance_df = None
instance_index = None

if instance_source == "From Test Data":
    st.sidebar.subheader("Select a Test Instance")
    instance_info = [f"Idx {i}: Pred={predictions_test[i]}, Actual={y_test.iloc[i]}" for i in range(len(X_test))]
    selected_instance_idx = st.sidebar.selectbox("Choose instance index:", list(range(len(X_test))), format_func=lambda x: instance_info[x])
    
    current_instance_df = X_test.iloc[[selected_instance_idx]]
    instance_index = selected_instance_idx
else:
    st.sidebar.subheader("Enter Patient Features Manually")
    manual_input_data = {}
    for feature in feature_names:
        default_value = float(np.random.rand() * 100)
        manual_input_data[feature] = st.sidebar.number_input(f"{feature}", value=default_value, key=f"manual_{feature}")
    current_instance_df = pd.DataFrame([manual_input_data])
    instance_index = "manual"

if current_instance_df is not None:
    st.subheader(f"Current Instance (ID: {instance_index})")
    st.dataframe(current_instance_df)

    if st.button("Get Prediction and Explanation"):
        try:
            prediction_proba = model.predict_proba(current_instance_df)[0]
            prediction_class = np.argmax(prediction_proba)
            
            st.success(f"Model Prediction: Class {prediction_class} (Probability: {prediction_proba[prediction_class]:.2f})")
            
            explanation = explainer.explain_instance(current_instance_df)
            st.session_state.explanations[instance_index] = explanation
            st.markdown("### Explanation (Simulated LACE)")
            st.markdown(explanation)
            
        except RuntimeError as e:
            st.error(f"Error: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

    st.markdown("---")
    st.subheader("What-If Analysis")
    st.write("Modify features below to see how the prediction and explanation change.")

    what_if_data = current_instance_df.copy()
    col1, col2 = st.columns(2)
    with col1:
        st.write("Original Values")
        st.dataframe(current_instance_df.T)
    with col2:
        st.write("What-If Values")
        modified_input = {}
        for feature in feature_names:
            modified_input[feature] = st.number_input(f"What-If {feature}", value=float(current_instance_df[feature].iloc[0]), key=f"what_if_{feature}")
        what_if_df = pd.DataFrame([modified_input])
        st.dataframe(what_if_df.T)

    if st.button("Run What-If Analysis"):
        try:
            what_if_proba = model.predict_proba(what_if_df)[0]
            what_if_class = np.argmax(what_if_proba)
            st.success(f"What-If Prediction: Class {what_if_class} (Probability: {what_if_proba[what_if_class]:.2f})")
            
            what_if_explanation = explainer.explain_instance(what_if_df)
            st.markdown("### What-If Explanation")
            st.markdown(what_if_explanation)
            
            if instance_index in st.session_state.explanations:
                original_explanation = st.session_state.explanations[instance_index]
                explainer.compare_explanations(original_explanation, what_if_explanation)
            
        except RuntimeError as e:
            st.error(f"Error: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

    st.markdown("---")
    st.subheader("Explanation Metadata (Global Insights - Placeholder)")
    st.write("This section would summarize multiple local explanations into 'explanation metadata' (attribute, item, local rule views) for global insights.")
    st.write("For instance, it could show common influential features across a set of misclassified cases or specific rules that frequently lead to a particular diagnosis.")
    st.write("*(Implementation for this would involve storing and analyzing many individual explanations.)*")