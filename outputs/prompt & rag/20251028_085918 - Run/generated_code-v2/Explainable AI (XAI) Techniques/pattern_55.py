import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score

from fastapi import FastAPI
from pydantic import BaseModel


# --- 1. Data Preparation and Preprocessing ---
class DataPreprocessor:
    def __init__(self):
        self.preprocessor = None
        self.feature_names_out = None

    def fit(self, X):
        categorical_features = X.select_dtypes(include=['object', 'category']).columns
        numerical_features = X.select_dtypes(include=np.number).columns

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])
        self.preprocessor.fit(X)

        # Get feature names after preprocessing
        num_features_out = numerical_features.tolist()
        cat_features_out = self.preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features).tolist()
        self.feature_names_out = num_features_out + cat_features_out
        return self

    def transform(self, X):
        return self.preprocessor.transform(X)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def get_feature_names_out(self):
        return self.feature_names_out


# --- 2. Black-Box Credit Risk Model (Pre-trained/Mock) ---
class BlackBoxCreditModel:
    def __init__(self, preprocessor, model=None):
        self.preprocessor = preprocessor
        self.model = model if model else RandomForestClassifier(random_state=42)

    def train(self, X_raw, y):
        X_processed = self.preprocessor.fit_transform(X_raw)
        self.model.fit(X_processed, y)

    def predict_proba(self, X_raw):
        X_processed = self.preprocessor.transform(X_raw)
        # Assuming binary classification, predict probability of class 1 (denial for this example)
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_processed)[:, 1]
        else:
            # For models without predict_proba, return raw predictions and adjust
            return self.model.predict(X_processed)


# --- 3. LACE Explanation Module ---
class LACEExplainer:
    def __init__(self, black_box_model, preprocessor, training_data_raw, training_data_processed, feature_names):
        self.black_box_model = black_box_model
        self.preprocessor = preprocessor
        self.training_data_raw = training_data_raw
        self.training_data_processed = training_data_processed
        self.feature_names = feature_names

    def _find_local_data(self, instance_processed, K=None):
        # Simple K-tuning heuristic: sqrt of dataset size, capped
        if K is None:
            K = min(int(np.sqrt(len(self.training_data_processed))), 100)
            if K < 10 and len(self.training_data_processed) >= 10: K = 10 # Ensure minimum neighbors
            elif K < 5 and len(self.training_data_processed) < 10: K = len(self.training_data_processed) - 1 if len(self.training_data_processed) > 1 else 1

        nn = NearestNeighbors(n_neighbors=K + 1, algorithm='auto').fit(self.training_data_processed)
        distances, indices = nn.kneighbors(instance_processed.reshape(1, -1))

        # Exclude the instance itself if it's in training data
        local_indices = [idx for idx in indices.flatten() if not np.array_equal(self.training_data_processed[idx], instance_processed)]
        if not local_indices and len(self.training_data_processed) > 0: # Fallback if instance not distinct or K=0
            local_indices = indices.flatten()[1:K+1] if len(indices.flatten()) > 1 else [0]
        elif not local_indices: # if no training data or K=0 and instance is unique
            return pd.DataFrame(), pd.DataFrame(), np.array([])

        local_data_raw = self.training_data_raw.iloc[local_indices]
        local_data_processed = self.training_data_processed[local_indices]
        return local_data_raw, local_data_processed

    def _train_surrogate(self, local_data_processed, black_box_predictions):
        # Use a shallow Decision Tree as a proxy for L3 rules
        surrogate_model = DecisionTreeClassifier(max_depth=3, random_state=42)
        surrogate_model.fit(local_data_processed, black_box_predictions > 0.5) # Assuming 0.5 is denial threshold
        return surrogate_model

    def _calculate_prediction_difference_individual(self, instance_raw, original_prediction_proba, local_data_raw):
        contributions = {}
        base_instance_df = pd.DataFrame([instance_raw])

        for feature in instance_raw.index:
            temp_instance_raw = base_instance_df.copy()
            if not local_data_raw.empty:
                # Approximate omission by replacing with feature's marginal mean/mode from local data
                if pd.api.types.is_numeric_dtype(local_data_raw[feature]):
                    temp_instance_raw[feature] = local_data_raw[feature].mean()
                else:
                    temp_instance_raw[feature] = local_data_raw[feature].mode()[0] if not local_data_raw[feature].mode().empty else temp_instance_raw[feature][0]
            else:
                # Fallback if no local data: replace with feature's global mean/mode or a default
                if pd.api.types.is_numeric_dtype(base_instance_df[feature]):
                    temp_instance_raw[feature] = self.training_data_raw[feature].mean()
                else:
                    temp_instance_raw[feature] = self.training_data_raw[feature].mode()[0] if not self.training_data_raw[feature].mode().empty else temp_instance_raw[feature][0]

            perturbed_prediction_proba = self.black_box_model.predict_proba(temp_instance_raw)[0]
            contributions[feature] = original_prediction_proba - perturbed_prediction_proba
        return contributions

    def _extract_and_quantify_rules(self, instance_raw, original_prediction_proba, surrogate_model, local_data_raw):
        rules_contributions = {}
        tree_rules = export_text(surrogate_model, feature_names=self.feature_names)
        # This is a highly simplified approach to rule quantification.
        # A full LACE implementation would involve more sophisticated rule extraction
        # and perturbation strategies (e.g., counterfactuals or explicit marginalization).

        # For simplicity, we'll try to identify simple path rules and estimate their effect.
        # This is not a direct LACE rule quantification but an illustrative approximation.
        # Real LACE would involve carefully constructing instances where rules are active/inactive.

        # Identify the leaf node for the instance in the surrogate tree
        instance_processed = self.preprocessor.transform(pd.DataFrame([instance_raw]))
        leaf_id = surrogate_model.apply(instance_processed)[0]
        node_indicator = surrogate_model.decision_path(instance_processed)
        node_indices = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[0+1]]

        path_features = set()
        path_rules = []

        for i in node_indices[:-1]: # Exclude leaf node itself
            feature_idx = surrogate_model.tree_.feature[i]
            if feature_idx != -2: # Not a leaf node
                feature_name = self.feature_names[feature_idx]
                threshold = surrogate_model.tree_.threshold[i]
                value = instance_raw[feature_name]

                if value <= threshold:
                    rule = f"{feature_name} <= {threshold:.2f}"
                else:
                    rule = f"{feature_name} > {threshold:.2f}"
                path_rules.append(rule)
                path_features.add(feature_name)

        if path_rules:
            combined_rule_str = " AND ".join(path_rules)
            # To quantify: Create a hypothetical instance where this rule is 'broken'
            # This is a very rough approximation and not true marginalization as in LACE
            perturbed_instance_raw = pd.DataFrame([instance_raw.copy()])
            # For each feature in the path, try to change it to 'break' the rule
            # This needs to be smarter - e.g., if credit_score > 700, change to 600
            # This is hard to generalize without specific rule parsing.

            # Instead, for simplicity, we'll just use the prediction of the surrogate model
            # for the instance as a proxy for the rule's effect relative to average.
            # This is NOT the prediction difference as described in LACE for interactions.
            rules_contributions[combined_rule_str] = surrogate_model.predict_proba(instance_processed)[0, 1] - np.mean(surrogate_model.predict_proba(local_data_processed)[:, 1])

        return rules_contributions


    def explain_instance(self, instance_raw_df, K=None):
        instance_raw = instance_raw_df.iloc[0]
        instance_processed = self.preprocessor.transform(instance_raw_df).flatten()

        original_prediction_proba = self.black_box_model.predict_proba(instance_raw_df)[0]

        # Only explain if denied (probability > 0.5 for denial class)
        if original_prediction_proba <= 0.5: # Assuming denial is class 1 and prob > 0.5
            return {
                "prediction": float(original_prediction_proba),
                "explanation_status": "Not denied, no explanation generated."
            }

        local_data_raw, local_data_processed = self._find_local_data(instance_processed, K=K)

        if local_data_raw.empty or len(local_data_processed) < 2: # Need at least 2 neighbors for a meaningful surrogate
             return {"prediction": float(original_prediction_proba),
                     "explanation_status": "Not enough local data to generate explanation.",
                     "contributions_individual": {}, "contributions_rules": {},
                     "visualization": None}

        local_black_box_predictions = self.black_box_model.predict_proba(local_data_raw)[:, 1]
        surrogate_model = self._train_surrogate(local_data_processed, local_black_box_predictions)

        individual_contributions = self._calculate_prediction_difference_individual(instance_raw, original_prediction_proba, local_data_raw)
        # Simplified interaction rule contributions - see notes in method
        rule_contributions = self._extract_and_quantify_rules(instance_raw, original_prediction_proba, surrogate_model, local_data_raw)

        # Generate visualization
        plot_base64 = self._visualize_contributions(individual_contributions, rule_contributions)

        return {
            "prediction": float(original_prediction_proba),
            "explanation_status": "Explanation generated.",
            "contributions_individual": {str(k): float(v) for k, v in individual_contributions.items()},
            "contributions_rules": {str(k): float(v) for k, v in rule_contributions.items()},
            "visualization": plot_base64
        }

    def _visualize_contributions(self, individual_contributions, rule_contributions):
        all_contributions = {**individual_contributions, **rule_contributions}
        if not all_contributions:
            return None

        sorted_contributions = sorted(all_contributions.items(), key=lambda item: abs(item[1]), reverse=True)
        labels = [item[0] for item in sorted_contributions[:10]] # Top 10
        values = [item[1] for item in sorted_contributions[:10]]

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['skyblue' if v >= 0 else 'salmon' for v in values]
        ax.barh(labels, values, color=colors)
        ax.set_xlabel('Prediction Difference (Impact on Denial Probability)')
        ax.set_title('LACE Explanation: Feature/Rule Contributions to Loan Denial')
        ax.invert_yaxis() # Highest contribution on top
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode('utf-8')


# --- FastAPI Application --- 
app = FastAPI()

# Mock Data Generation
def generate_mock_data(num_samples=1000):
    np.random.seed(42)
    data = {
        'credit_score': np.random.randint(300, 850, num_samples),
        'debt_to_income_ratio': np.random.uniform(0.1, 0.6, num_samples),
        'loan_amount': np.random.randint(5000, 100000, num_samples),
        'employment_length_years': np.random.randint(0, 20, num_samples),
        'has_mortgage': np.random.choice([0, 1], num_samples, p=[0.7, 0.3]),
        'income': np.random.randint(30000, 200000, num_samples),
        'education': np.random.choice(['High School', 'Bachelors', 'Masters', 'PhD'], num_samples, p=[0.25, 0.4, 0.2, 0.15]),
        'marital_status': np.random.choice(['Single', 'Married', 'Divorced'], num_samples, p=[0.4, 0.5, 0.1])
    }
    df = pd.DataFrame(data)

    # Simple rule for loan denial (target variable)
    # Higher denial probability for low credit score, high DTI, high loan amount, short employment
    y = ((df['credit_score'] < 600) * 0.4 +
         (df['debt_to_income_ratio'] > 0.4) * 0.3 +
         (df['loan_amount'] > 50000) * 0.2 +
         (df['employment_length_years'] < 2) * 0.1 +
         np.random.rand(num_samples) * 0.2 # noise
        ) > 0.5
    y = y.astype(int) # 1 for denial, 0 for approval

    return df, y

X_train_raw, y_train = generate_mock_data(num_samples=1000)

# Initialize and train components
data_preprocessor = DataPreprocessor()
X_train_processed = data_preprocessor.fit_transform(X_train_raw)
processed_feature_names = data_preprocessor.get_feature_names_out()

black_box_model = BlackBoxCreditModel(data_preprocessor)
black_box_model.train(X_train_raw, y_train)

lace_explainer = LACEExplainer(
    black_box_model=black_box_model,
    preprocessor=data_preprocessor,
    training_data_raw=X_train_raw,
    training_data_processed=X_train_processed,
    feature_names=processed_feature_names
)


class LoanApplication(BaseModel):
    credit_score: int
    debt_to_income_ratio: float
    loan_amount: int
    employment_length_years: int
    has_mortgage: int # 0 or 1
    income: int
    education: str # High School, Bachelors, Masters, PhD
    marital_status: str # Single, Married, Divorced


@app.post("/explain_loan_denial")
async def explain_loan_denial(application: LoanApplication):
    instance_raw_df = pd.DataFrame([application.dict()])
    explanation_result = lace_explainer.explain_instance(instance_raw_df)
    return explanation_result

# To run this application:
# 1. Save the code as `credit_risk_explanation_system.py`
# 2. Install necessary libraries: `pip install pandas numpy scikit-learn matplotlib fastapi uvicorn pydantic`
# 3. Run from your terminal: `uvicorn credit_risk_explanation_system:app --reload`
# 4. Access the API at `http://127.0.0.1:8000/docs` for the Swagger UI.

