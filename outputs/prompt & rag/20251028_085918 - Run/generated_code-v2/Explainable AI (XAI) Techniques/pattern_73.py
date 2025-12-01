import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

def generate_synthetic_medical_data(n_samples=1000):
    np.random.seed(42)
    data = {
        'age': np.random.randint(20, 80, n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'symptom_fever': np.random.randint(0, 2, n_samples),
        'symptom_cough': np.random.randint(0, 2, n_samples),
        'symptom_fatigue': np.random.randint(0, 2, n_samples),
        'lab_result_wbc': np.random.normal(7, 2, n_samples),
        'lab_result_crp': np.random.normal(10, 5, n_samples),
        'risk_factor_smoker': np.random.randint(0, 2, n_samples)
    }
    df = pd.DataFrame(data)

    # Simulate a 'diagnosis' based on some features
    df['diagnosis'] = 0
    df.loc[(df['age'] > 50) & (df['symptom_fever'] == 1) & (df['lab_result_wbc'] > 9), 'diagnosis'] = 1
    df.loc[(df['symptom_cough'] == 1) & (df['risk_factor_smoker'] == 1) & (df['age'] > 40), 'diagnosis'] = 1
    df.loc[(df['lab_result_crp'] > 15) & (df['symptom_fatigue'] == 1), 'diagnosis'] = 1
    df.loc[np.random.rand(n_samples) < 0.05, 'diagnosis'] = 1 # Some random cases
    
    # Make sure there's some balance
    df.loc[df['diagnosis'] == 0, 'diagnosis'] = np.random.choice([0, 1], sum(df['diagnosis'] == 0), p=[0.95, 0.05])

    return df

class BlackBoxDiagnosisModel:
    def __init__(self, random_state=42):
        self.model = RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight='balanced')
        self.preprocessor = None
        self.feature_names = None

    def train(self, X_train, y_train):
        numerical_features = X_train.select_dtypes(include=np.number).columns
        categorical_features = X_train.select_dtypes(include='object').columns

        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        X_train_processed = self.preprocessor.fit_transform(X_train)
        
        # Get feature names after one-hot encoding
        ohe_feature_names = self.preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
        self.feature_names = list(numerical_features) + list(ohe_feature_names)
        
        self.model.fit(X_train_processed, y_train)

    def predict_proba(self, X):
        X_processed = self.preprocessor.transform(X)
        return self.model.predict_proba(X_processed)

class LACEExplainer:
    def __init__(self, black_box_model, preprocessor, feature_names, categorical_features):
        self.black_box_model = black_box_model
        self.preprocessor = preprocessor
        self.feature_names = feature_names
        self.categorical_features = categorical_features

    def _find_k_nearest_neighbors(self, instance, training_data, k=10):
        # Use preprocessed training data for KNN
        nn = NearestNeighbors(n_neighbors=k, metric='euclidean')
        nn.fit(self.preprocessor.transform(training_data))
        
        distances, indices = nn.kneighbors(self.preprocessor.transform(pd.DataFrame([instance])), k)
        return training_data.iloc[indices[0]]

    def _train_local_surrogate(self, local_data, black_box_predictions):
        local_data_processed = self.preprocessor.transform(local_data)
        surrogate_model = DecisionTreeClassifier(max_depth=3, random_state=42)
        surrogate_model.fit(local_data_processed, black_box_predictions)
        return surrogate_model

    def _get_rules_from_tree(self, tree_model, feature_names):
        tree_rules = export_text(tree_model, feature_names=feature_names)
        rules = []
        for line in tree_rules.split('\n'):
            if '|---' in line and not 'class:' in line:
                rule_condition = line.strip().replace('|--- ', '').replace(' ', '')
                rules.append(rule_condition)
        return rules

    def _calculate_prediction_difference(self, instance, black_box_prediction, local_data, surrogate_model, target_class=1):
        contributions = {}
        original_proba = black_box_prediction[target_class]

        # Individual attribute contributions
        for feature in self.feature_names:
            temp_instance = instance.copy()
            if feature in self.categorical_features:
                # Replace with most frequent category in local data
                mode_val = local_data[feature.split('_')[0]].mode()[0] if '_' in feature else local_data[feature].mode()[0]
                if '_' in feature:
                     # Need to map back to original categorical feature for perturbation
                    original_cat_feature = feature.split('_')[0]
                    temp_instance[original_cat_feature] = mode_val
                else:
                    temp_instance[feature] = mode_val
            else:
                # Replace with mean of local data
                temp_instance[feature] = local_data[feature].mean()
            
            perturbed_proba = self.black_box_model.predict_proba(pd.DataFrame([temp_instance]))[0][target_class]
            contributions[f'Feature: {feature}'] = original_proba - perturbed_proba

        # Pattern (rule) contributions from surrogate model
        rules = self._get_rules_from_tree(surrogate_model, self.black_box_model.feature_names)
        for i, rule in enumerate(rules):
            # This is a simplified approximation for rule contribution
            # A proper LACE rule contribution involves more complex marginalization
            # For demonstration, we'll check if the instance satisfies the rule
            # and consider the impact if it didn't
            
            # For simplicity, we just mark rules as present/absent for the instance.
            # Actual LACE calculates prediction difference by 'omitting' the rule.
            # Here, we'll just show the rules that apply to the instance.
            # The impact calculation is more involved and depends on the base prediction vs. rule-applied prediction.

            # To approximate, let's create a counterfactual where this rule is 'broken'
            # This is highly dependent on the rule structure and complex to generalize.
            # For now, we will simply list the rules and not attempt a quantitative 'prediction difference' for each rule directly
            # as it requires sophisticated parsing and counterfactual generation for each rule condition.
            pass # Skipping direct quantitative rule contribution for simplicity of single file example

        return contributions, rules

    def explain_instance(self, instance, training_data, k=10, target_class=1):
        local_neighbors = self._find_k_nearest_neighbors(instance, training_data, k)
        
        # Predict probabilities for the local neighbors using the black-box model
        neighbor_predictions = self.black_box_model.predict_proba(local_neighbors)[:, target_class]

        surrogate_model = self._train_local_surrogate(local_neighbors, neighbor_predictions)

        instance_prediction = self.black_box_model.predict_proba(pd.DataFrame([instance]))[0]

        contributions, rules = self._calculate_prediction_difference(instance, instance_prediction, local_neighbors, surrogate_model, target_class)
        
        return {
            'instance': instance,
            'black_box_prediction': instance_prediction,
            'feature_contributions': contributions,
            'local_rules': rules,
            'local_model_tree_structure': export_text(surrogate_model, feature_names=self.black_box_model.feature_names)
        }

    def plot_contributions(self, explanation_results, top_n=10):
        contributions = explanation_results['feature_contributions']
        sorted_contributions = sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)[:top_n]
        features = [item[0] for item in sorted_contributions]
        values = [item[1] for item in sorted_contributions]

        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = np.arange(len(features))
        ax.barh(y_pos, values, align='center', color=['green' if v > 0 else 'red' for v in values])
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel('Prediction Difference (Change in P(Diagnosis=1))')
        ax.set_title('LACE Feature Contributions for Diagnosis Explanation')
        plt.show()

if __name__ == '__main__':
    # 1. Generate Synthetic Data
    df = generate_synthetic_medical_data(n_samples=2000)
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Identify feature types
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include='object').columns.tolist()

    # 2. Train Black-Box Model
    black_box_model = BlackBoxDiagnosisModel()
    black_box_model.train(X_train, y_train)
    print(f"Black-box model accuracy on test set: {black_box_model.model.score(black_box_model.preprocessor.transform(X_test), y_test):.4f}")

    # 3. Instantiate LACE Explainer
    lace_explainer = LACEExplainer(
        black_box_model=black_box_model,
        preprocessor=black_box_model.preprocessor, 
        feature_names=black_box_model.feature_names,
        categorical_features=categorical_features # Pass original categorical features for perturbation logic
    )

    # 4. Select an instance to explain (e.g., the first instance from the test set)
    instance_to_explain = X_test.iloc[0]
    print(f"\nInstance to explain:\n{instance_to_explain}")

    # 5. Get Explanation
    explanation = lace_explainer.explain_instance(instance_to_explain, X_train, k=20, target_class=1)

    print(f"\nBlack-box model predicted probability for Diagnosis=1: {explanation['black_box_prediction'][1]:.4f}")
    print("\n--- Feature Contributions (Prediction Difference) ---")
    for feature, contribution in explanation['feature_contributions'].items():
        print(f"{feature}: {contribution:.4f}")

    print("\n--- Local Rules (from Surrogate Model) ---")
    for rule in explanation['local_rules']:
        print(rule)
        
    print("\n--- Local Model Decision Tree Structure ---")
    print(explanation['local_model_tree_structure'])

    # 6. Visualize Contributions
    lace_explainer.plot_contributions(explanation)
