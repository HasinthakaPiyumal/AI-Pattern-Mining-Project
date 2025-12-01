import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

class DataPreparation:
    def generate_synthetic_data(self, num_samples=1000):
        np.random.seed(42)
        data = {
            'price': np.random.uniform(10, 500, num_samples),
            'brand_A': np.random.randint(0, 2, num_samples),
            'brand_B': np.random.randint(0, 2, num_samples),
            'category_electronics': np.random.randint(0, 2, num_samples),
            'category_clothing': np.random.randint(0, 2, num_samples),
            'customer_rating': np.random.uniform(1, 5, num_samples),
            'discount_level': np.random.uniform(0, 0.5, num_samples),
        }
        df = pd.DataFrame(data)

        df['purchased'] = (
            0.3 * df['price']
            + 0.2 * df['customer_rating']
            - 0.1 * df['discount_level']
            + 0.15 * df['brand_A']
            + 0.1 * df['category_electronics']
            + np.random.randn(num_samples) * 0.5
        ) > np.mean(df['price']) * 0.5 # Simple threshold for binary classification
        df['purchased'] = df['purchased'].astype(int)
        return df

class RecommendationModel:
    def train_model(self, X_train, y_train):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model

    def evaluate_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

    def save_model(self, model, filename):
        joblib.dump(model, filename)

    def load_model(self, filename):
        return joblib.load(filename)

class PermutationImportanceExplainer:
    def calculate_permutation_importance(self, model, X_test, y_test, features, metric_function):
        baseline_performance = metric_function(model, X_test, y_test)
        importances = {}

        for feature in features:
            X_test_permuted = X_test.copy()
            X_test_permuted[feature] = np.random.permutation(X_test_permuted[feature])
            permuted_performance = metric_function(model, X_test_permuted, y_test)
            importance = baseline_performance - permuted_performance
            importances[feature] = importance
        return importances

if __name__ == "__main__":
    data_prep = DataPreparation()
    df = data_prep.generate_synthetic_data(num_samples=2000)

    X = df.drop('purchased', axis=1)
    y = df['purchased']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rec_model = RecommendationModel()
    model = rec_model.train_model(X_train, y_train)

    model_filename = 'recommendation_model.joblib'
    rec_model.save_model(model, model_filename)
    loaded_model = rec_model.load_model(model_filename)

    baseline_accuracy = rec_model.evaluate_model(loaded_model, X_test, y_test)
    print(f"Baseline Model Accuracy: {baseline_accuracy:.4f}")

    explainer = PermutationImportanceExplainer()
    features_to_explain = X.columns.tolist()

    def model_accuracy_metric(model, X, y):
        return accuracy_score(y, model.predict(X))

    permutation_importances = explainer.calculate_permutation_importance(
        loaded_model, X_test, y_test, features_to_explain, model_accuracy_metric
    )

    print("\nPermutation Feature Importances (drop in accuracy):")
    sorted_importances = sorted(permutation_importances.items(), key=lambda item: item[1], reverse=True)
    for feature, importance in sorted_importances:
        print(f"{feature}: {importance:.4f}")