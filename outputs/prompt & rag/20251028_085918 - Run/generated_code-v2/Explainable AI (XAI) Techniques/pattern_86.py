import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

class DiabetesRiskModel:
    def __init__(self):
        self.model = None
        self.features = ['Age', 'BMI', 'BloodPressure', 'Glucose', 'Smoking', 'Exercise', 'FamilyHistory']

    def generate_synthetic_data(self, num_samples=1000):
        np.random.seed(42)
        data = {
            'Age': np.random.randint(20, 70, num_samples),
            'BMI': np.random.uniform(18.0, 35.0, num_samples),
            'BloodPressure': np.random.randint(90, 140, num_samples),
            'Glucose': np.random.randint(70, 200, num_samples),
            'Smoking': np.random.choice([0, 1], num_samples, p=[0.7, 0.3]),
            'Exercise': np.random.choice([0, 1], num_samples, p=[0.4, 0.6]),
            'FamilyHistory': np.random.choice([0, 1], num_samples, p=[0.6, 0.4])
        }
        df = pd.DataFrame(data)

        # Generate target variable (Diabetes) based on a simplified linear combination of features
        # and some randomness to simulate a real-world scenario
        df['Diabetes'] = (
            0.05 * df['Age']
            + 0.1 * df['BMI']
            + 0.02 * df['BloodPressure']
            + 0.05 * df['Glucose']
            + 1.5 * df['Smoking']
            - 0.8 * df['Exercise']
            + 1.0 * df['FamilyHistory']
            + np.random.normal(0, 1, num_samples)
        ) > 4.5 # Threshold for diabetes

        df['Diabetes'] = df['Diabetes'].astype(int)
        return df

    def preprocess_data(self, df):
        X = df[self.features]
        y = df['Diabetes']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train):
        self.model = LogisticRegression(solver='liblinear', random_state=42)
        self.model.fit(X_train, y_train)

    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        return accuracy, report

    def explain_prediction(self, individual_data):
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")

        feature_values = np.array([individual_data[f] for f in self.features]).reshape(1, -1)
        prediction_proba = self.model.predict_proba(feature_values)[:, 1][0]
        prediction_class = self.model.predict(feature_values)[0]

        explanation = f"Prediction: {'High Risk (1)' if prediction_class == 1 else 'Low Risk (0)'} of Diabetes (Probability: {prediction_proba:.2f}).\n"
        explanation += "Key contributing factors:\n"

        coefficients = dict(zip(self.features, self.model.coef_[0]))

        positive_contributors = []
        negative_contributors = []

        for feature, coef in coefficients.items():
            value = individual_data[feature]
            impact = coef * value
            if impact > 0.05: # Threshold for a noticeable positive impact
                positive_contributors.append(f"  - {feature} (Value: {value:.1f}, Positive Impact: {impact:.2f})")
            elif impact < -0.05: # Threshold for a noticeable negative impact
                negative_contributors.append(f"  - {feature} (Value: {value:.1f}, Negative Impact: {impact:.2f})")

        if positive_contributors:
            explanation += "Factors increasing risk:\n"
            explanation += "\n".join(positive_contributors)
        else:
            explanation += "No significant factors increasing risk identified.\n"

        if negative_contributors:
            explanation += "\nFactors decreasing risk:\n"
            explanation += "\n".join(negative_contributors)
        else:
            explanation += "No significant factors decreasing risk identified.\n"

        return prediction_class, explanation

    def get_model_coefficients(self):
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        return dict(zip(self.features, self.model.coef_[0]))

if __name__ == "__main__":
    diabetes_model = DiabetesRiskModel()

    # 1. Data Generation
    df = diabetes_model.generate_synthetic_data()
    print("Synthetic Data Head:\n", df.head())
    print("\nSynthetic Data Info:\n")
    df.info()

    # 2. Data Preprocessing
    X_train, X_test, y_train, y_test = diabetes_model.preprocess_data(df)
    print(f"\nTraining data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")

    # 3. Model Training
    diabetes_model.train_model(X_train, y_train)
    print("\nModel trained successfully.")

    # 4. Model Evaluation
    accuracy, report = diabetes_model.evaluate_model(X_test, y_test)
    print(f"\nModel Accuracy on Test Set: {accuracy:.2f}")
    print("\nClassification Report:\n", report)

    # 5. Interpretability & Prediction
    print("\nModel Coefficients (Impact on Diabetes Risk):\n")
    coefficients = diabetes_model.get_model_coefficients()
    for feature, coef in coefficients.items():
        print(f"  - {feature}: {coef:.4f}")

    # Example prediction for a new individual
    new_individual = {
        'Age': 55,
        'BMI': 32.5,
        'BloodPressure': 135,
        'Glucose': 180,
        'Smoking': 1,
        'Exercise': 0,
        'FamilyHistory': 1
    }
    print(f"\nPredicting for new individual: {new_individual}")
    prediction, explanation = diabetes_model.explain_prediction(new_individual)
    print(explanation)

    new_individual_2 = {
        'Age': 30,
        'BMI': 22.0,
        'BloodPressure': 110,
        'Glucose': 90,
        'Smoking': 0,
        'Exercise': 1,
        'FamilyHistory': 0
    }
    print(f"\nPredicting for new individual: {new_individual_2}")
    prediction_2, explanation_2 = diabetes_model.explain_prediction(new_individual_2)
    print(explanation_2)
