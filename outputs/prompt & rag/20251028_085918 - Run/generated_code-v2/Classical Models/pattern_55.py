import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

# 1. Data Ingestion and Preprocessing Module
def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    age = np.random.randint(29, 77, n_samples)
    sex = np.random.randint(0, 2, n_samples)
    cp = np.random.randint(0, 4, n_samples)
    trestbps = np.random.randint(90, 200, n_samples)
    chol = np.random.randint(120, 564, n_samples)
    fbs = np.random.randint(0, 2, n_samples)
    restecg = np.random.randint(0, 3, n_samples)
    thalach = np.random.randint(71, 202, n_samples)
    exang = np.random.randint(0, 2, n_samples)
    oldpeak = np.random.uniform(0, 6.2, n_samples)
    slope = np.random.randint(0, 3, n_samples)
    ca = np.random.randint(0, 4, n_samples)
    thal = np.random.randint(0, 3, n_samples)

    # Simulate target variable (heart disease presence)
    # A simple linear combination with some noise for simulation
    target = (0.05 * age + 0.8 * sex + 0.3 * cp + 0.02 * trestbps + 0.01 * chol - 0.5 * thalach + 0.7 * exang + np.random.randn(n_samples) * 5) > 30
    target = target.astype(int)

    data = np.column_stack([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])
    return data, target

X, y = generate_synthetic_data()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Model Training Module
models = {
    "Logistic Regression": LogisticRegression(random_state=42, solver='liblinear'),
    "Support Vector Machine": SVC(random_state=42, probability=True),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

trained_models = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    trained_models[name] = model

# 3. Model Evaluation Module
print("\n--- Model Evaluation ---")
best_model_name = ""
best_roc_auc = -1

for name, model in trained_models.items():
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"\nModel: {name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

    if roc_auc > best_roc_auc:
        best_roc_auc = roc_auc
        best_model_name = name

print(f"\n--- Best Performing Model: {best_model_name} (ROC AUC: {best_roc_auc:.4f}) ---")

# 4. Prediction Interface (demonstrative)
def predict_heart_disease(model, scaler, new_patient_data):
    new_patient_data_scaled = scaler.transform(np.array(new_patient_data).reshape(1, -1))
    prediction = model.predict(new_patient_data_scaled)[0]
    probability = model.predict_proba(new_patient_data_scaled)[0, 1]
    return "Positive" if prediction == 1 else "Negative", probability

if best_model_name:
    best_model = trained_models[best_model_name]
    
    # Example new patient data (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
    # This should be in the same order and scale as the training data features before scaling.
    # For demonstration, let's use a random sample from the test set for 'new data' concept
    # Or a completely new synthetic data point:
    new_patient_example = [55, 1, 2, 140, 230, 0, 1, 150, 0, 1.5, 2, 1, 2] # Example values
    
    diagnosis, probability = predict_heart_disease(best_model, scaler, new_patient_example)
    print(f"\n--- Demonstrative Prediction Using {best_model_name} ---")
    print(f"New Patient Data: {new_patient_example}")
    print(f"Predicted Diagnosis: {diagnosis} (Probability of Heart Disease: {probability:.4f})")
