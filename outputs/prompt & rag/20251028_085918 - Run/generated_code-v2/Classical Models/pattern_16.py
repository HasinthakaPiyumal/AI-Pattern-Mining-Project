import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Data Ingestion & Storage: Simulate patient data
def generate_simulated_data(num_samples=1000):
    np.random.seed(42)
    data = {
        'Age': np.random.randint(20, 80, num_samples),
        'Fever': np.random.randint(0, 2, num_samples),
        'Cough': np.random.randint(0, 2, num_samples),
        'Fatigue': np.random.randint(0, 2, num_samples),
        'Blood_Pressure': np.random.randint(90, 180, num_samples),
        'Cholesterol': np.random.randint(150, 250, num_samples),
        'Diagnosis': np.random.randint(0, 2, num_samples) # 0 for No Disease, 1 for Disease
    }
    df = pd.DataFrame(data)
    # Introduce some correlation for 'Diagnosis'
    df['Diagnosis'] = (df['Fever'] * 0.3 + df['Cough'] * 0.2 + df['Fatigue'] * 0.1 + 
                       (df['Blood_Pressure'] > 140) * 0.2 + (df['Cholesterol'] > 200) * 0.1 + 
                       np.random.rand(num_samples) * 0.5 > 0.5).astype(int)
    return df

# Generate data
patient_data = generate_simulated_data()

# Separate features and target
X = patient_data.drop('Diagnosis', axis=1)
y = patient_data['Diagnosis']

# 2. Data Preprocessing
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Model Training & Selection
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Support Vector Machine': SVC(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

trained_models = {}
metrics = {} 

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    trained_models[name] = model
    y_pred = model.predict(X_test_scaled)
    
    metrics[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred)
    }

# Output evaluation metrics
print("\n--- Model Evaluation Metrics ---")
for name, model_metrics in metrics.items():
    print(f"\n{name}:")
    for metric_name, value in model_metrics.items():
        print(f"  {metric_name}: {value:.4f}")

# 4. Prediction Service
def predict_diagnosis(new_patient_data_raw, scaler_obj, trained_model, feature_names):
    # Ensure new_patient_data_raw is a DataFrame
    if isinstance(new_patient_data_raw, dict):
        new_patient_data_df = pd.DataFrame([new_patient_data_raw])
    elif isinstance(new_patient_data_raw, list):
         new_patient_data_df = pd.DataFrame(new_patient_data_raw)
    else:
        raise ValueError("new_patient_data_raw must be a dictionary or a list of dictionaries.")

    # Reorder columns to match training data order
    new_patient_data_df = new_patient_data_df[feature_names]
    
    # Preprocess new data
    new_patient_data_scaled = scaler_obj.transform(new_patient_data_df)
    
    # Predict
    prediction = trained_model.predict(new_patient_data_scaled)
    return "Disease" if prediction[0] == 1 else "No Disease"

# 5. Output & Visualization (Basic): Show a sample prediction
print("\n--- Sample Prediction ---")
sample_patient = X_test.iloc[0].to_dict()
print(f"Sample Patient Data: {sample_patient}")

# Use Logistic Regression model for sample prediction
selected_model_name = 'Logistic Regression'
predicted_diagnosis = predict_diagnosis(sample_patient, scaler, trained_models[selected_model_name], X.columns)
actual_diagnosis = "Disease" if y_test.iloc[0] == 1 else "No Disease"

print(f"Predicted Diagnosis ({selected_model_name}): {predicted_diagnosis}")
print(f"Actual Diagnosis: {actual_diagnosis}")

# Another sample prediction for a different model (e.g., Decision Tree)
print("\n--- Another Sample Prediction (Decision Tree) ---")
sample_patient_2 = X_test.iloc[10].to_dict()
print(f"Sample Patient Data: {sample_patient_2}")

selected_model_name_2 = 'Decision Tree'
predicted_diagnosis_2 = predict_diagnosis(sample_patient_2, scaler, trained_models[selected_model_name_2], X.columns)
actual_diagnosis_2 = "Disease" if y_test.iloc[10] == 1 else "No Disease"

print(f"Predicted Diagnosis ({selected_model_name_2}): {predicted_diagnosis_2}")
print(f"Actual Diagnosis: {actual_diagnosis_2}")