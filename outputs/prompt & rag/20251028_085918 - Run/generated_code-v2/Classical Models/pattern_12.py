import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def generate_simulated_data(num_samples=1000):
    np.random.seed(42)
    data = {
        'Age': np.random.randint(20, 80, num_samples),
        'BMI': np.random.uniform(18, 40, num_samples),
        'BloodPressure': np.random.randint(90, 180, num_samples),
        'Cholesterol': np.random.randint(150, 250, num_samples),
        'Glucose': np.random.randint(70, 200, num_samples),
        'FamilyHistory': np.random.choice([0, 1], num_samples, p=[0.7, 0.3]),
        'Smoker': np.random.choice([0, 1], num_samples, p=[0.8, 0.2]),
        'ExerciseHoursWeek': np.random.uniform(0, 10, num_samples),
        'Disease': np.random.choice([0, 1], num_samples, p=[0.6, 0.4]) # Target variable
    }
    df = pd.DataFrame(data)
    
    # Introduce some correlation for 'Disease'
    df.loc[df['BMI'] > 30, 'Disease'] = np.random.choice([0, 1], len(df[df['BMI'] > 30]), p=[0.3, 0.7])
    df.loc[df['BloodPressure'] > 140, 'Disease'] = np.random.choice([0, 1], len(df[df['BloodPressure'] > 140]), p=[0.4, 0.6])
    df.loc[df['Cholesterol'] > 220, 'Disease'] = np.random.choice([0, 1], len(df[df['Cholesterol'] > 220]), p=[0.35, 0.65])
    df.loc[df['Glucose'] > 120, 'Disease'] = np.random.choice([0, 1], len(df[df['Glucose'] > 120]), p=[0.25, 0.75])
    df.loc[df['FamilyHistory'] == 1, 'Disease'] = np.random.choice([0, 1], len(df[df['FamilyHistory'] == 1]), p=[0.4, 0.6])
    df.loc[df['Smoker'] == 1, 'Disease'] = np.random.choice([0, 1], len(df[df['Smoker'] == 1]), p=[0.45, 0.55])
    
    df['Disease'] = df['Disease'].astype(int)
    return df

def preprocess_data(df, target_column='Disease', scaler=None, fit_scaler=True):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    numerical_cols = X.select_dtypes(include=np.number).columns
    
    if fit_scaler:
        scaler = StandardScaler()
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    else:
        if scaler is None:
            raise ValueError("Scaler must be provided for transformation if fit_scaler is False.")
        X[numerical_cols] = scaler.transform(X[numerical_cols])
        
    return X, y, scaler

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n--- {model_name} Evaluation ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    return model

def predict_disease(new_data, trained_model, scaler):
    new_df = pd.DataFrame([new_data])
    numerical_cols = new_df.select_dtypes(include=np.number).columns
    new_df[numerical_cols] = scaler.transform(new_df[numerical_cols])
    prediction = trained_model.predict(new_df)
    probability = trained_model.predict_proba(new_df)[:, 1]
    return "Positive" if prediction[0] == 1 else "Negative", probability[0]

if __name__ == "__main__":
    print("--- Disease Prediction System ---\n")

    # 1. Generate Simulated Data
    print("Generating simulated medical data...")
    df = generate_simulated_data(num_samples=1000)
    print(f"Generated {len(df)} samples.\n")

    # 2. Preprocess Data
    print("Preprocessing data (scaling features)...")
    X, y, scaler = preprocess_data(df, target_column='Disease', fit_scaler=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data split into {len(X_train)} training and {len(X_test)} testing samples.\n")

    # 3. Model Training and Evaluation
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, solver='liblinear'),
        "Support Vector Machine": SVC(probability=True, random_state=42),
        "Decision Tree Classifier": DecisionTreeClassifier(random_state=42)
    }

    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        trained_models[name] = train_and_evaluate_model(model, X_train, y_train, X_test, y_test, name)

    # 4. Demonstrate Prediction
    print("\n--- Demonstrating Prediction with New Data ---")
    new_patient_data = {
        'Age': 65,
        'BMI': 32.5,
        'BloodPressure': 160,
        'Cholesterol': 230,
        'Glucose': 130,
        'FamilyHistory': 1,
        'Smoker': 0,
        'ExerciseHoursWeek': 2.0
    }

    print(f"\nNew Patient Data: {new_patient_data}")
    
    # Using Logistic Regression for demonstration
    model_to_use = "Logistic Regression"
    predicted_status, prediction_proba = predict_disease(new_patient_data, trained_models[model_to_use], scaler)
    print(f"Prediction using {model_to_use}: Disease Status: {predicted_status}, Probability of Disease: {prediction_proba:.4f}")

    new_patient_data_healthy = {
        'Age': 30,
        'BMI': 22.0,
        'BloodPressure': 110,
        'Cholesterol': 170,
        'Glucose': 90,
        'FamilyHistory': 0,
        'Smoker': 0,
        'ExerciseHoursWeek': 7.0
    }
    print(f"\nNew Patient Data (Healthy Profile): {new_patient_data_healthy}")
    predicted_status_healthy, prediction_proba_healthy = predict_disease(new_patient_data_healthy, trained_models[model_to_use], scaler)
    print(f"Prediction using {model_to_use}: Disease Status: {predicted_status_healthy}, Probability of Disease: {prediction_proba_healthy:.4f}")