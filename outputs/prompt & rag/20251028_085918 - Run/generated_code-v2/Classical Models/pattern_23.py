import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)
    data = {
        "Age": np.random.randint(20, 70, num_samples),
        "Gender": np.random.choice(["Male", "Female"], num_samples),
        "BMI": np.random.uniform(18.0, 40.0, num_samples),
        "BloodPressure_Systolic": np.random.randint(90, 180, num_samples),
        "BloodPressure_Diastolic": np.random.randint(60, 120, num_samples),
        "Cholesterol_Total": np.random.randint(150, 280, num_samples),
        "Glucose": np.random.randint(70, 200, num_samples),
        "FamilyHistory": np.random.choice([0, 1], num_samples, p=[0.7, 0.3]),
        "SmokingStatus": np.random.choice([0, 1], num_samples, p=[0.8, 0.2]),
        "Diabetes": np.random.choice([0, 1], num_samples, p=[0.75, 0.25])
    }
    df = pd.DataFrame(data)
    df["BMI"] = np.round(df["BMI"], 2)
    df["Glucose"] = np.round(df["Glucose"].apply(lambda x: x + np.random.normal(0, 5) if x > 120 else x), 0)
    df["Diabetes"] = ((df["Glucose"] > 140) | (df["BMI"] > 30) | (df["Age"] > 50) | (df["FamilyHistory"] == 1)).astype(int)
    return df

def preprocess_data(df):
    categorical_cols = ["Gender"]
    numerical_cols = ["Age", "BMI", "BloodPressure_Systolic", "BloodPressure_Diastolic", "Cholesterol_Total", "Glucose"]
    binary_cols = ["FamilyHistory", "SmokingStatus"]

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoded_features = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols))

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[numerical_cols])
    scaled_df = pd.DataFrame(scaled_features, columns=numerical_cols)

    processed_df = pd.concat([scaled_df, encoded_df, df[binary_cols]], axis=1)
    return processed_df, scaler, encoder

def train_evaluate_model(X_train, X_test, y_train, y_test, model, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    return {
        "model_name": model_name,
        "model": model,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    }

def predict_disease(model, scaler, encoder, new_data):
    categorical_cols = ["Gender"]
    numerical_cols = ["Age", "BMI", "BloodPressure_Systolic", "BloodPressure_Diastolic", "Cholesterol_Total", "Glucose"]

    new_data_df = pd.DataFrame([new_data])

    encoded_features = encoder.transform(new_data_df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols))

    scaled_features = scaler.transform(new_data_df[numerical_cols])
    scaled_df = pd.DataFrame(scaled_features, columns=numerical_cols)

    processed_new_data = pd.concat([scaled_df, encoded_df, new_data_df[["FamilyHistory", "SmokingStatus"]].reset_index(drop=True)], axis=1)

    prediction = model.predict(processed_new_data)[0]
    prediction_proba = model.predict_proba(processed_new_data)[0][1]
    return {"prediction": int(prediction), "probability": float(prediction_proba)}

if __name__ == "__main__":
    df = generate_synthetic_data(num_samples=2000)

    X = df.drop("Diabetes", axis=1)
    y = df["Diabetes"]

    X_processed, scaler, encoder = preprocess_data(X)

    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "LogisticRegression": LogisticRegression(random_state=42, solver='liblinear'),
        "RandomForestClassifier": RandomForestClassifier(random_state=42),
        "XGBoostClassifier": xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
        "LightGBMClassifier": lgb.LGBMClassifier(random_state=42)
    }

    results = []
    for name, model in models.items():
        result = train_evaluate_model(X_train, X_test, y_train, y_test, model, name)
        results.append(result)

    best_model_info = max(results, key=lambda item: item['roc_auc'])

    joblib.dump(best_model_info["model"], "best_disease_prediction_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(encoder, "encoder.pkl")

    loaded_model = joblib.load("best_disease_prediction_model.pkl")
    loaded_scaler = joblib.load("scaler.pkl")
    loaded_encoder = joblib.load("encoder.pkl")

    new_patient_data = {
        "Age": 55,
        "Gender": "Female",
        "BMI": 32.5,
        "BloodPressure_Systolic": 145,
        "BloodPressure_Diastolic": 90,
        "Cholesterol_Total": 230,
        "Glucose": 160,
        "FamilyHistory": 1,
        "SmokingStatus": 0
    }

    prediction_result = predict_disease(loaded_model, loaded_scaler, loaded_encoder, new_patient_data)
