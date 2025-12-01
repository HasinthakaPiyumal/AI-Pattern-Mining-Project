import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Union


# --- 1. Data Ingestion and Preprocessing & 2. Machine Learning Model Training ---
class Trainer:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.numerical_features = ['age', 'tumor_size', 'initial_lymph_nodes']
        self.categorical_features = ['genetic_mutation_X', 'previous_treatment_Y', 'tumor_grade']

    def generate_dummy_data(self, n_samples=1000):
        np.random.seed(42)
        data = {
            'age': np.random.randint(30, 80, n_samples),
            'tumor_size': np.random.uniform(1.0, 15.0, n_samples),
            'genetic_mutation_X': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'previous_treatment_Y': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'initial_lymph_nodes': np.random.randint(0, 10, n_samples),
            'tumor_grade': np.random.choice(['G1', 'G2', 'G3'], n_samples, p=[0.3, 0.4, 0.3])
        }
        df = pd.DataFrame(data)
        
        # Simulate a complex response based on features
        df['response'] = (
            0.2 * df['age']
            + 1.5 * df['tumor_size']
            - 10 * df['genetic_mutation_X']  # Mutation might be bad for response
            + 5 * df['previous_treatment_Y'] # Some previous treatments might improve response
            - 2 * df['initial_lymph_nodes']
            + np.where(df['tumor_grade'] == 'G1', 10, np.where(df['tumor_grade'] == 'G2', 5, -5))
            + np.random.normal(0, 5, n_samples)
        ) > (np.mean(df['age']) + np.mean(df['tumor_size'])) # Binary classification
        df['response'] = df['response'].astype(int)
        return df

    def preprocess_data(self, df):
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_features)
            ])
        
        X = df.drop('response', axis=1)
        y = df['response']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.preprocessor = preprocessor.fit(X_train)
        X_train_processed = self.preprocessor.transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        
        # Capture feature names after one-hot encoding for ICE plots
        cat_feature_names = self.preprocessor.named_transformers_['cat'].get_feature_names_out(self.categorical_features)
        self.feature_names = self.numerical_features + list(cat_feature_names)

        return X_train_processed, X_test_processed, y_train, y_test

    def train_model(self, X_train, y_train):
        self.model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, random_state=42)
        self.model.fit(X_train, y_train)

    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.4f}")

    def save_artifacts(self, model_path='xgb_model.joblib', preprocessor_path='preprocessor.joblib', feature_names_path='feature_names.joblib'):
        joblib.dump(self.model, model_path)
        joblib.dump(self.preprocessor, preprocessor_path)
        joblib.dump(self.feature_names, feature_names_path)

    def load_artifacts(self, model_path='xgb_model.joblib', preprocessor_path='preprocessor.joblib', feature_names_path='feature_names.joblib'):
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        self.feature_names = joblib.load(feature_names_path)


# --- FastAPI Application --- 
app = FastAPI()

# Global variables to load model and preprocessor once
ml_model: xgb.XGBClassifier = None
ml_preprocessor: Pipeline = None
ml_feature_names: List[str] = None

# Pydantic models for request/response
class PatientData(BaseModel):
    age: int
    tumor_size: float
    genetic_mutation_X: int
    previous_treatment_Y: int
    initial_lymph_nodes: int
    tumor_grade: str

class PredictionResponse(BaseModel):
    prediction: int
    probability_positive_response: float

class ICEPlotRequest(BaseModel):
    patient_data: PatientData
    target_feature: str

@app.on_event("startup")
async def load_model_and_preprocessor():
    global ml_model, ml_preprocessor, ml_feature_names
    try:
        trainer = Trainer()
        trainer.load_artifacts()
        ml_model = trainer.model
        ml_preprocessor = trainer.preprocessor
        ml_feature_names = trainer.feature_names
        print("Model and preprocessor loaded successfully!")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load ML artifacts: {e}")


@app.post("/predict", response_model=PredictionResponse)
async def predict_treatment_response(patient: PatientData):
    if ml_model is None or ml_preprocessor is None:
        raise HTTPException(status_code=500, detail="ML model or preprocessor not loaded.")

    patient_df = pd.DataFrame([patient.dict()])
    processed_patient_data = ml_preprocessor.transform(patient_df)

    prediction = ml_model.predict(processed_patient_data)[0]
    probability = ml_model.predict_proba(processed_patient_data)[0][1]

    return PredictionResponse(prediction=int(prediction), probability_positive_response=float(probability))


@app.post("/ice_plot")
async def generate_ice_plot(request: ICEPlotRequest):
    if ml_model is None or ml_preprocessor is None or ml_feature_names is None:
        raise HTTPException(status_code=500, detail="ML model, preprocessor, or feature names not loaded.")

    patient_data = request.patient_data
    target_feature = request.target_feature

    original_df = pd.DataFrame([patient_data.dict()])

    if target_feature not in original_df.columns:
        raise HTTPException(status_code=400, detail=f"Target feature '{target_feature}' not found in patient data.")

    # Determine if the feature is numerical or categorical from the original features
    trainer_instance = Trainer() # Use a temporary trainer instance to access feature lists
    is_numerical = target_feature in trainer_instance.numerical_features
    is_categorical = target_feature in trainer_instance.categorical_features

    if not (is_numerical or is_categorical):
        raise HTTPException(status_code=400, detail=f"Target feature '{target_feature}' is not recognized as a numerical or categorical feature for ICE plot generation.")

    predictions = []
    feature_values_to_plot = []

    if is_numerical:
        min_val = original_df[target_feature].min() - (original_df[target_feature].std() * 2) if original_df[target_feature].std() > 0 else original_df[target_feature].min() * 0.5
        max_val = original_df[target_feature].max() + (original_df[target_feature].std() * 2) if original_df[target_feature].std() > 0 else original_df[target_feature].max() * 1.5
        
        # Ensure reasonable bounds, especially if std is 0 or very small
        if min_val == max_val: # Handle cases where there's no variance
            min_val = min_val * 0.5 if min_val != 0 else -1
            max_val = max_val * 1.5 if max_val != 0 else 1
        
        feature_range = np.linspace(min_val, max_val, 50)

        for val in feature_range:
            temp_df = original_df.copy()
            temp_df[target_feature] = val
            processed_temp_data = ml_preprocessor.transform(temp_df)
            pred_prob = ml_model.predict_proba(processed_temp_data)[0][1]
            predictions.append(pred_prob)
            feature_values_to_plot.append(val)

    elif is_categorical:
        unique_categories = original_df[target_feature].unique().tolist() if target_feature in original_df.columns else []
        
        # Add all possible categories seen during training for a robust plot
        if ml_preprocessor is not None and 'cat' in ml_preprocessor.named_transformers_:
            ohe = ml_preprocessor.named_transformers_['cat']
            try:
                cat_idx = trainer_instance.categorical_features.index(target_feature)
                all_possible_categories = ohe.categories_[cat_idx].tolist()
                unique_categories = sorted(list(set(unique_categories + all_possible_categories)))
            except ValueError:
                pass # target_feature not in preprocessor's known categorical features

        if not unique_categories:
            raise HTTPException(status_code=400, detail=f"Could not determine categories for '{target_feature}'.")

        for cat_val in unique_categories:
            temp_df = original_df.copy()
            temp_df[target_feature] = cat_val
            processed_temp_data = ml_preprocessor.transform(temp_df)
            pred_prob = ml_model.predict_proba(processed_temp_data)[0][1]
            predictions.append(pred_prob)
            feature_values_to_plot.append(cat_val)

    plt.figure(figsize=(10, 6))
    sns.lineplot(x=feature_values_to_plot, y=predictions, marker='o', color='blue')
    plt.title(f"ICE Plot for {target_feature} for a Specific Patient")
    plt.xlabel(target_feature)
    plt.ylabel("Predicted Probability of Positive Response")
    plt.grid(True)

    if is_categorical:
        plt.xticks(rotation=45, ha='right')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close() # Close the plot to free memory
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return {"ice_plot_image_base64": image_base64}

# --- Main execution block for training and then running the app ---
if __name__ == "__main__":
    # Training part (run once to generate model and preprocessor files)
    print("Starting model training...")
    trainer = Trainer()
    df = trainer.generate_dummy_data()
    X_train_processed, X_test_processed, y_train, y_test = trainer.preprocess_data(df)
    trainer.train_model(X_train_processed, y_train)
    trainer.evaluate_model(X_test_processed, y_test)
    trainer.save_artifacts()
    print("Model training complete and artifacts saved.")

    # Run the FastAPI app
    print("Starting FastAPI application. Use 'uvicorn main:app --reload' to run in development.")
    # For direct execution within __main__, you'd typically use uvicorn.run, 
    # but for a single file submission, just having the app object is sufficient.
    # To run this, save as main.py and execute: uvicorn main:app --host 0.0.0.0 --port 8000
