import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import dice_ml
import os

class LoanApprovalModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None

    def _generate_synthetic_data(self, n_samples=1000):
        np.random.seed(42)
        data = {
            "credit_score": np.random.randint(300, 850, n_samples),
            "debt_to_income": np.random.uniform(0.1, 0.6, n_samples),
            "employment_years": np.random.randint(0, 20, n_samples),
            "loan_amount": np.random.randint(5000, 100000, n_samples),
            "loan_term_years": np.random.randint(1, 10, n_samples),
            "has_collateral": np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
            "previous_loan_defaults": np.random.randint(0, 3, n_samples)
        }
        df = pd.DataFrame(data)

        df["approved"] = (
            (df["credit_score"] > 650).astype(int) +
            (df["debt_to_income"] < 0.4).astype(int) +
            (df["employment_years"] > 2).astype(int) +
            (df["has_collateral"] == 1).astype(int) +
            (df["previous_loan_defaults"] == 0).astype(int)
        )
        df["approved"] = (df["approved"] >= 3).astype(int)
        df["approved"] = df.apply(lambda row: 1 if np.random.rand() < 0.1 else row["approved"], axis=1)
        df["approved"] = df.apply(lambda row: 0 if np.random.rand() < 0.05 else row["approved"], axis=1)

        return df

    def train_model(self, data_path=None):
        if data_path:
            df = pd.read_csv(data_path)
        else:
            df = self._generate_synthetic_data()

        X = df.drop("approved", axis=1)
        y = df["approved"]
        self.feature_names = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model = LogisticRegression(random_state=42, solver="liblinear")
        self.model.fit(X_train_scaled, y_train)

        print(f"Model trained with accuracy: {self.model.score(X_test_scaled, y_test):.4f}")

    def predict_loan(self, applicant_features: pd.DataFrame):
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model or scaler not trained. Please call train_model() first.")
        
        if not all(feature in applicant_features.columns for feature in self.feature_names):
            missing_features = [f for f in self.feature_names if f not in applicant_features.columns]
            raise ValueError(f"Missing required features: {missing_features}")

        applicant_scaled = self.scaler.transform(applicant_features[self.feature_names])
        prediction = self.model.predict(applicant_scaled)[0]
        prediction_proba = self.model.predict_proba(applicant_scaled)[0]
        return prediction, prediction_proba

    def save_model(self, model_path="loan_model.joblib", scaler_path="scaler.joblib", feature_names_path="feature_names.joblib"):
        if self.model is None or self.scaler is None:
            raise RuntimeError("No model or scaler to save. Train the model first.")
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.feature_names, feature_names_path)
        print(f"Model, scaler, and feature names saved to {model_path}, {scaler_path}, {feature_names_path}")

    @staticmethod
    def load_model(model_path="loan_model.joblib", scaler_path="scaler.joblib", feature_names_path="feature_names.joblib"):
        model_instance = LoanApprovalModel()
        model_instance.model = joblib.load(model_path)
        model_instance.scaler = joblib.load(scaler_path)
        model_instance.feature_names = joblib.load(feature_names_path)
        print(f"Model, scaler, and feature names loaded from {model_path}, {scaler_path}, {feature_names_path}")
        return model_instance

app = FastAPI(title="CreditInsight: Explainable Loan Approval System")

loan_predictor = LoanApprovalModel()
MODEL_PATH = "loan_model.joblib"
SCALER_PATH = "scaler.joblib"
FEATURE_NAMES_PATH = "feature_names.joblib"

@app.on_event("startup")
async def startup_event():
    print("Loading or training model on startup...")
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(FEATURE_NAMES_PATH):
        try:
            global loan_predictor
            loan_predictor = LoanApprovalModel.load_model(MODEL_PATH, SCALER_PATH, FEATURE_NAMES_PATH)
            print("Existing model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}. Retraining a new model.")
            loan_predictor.train_model()
            loan_predictor.save_model(MODEL_PATH, SCALER_PATH, FEATURE_NAMES_PATH)
            print("New model trained and saved.")
    else:
        print("Model files not found. Training a new model.")
        loan_predictor.train_model()
        loan_predictor.save_model(MODEL_PATH, SCALER_PATH, FEATURE_NAMES_PATH)
        print("New model trained and saved.")

class LoanApplicationFeatures(BaseModel):
    credit_score: int
    debt_to_income: float
    employment_years: int
    loan_amount: int
    loan_term_years: int
    has_collateral: int
    previous_loan_defaults: int

class ExplainRequest(BaseModel):
    applicant_data: LoanApplicationFeatures
    desired_outcome: int = 1

def generate_counterfactual_explanation_dice(applicant_data_df: pd.DataFrame, desired_outcome: int):
    model = loan_predictor.model
    scaler = loan_predictor.scaler
    feature_names = loan_predictor.feature_names

    def predict_proba_func(data_instance_df):
        data_scaled = scaler.transform(data_instance_df[feature_names])
        return model.predict_proba(data_scaled)

    synthetic_data_for_dice = loan_predictor._generate_synthetic_data(n_samples=100)
    synthetic_data_for_dice = synthetic_data_for_dice.drop("approved", axis=1)

    numerical_features = feature_names
    categorical_features = []

    d = dice_ml.Data(dataframe=synthetic_data_for_dice, 
                     continuous_features=numerical_features, 
                     outcome_name="approved")

    m = dice_ml.Model(model=predict_proba_func, backend="sklearn", model_type="classifier")

    explainer = dice_ml.Dice(d, m, method="kdtree")

    query_instance = applicant_data_df[feature_names]

    dice_exp = explainer.generate_counterfactuals(
        query_instance, 
        total_CFs=3, 
        desired_class=desired_outcome, 
        proximity_weight=0.5, 
        diversity_weight=1.0
    )

    cf_df = dice_exp.cf_examples_list[0].final_cfs_df

    explanation_results = []
    if cf_df is not None and not cf_df.empty:
        for i, row in cf_df.iterrows():
            changes = {}
            for col in feature_names:
                original_val = query_instance.iloc[0][col]
                cf_val = row[col]
                if isinstance(original_val, (float, np.float32, np.float64)) and isinstance(cf_val, (float, np.float32, np.float64)):
                    if not np.isclose(original_val, cf_val):
                        changes[col] = {"original": round(original_val, 2), "counterfactual": round(cf_val, 2)}
                else:
                    if original_val != cf_val:
                        changes[col] = {"original": original_val, "counterfactual": cf_val}
            
            if changes:
                explanation_results.append({
                    "counterfactual_id": i + 1,
                    "changes": changes,
                    "predicted_outcome": "Approved" if row["approved"] == 1 else "Denied",
                    "description": f"Change {', '.join([f'{k} from {v["original"]} to {v["counterfactual"]}' for k, v in changes.items()])} to get an {'Approved' if row['approved'] == 1 else 'Denied'} loan."
                })
            else:
                explanation_results.append({
                    "counterfactual_id": i + 1,
                    "changes": "No explicit feature changes found, but prediction flipped.",
                    "predicted_outcome": "Approved" if row["approved"] == 1 else "Denied",
                    "description": "The model found a counterfactual with no explicit feature changes, implying a very close decision boundary or numerical precision differences."
                })
    else:
        explanation_results.append("No counterfactuals found.")

    return explanation_results

@app.post("/predict")
async def predict_loan_status(application: LoanApplicationFeatures):
    try:
        applicant_df = pd.DataFrame([application.dict()])
        prediction, prediction_proba = loan_predictor.predict_loan(applicant_df)
        status = "Approved" if prediction == 1 else "Denied"
        return {
            "status": status,
            "probability_approved": prediction_proba[1].item(),
            "probability_denied": prediction_proba[0].item()
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Model not ready: {e}")

@app.post("/explain")
async def get_counterfactual_explanation(request: ExplainRequest):
    try:
        applicant_df = pd.DataFrame([request.applicant_data.dict()])
        original_prediction, original_proba = loan_predictor.predict_loan(applicant_df)
        original_decision = "Approved" if original_prediction == 1 else "Denied"

        if original_prediction == request.desired_outcome:
            return {
                "original_application": request.applicant_data.dict(),
                "original_prediction": original_decision,
                "original_probabilities": {"approved": original_proba[1].item(), "denied": original_proba[0].item()},
                "counterfactual_explanation": "N/A - Desired outcome already achieved."
            }

        explanations = generate_counterfactual_explanation_dice(applicant_df, request.desired_outcome)
        
        return {
            "original_application": request.applicant_data.dict(),
            "original_prediction": original_decision,
            "original_probabilities": {"approved": original_proba[1].item(), "denied": original_proba[0].item()},
            "counterfactual_explanations": explanations
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Model not ready: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating explanation: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)