from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import shap
import json

app = FastAPI()

# --- Mock Data and Model Storage ---
# In a real application, this would be persisted (e.g., database, file system)
data_store = {"data": None, "model": None, "feature_names": None}
divexplorer_results_store = {"subgroups": [], "lattice": {}}

# --- Mock Black-Box Model --- 
class MockBlackBoxModel:
    def predict_proba(self, X):
        # Simulate probability predictions
        return np.random.rand(X.shape[0], 2)

    def predict(self, X):
        # Simulate class predictions
        return np.random.randint(0, 2, X.shape[0])

# --- Mock DivExplorer Algorithm --- 
class MockDivExplorer:
    def __init__(self, data, model, feature_names):
        self.data = data
        self.model = model
        self.feature_names = feature_names

    def run(self):
        # Simulate divergent subgroup discovery
        num_subgroups = 5
        subgroups = []
        for i in range(num_subgroups):
            itemset = f"Feature{i+1} > 0.5 AND Feature{i+2} < 0.2"
            subgroups.append({
                "id": i + 1,
                "itemset": itemset,
                "divergence_score": round(np.random.uniform(0.1, 0.9), 3),
                "subgroup_size": np.random.randint(50, 500),
                "observed_performance": round(np.random.uniform(0.6, 0.9), 3),
                "expected_performance": round(np.random.uniform(0.7, 0.95), 3),
            })
        return subgroups

# --- Pydantic Models for API --- 
class DivergentSubgroup(BaseModel):
    id: int
    itemset: str
    divergence_score: float
    subgroup_size: int
    observed_performance: float
    expected_performance: float

class ShapleyValue(BaseModel):
    feature: str
    value: float

class LatticeNode(BaseModel):
    id: int
    label: str

class LatticeEdge(BaseModel):
    source: int
    target: int

class LatticeData(BaseModel):
    nodes: List[LatticeNode]
    edges: List[LatticeEdge]

# --- API Endpoints --- 
@app.post("/upload-data-model")
async def upload_data_model(data_file: UploadFile = File(...), model_file: UploadFile = File(...), feature_names: str = "[]"):
    try:
        df = pd.read_csv(data_file.file)
        data_store["data"] = df
        data_store["model"] = MockBlackBoxModel() # Assuming a mock model for now
        data_store["feature_names"] = json.loads(feature_names)
        if not data_store["feature_names"] and not df.empty:
            data_store["feature_names"] = df.columns.tolist()

        return {"message": "Data and model uploaded successfully", "data_shape": df.shape}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing upload: {e}")

@app.post("/run-divexplorer")
async def run_divexplorer():
    if data_store["data"] is None or data_store["model"] is None or data_store["feature_names"] is None:
        raise HTTPException(status_code=400, detail="Please upload data and model first.")

    divexplorer = MockDivExplorer(data_store["data"], data_store["model"], data_store["feature_names"])
    subgroups = divexplorer.run()
    divexplorer_results_store["subgroups"] = subgroups

    # Simulate lattice generation
    nodes = []
    edges = []
    for sg in subgroups:
        nodes.append(LatticeNode(id=sg["id"], label=sg["itemset"]))
        # Simple mock edges: connect subgroup i to i+1
        if sg["id"] < len(subgroups):
            edges.append(LatticeEdge(source=sg["id"], target=sg["id"] + 1))

    divexplorer_results_store["lattice"] = LatticeData(nodes=nodes, edges=edges).dict()

    return {"message": "DivExplorer run completed", "num_subgroups": len(subgroups)}

@app.get("/subgroups", response_model=List[DivergentSubgroup])
async def get_subgroups():
    return divexplorer_results_store["subgroups"]

@app.get("/subgroup/{subgroup_id}/shapley", response_model=List[ShapleyValue])
async def get_local_shapley(subgroup_id: int):
    if data_store["data"] is None or data_store["model"] is None:
        raise HTTPException(status_code=400, detail="Data or model not available.")

    # Simulate local Shapley values for a subgroup
    features = data_store["feature_names"]
    if not features:
        raise HTTPException(status_code=400, detail="Feature names not available.")

    # In a real scenario, you'd filter data for the subgroup and run SHAP
    # For this mock, we just generate random values for available features
    local_shap_values = [
        ShapleyValue(feature=f, value=round(np.random.uniform(-0.5, 0.5), 3))
        for f in features
    ]
    return local_shap_values

@app.get("/global-shapley", response_model=List[ShapleyValue])
async def get_global_shapley():
    if data_store["data"] is None or data_store["model"] is None:
        raise HTTPException(status_code=400, detail="Data or model not available.")

    # Simulate global Shapley values
    features = data_store["feature_names"]
    if not features:
        raise HTTPException(status_code=400, detail="Feature names not available.")

    # In a real scenario, you'd run SHAP on the entire dataset
    # For this mock, we just generate random values for available features
    global_shap_values = [
        ShapleyValue(feature=f, value=round(np.random.uniform(0.01, 0.8), 3))
        for f in features
    ]
    return global_shap_values

@app.get("/lattice", response_model=LatticeData)
async def get_lattice():
    if not divexplorer_results_store["lattice"]:
        raise HTTPException(status_code=400, detail="DivExplorer results not available. Please run DivExplorer first.")
    return divexplorer_results_store["lattice"]