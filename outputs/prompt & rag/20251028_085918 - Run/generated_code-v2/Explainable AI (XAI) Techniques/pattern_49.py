from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import random
import math

app = FastAPI()

# Global storage for data and predictions
_global_data = []
_global_features = []
_global_predictions = []

class DataInput(BaseModel):
    data: list[dict] # Example: [{"age": 30, "gender": "male"}]
    features: list[str] # List of feature names to consider for itemsets

@app.post("/load_data")
async def load_data(data_input: DataInput):
    global _global_data, _global_features
    _global_data = data_input.data
    _global_features = data_input.features
    if not _global_data:
        raise HTTPException(status_code=400, detail="No data provided.")
    return {"message": "Data loaded successfully", "num_records": len(_global_data)}

@app.post("/predict")
async def run_mock_predictions():
    global _global_predictions
    if not _global_data:
        raise HTTPException(status_code=400, detail="No data loaded. Please load data first.")

    # Simulate a binary classification model (0 or 1)
    _global_predictions = [random.randint(0, 1) for _ in _global_data]
    return {"message": "Mock predictions generated", "num_predictions": len(_global_predictions)}

def calculate_divergence_score(subgroup_predictions, global_predictions):
    if not global_predictions or not subgroup_predictions:
        return 0.0
    
    # Simple divergence: absolute difference in positive prediction rate
    global_pos_rate = sum(global_predictions) / len(global_predictions)
    subgroup_pos_rate = sum(subgroup_predictions) / len(subgroup_predictions)
    
    return abs(subgroup_pos_rate - global_pos_rate)

@app.post("/run_divexplorer")
async def run_divexplorer(min_support: float = 0.01, max_itemset_size: int = 2):
    if not _global_data or not _global_predictions:
        raise HTTPException(status_code=400, detail="Data or predictions not loaded. Please load data and run predictions first.")

    all_candidate_itemsets = []
    
    # Generate single itemsets
    for feature in _global_features:
        unique_values = set(record.get(feature) for record in _global_data if feature in record)
        for value in unique_values:
            if value is not None: # Exclude None values from itemsets
                all_candidate_itemsets.append({feature: value})
    
    # Generate two-item itemsets (up to max_itemset_size)
    if max_itemset_size >= 2:
        single_items = []
        for itemset in all_candidate_itemsets:
            if len(itemset) == 1:
                feature, value = next(iter(itemset.items()))
                single_items.append((feature, value))

        for i in range(len(single_items)):
            for j in range(i + 1, len(single_items)):
                item1_feature, item1_value = single_items[i]
                item2_feature, item2_value = single_items[j]
                # Ensure distinct features for 2-item itemsets to avoid redundancy
                if item1_feature != item2_feature:
                    all_candidate_itemsets.append({item1_feature: item1_value, item2_feature: item2_value})
        
    results = []
    for itemset in all_candidate_itemsets:
        subgroup_data_indices = []
        for i, record in enumerate(_global_data):
            match = True
            for feature, value in itemset.items():
                if record.get(feature) != value:
                    match = False
                    break
            if match:
                subgroup_data_indices.append(i)

        subgroup_size = len(subgroup_data_indices)
        current_support = subgroup_size / len(_global_data) if _global_data else 0

        if current_support >= min_support:
            subgroup_predictions = [_global_predictions[i] for i in subgroup_data_indices]
            divergence = calculate_divergence_score(subgroup_predictions, _global_predictions)
            
            # Mock Shapley values for demonstration of local item contributions
            mock_shapley_values = {}
            for feature, value in itemset.items():
                # Simulate some impact, higher impact for more divergent itemsets
                mock_shapley_values[f"{feature}={value}"] = random.uniform(-0.3, 0.3) + (divergence * random.uniform(0.5, 1.0) if divergence > 0.1 else 0)

            results.append({
                "itemset": itemset,
                "support": current_support,
                "divergence_score": divergence,
                "subgroup_size": subgroup_size,
                "mock_local_contributions": mock_shapley_values # Placeholder for local item contributions
            })
    
    # Sort by divergence score (descending)
    results.sort(key=lambda x: x["divergence_score"], reverse=True)
    
    return {"divergent_subgroups": results}
