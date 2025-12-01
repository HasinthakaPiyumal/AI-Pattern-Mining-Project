import pandas as pd
import numpy as np
import streamlit as st
import requests
import json
import plotly.express as px
import matplotlib.pyplot as plt
import networkx as nx
import io
import joblib
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import shap
from typing import List, Dict, Any, Optional

# --- div_explorer.py (Mock DivExplorer) ---
class DivExplorer:
    def __init__(self):
        pass

    def find_divergent_itemsets(self, X, y_true, y_pred, feature_names, min_support=0.01, max_items=3):
        divergent_itemsets_data = []

        np.random.seed(42)
        num_samples = len(X)
        num_features = len(feature_names)

        for _ in range(5):
            itemset_size = np.random.randint(1, max_items + 1)
            itemset_features_indices = np.random.choice(num_features, itemset_size, replace=False)
            itemset_features = [feature_names[i] for i in itemset_features_indices]

            condition_indices = np.random.choice(num_samples, int(num_samples * (np.random.rand() * 0.1 + min_support)), replace=False)
            
            if len(condition_indices) == 0:
                continue

            simulated_y_true = y_true[condition_indices]
            simulated_y_pred = y_pred[condition_indices]

            divergence_score = np.abs(np.mean(simulated_y_pred) - np.mean(simulated_y_true))
            if divergence_score > 0.1:
                divergent_itemsets_data.append({
                    "itemset": itemset_features,
                    "divergence_score": divergence_score,
                    "support": len(condition_indices) / num_samples,
                    "predicted_outcome_mean": np.mean(simulated_y_pred),
                    "actual_outcome_mean": np.mean(simulated_y_true),
                    "indices": condition_indices.tolist()
                })
        
        divergent_itemsets_data.sort(key=lambda x: x['divergence_score'], reverse=True)
        return divergent_itemsets_data

    def find_corrective_items(self, X, y_true, y_pred, itemset_indices, feature_names):
        itemset_X = X.iloc[itemset_indices]
        itemset_y_true = y_true[itemset_indices]
        itemset_y_pred = y_pred[itemset_indices]

        corrective_suggestions = []
        np.random.seed(43)
        
        if len(feature_names) > 0:
            random_feature = np.random.choice(feature_names)
            
            if np.random.rand() > 0.5:
                correction_value = f"Set {random_feature} to {np.random.randint(0, 2)}"
            else:
                correction_value = f"Change {random_feature} significantly"

            corrective_suggestions.append({
                "feature": random_feature,
                "suggestion": correction_value,
                "potential_impact": np.random.uniform(0.1, 0.5)
            })
        
        lattice_nodes = [
            {"id": "base_itemset", "label": f"Original ({len(itemset_indices)} samples)"},
            {"id": "corrected_a", "label": "Add " + feature_names[0] if len(feature_names)>0 else "Add Feature A"},
            {"id": "corrected_b", "label": "Remove " + feature_names[1] if len(feature_names)>1 else "Remove Feature B"}
        ]
        lattice_edges = [
            {"source": "base_itemset", "target": "corrected_a"},
            {"source": "base_itemset", "target": "corrected_b"}
        ]

        return corrective_suggestions, {"nodes": lattice_nodes, "edges": lattice_edges}


# --- main.py (FastAPI Backend) ---
app = FastAPI()

stored_data: Optional[pd.DataFrame] = None
stored_model: Any = None
stored_target_column: Optional[str] = None
stored_feature_names: Optional[List[str]] = None

class AnalysisRequest(BaseModel):
    target_column: str
    sensitive_attributes: Optional[List[str]] = None
    min_support: float = 0.01
    max_itemset_size: int = 3

class ItemsetShapRequest(BaseModel):
    itemset_indices: List[int]

@app.post("/upload_data_model/")
async def upload_data_model(
    data_file: UploadFile = File(...),
    model_file: UploadFile = File(...)
):
    global stored_data, stored_model, stored_target_column, stored_feature_names

    try:
        data_content = await data_file.read()
        stored_data = pd.read_csv(io.StringIO(data_content.decode('utf-8')))

        model_content = await model_file.read()
        stored_model = joblib.load(io.BytesIO(model_content))

        stored_target_column = None
        stored_feature_names = None

        return {"message": "Data and model uploaded successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading files: {e}")

@app.post("/run_analysis/")
async def run_analysis(request: AnalysisRequest):
    global stored_data, stored_model, stored_target_column, stored_feature_names

    if stored_data is None or stored_model is None:
        raise HTTPException(status_code=400, detail="Please upload data and model first.")

    stored_target_column = request.target_column

    if stored_target_column not in stored_data.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{stored_target_column}' not found in data.")

    X = stored_data.drop(columns=[stored_target_column])
    y_true = stored_data[stored_target_column].values
    
    stored_feature_names = X.columns.tolist()

    try:
        y_pred_proba = stored_model.predict_proba(X)
        if y_pred_proba.shape[1] > 1:
            y_pred = y_pred_proba[:, 1]
        else:
            y_pred = stored_model.predict_proba(X).flatten()
    except AttributeError:
        y_pred = stored_model.predict(X)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during model prediction: {e}")


    div_explorer = DivExplorer()
    divergent_itemsets = div_explorer.find_divergent_itemsets(
        X, y_true, y_pred, stored_feature_names,
        min_support=request.min_support,
        max_items=request.max_itemset_size
    )

    global_influence = {
        "feature_A": np.random.uniform(0,1),
        "feature_B": np.random.uniform(0,1),
        "feature_C": np.random.uniform(0,1),
    }

    return {
        "divergent_itemsets": divergent_itemsets,
        "global_influence": global_influence,
        "feature_names": stored_feature_names
    }

@app.post("/get_shap_values/")
async def get_shap_values(request: ItemsetShapRequest):
    global stored_data, stored_model, stored_target_column, stored_feature_names

    if stored_data is None or stored_model is None or stored_target_column is None or stored_feature_names is None:
        raise HTTPException(status_code=400, detail="Analysis not run or data/model not available.")

    if not request.itemset_indices:
        return {"shap_values": [], "feature_names": stored_feature_names}

    try:
        X_full_for_shap = stored_data.drop(columns=[stored_target_column])
        
        itemset_X = X_full_for_shap.iloc[request.itemset_indices]

        itemset_X = itemset_X[stored_feature_names]

        explainer = shap.KernelExplainer(stored_model.predict_proba, X_full_for_shap.sample(min(100, len(X_full_for_shap)), random_state=42))
        shap_values_raw = explainer.shap_values(itemset_X)

        if isinstance(shap_values_raw, list) and len(shap_values_raw) > 1:
            shap_values = shap_values_raw[1]
        else:
            shap_values = shap_values_raw

        avg_shap_values = np.mean(shap_values, axis=0)

        return {
            "shap_values": avg_shap_values.tolist(),
            "feature_names": stored_feature_names
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating SHAP values: {e}")

@app.post("/get_lattice_data/")
async def get_lattice_data(request: ItemsetShapRequest):
    global stored_data, stored_model, stored_target_column, stored_feature_names

    if stored_data is None or stored_model is None or stored_target_column is None or stored_feature_names is None:
        raise HTTPException(status_code=400, detail="Analysis not run or data/model not available.")

    if not request.itemset_indices:
        return {"corrective_suggestions": [], "lattice_graph": {"nodes": [], "edges": []}}

    try:
        X_full = stored_data.drop(columns=[stored_target_column])
        y_true_full = stored_data[stored_target_column].values
        
        y_pred_proba = stored_model.predict_proba(X_full)
        if y_pred_proba.shape[1] > 1:
            y_pred_full = y_pred_proba[:, 1]
        else:
            y_pred_full = stored_model.predict_proba(X_full).flatten()

        div_explorer = DivExplorer()
        corrective_suggestions, lattice_graph = div_explorer.find_corrective_items(
            X_full, y_true_full, y_pred_full, request.itemset_indices, stored_feature_names
        )

        return {
            "corrective_suggestions": corrective_suggestions,
            "lattice_graph": lattice_graph
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting lattice data: {e}")


# --- streamlit_app.py (Streamlit Frontend) ---

BACKEND_URL = "http://localhost:8000"

st.set_page_config(layout="wide", page_title="DivExplorer Bias Explorer")

st.title("Healthcare Treatment Outcome Bias Explorer")

st.sidebar.header("Upload Data & Model")

uploaded_data_file = st.sidebar.file_uploader("Upload Patient Data (CSV)", type=["csv"])
uploaded_model_file = st.sidebar.file_uploader("Upload Trained Model (PKL)", type=["pkl"])

target_column = st.sidebar.text_input("Target Column Name", "treatment_success")
sensitive_attributes_input = st.sidebar.text_input("Sensitive Attributes (comma-separated)", "gender,race")
sensitive_attributes = [attr.strip() for attr in sensitive_attributes_input.split(",")] if sensitive_attributes_input else []

min_support = st.sidebar.slider("Min Support for Itemsets", 0.001, 0.1, 0.01)
max_itemset_size = st.sidebar.slider("Max Itemset Size", 1, 5, 3)

if uploaded_data_file and uploaded_model_file:
    st.sidebar.success("Data and Model files loaded. Ready for analysis.")
    if st.sidebar.button("Upload to Backend"):
        files = {
            "data_file": (uploaded_data_file.name, uploaded_data_file.getvalue(), "text/csv"),
            "model_file": (uploaded_model_file.name, uploaded_model_file.getvalue(), "application/octet-stream")
        }
        try:
            response = requests.post(f"{BACKEND_URL}/upload_data_model/", files=files)
            if response.status_code == 200:
                st.sidebar.success(f"Backend: {response.json()['message']}")
            else:
                st.sidebar.error(f"Backend Error: {response.status_code} - {response.json().get('detail', 'Unknown error')}")
        except requests.exceptions.ConnectionError:
            st.sidebar.error("Could not connect to backend. Is it running?")
        except Exception as e:
            st.sidebar.error(f"An unexpected error occurred during backend upload: {e}")
else:
    st.sidebar.info("Please upload both a CSV dataset and a PKL model file.")

st.sidebar.markdown(" preconceived notions ")
if st.sidebar.button("Run DivExplorer Analysis"):
    if uploaded_data_file and uploaded_model_file:
        with st.spinner("Running DivExplorer analysis..."):
            try:
                analysis_payload = {
                    "target_column": target_column,
                    "sensitive_attributes": sensitive_attributes,
                    "min_support": min_support,
                    "max_itemset_size": max_itemset_size
                }
                response = requests.post(f"{BACKEND_URL}/run_analysis/", json=analysis_payload)

                if response.status_code == 200:
                    analysis_results = response.json()
                    st.session_state.divergent_itemsets = analysis_results.get("divergent_itemsets", [])
                    st.session_state.global_influence = analysis_results.get("global_influence", {})
                    st.session_state.feature_names = analysis_results.get("feature_names", [])
                    st.success("Analysis complete!")
                else:
                    st.error(f"Error running analysis: {response.status_code} - {response.json().get('detail', 'Unknown error')}")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to backend. Is it running?")
            except Exception as e:
                st.error(f"An unexpected error occurred during analysis: {e}")
    else:
        st.warning("Please upload data and model files first.")

if "divergent_itemsets" in st.session_state and st.session_state.divergent_itemsets:
    st.header("Divergent Itemsets")
    itemsets_df = pd.DataFrame(st.session_state.divergent_itemsets)
    itemsets_df["itemset_str"] = itemsets_df["itemset"].apply(lambda x: ", ".join(x))
    
    selected_itemset_row = st.radio(
        "Select a divergent itemset for detailed exploration:",
        options=itemsets_df.index,
        format_func=lambda x: itemsets_df.loc[x, "itemset_str"] + f" (Div: {itemsets_df.loc[x, 'divergence_score']:.3f}, Sup: {itemsets_df.loc[x, 'support']:.3f})"
    )

    selected_itemset_data = itemsets_df.loc[selected_itemset_row]
    st.session_state.selected_itemset_indices = selected_itemset_data["indices"]

    st.subheader(f"Selected Itemset: {selected_itemset_data['itemset_str']}")
    st.write(f"Divergence Score: {selected_itemset_data['divergence_score']:.4f}")
    st.write(f"Support (proportion of data): {selected_itemset_data['support']:.4f}")
    st.write(f"Predicted Outcome Mean: {selected_itemset_data['predicted_outcome_mean']:.4f}")
    st.write(f"Actual Outcome Mean: {selected_itemset_data['actual_outcome_mean']:.4f}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Local Item Contribution (SHAP Values)")
        if st.button("Calculate SHAP for Selected Itemset"):
            with st.spinner("Calculating SHAP values..."):
                try:
                    shap_payload = {"itemset_indices": st.session_state.selected_itemset_indices}
                    response = requests.post(f"{BACKEND_URL}/get_shap_values/", json=shap_payload)
                    if response.status_code == 200:
                        shap_results = response.json()
                        shap_values = shap_results["shap_values"]
                        feature_names = shap_results["feature_names"]

                        shap_df = pd.DataFrame({"Feature": feature_names, "SHAP Value": shap_values})
                        shap_df = shap_df.sort_values(by="SHAP Value", ascending=False)
                        
                        fig_shap = px.bar(
                            shap_df,
                            x="SHAP Value",
                            y="Feature",
                            orientation="h",
                            title=f"SHAP Values for Itemset: {selected_itemset_data['itemset_str']}",
                            height=600
                        )
                        st.plotly_chart(fig_shap, use_container_width=True)
                    else:
                        st.error(f"Error getting SHAP values: {response.status_code} - {response.json().get('detail', 'Unknown error')}")
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to backend. Is it running?")
                except Exception as e:
                    st.error(f"An unexpected error occurred during SHAP calculation: {e}")

    with col2:
        st.subheader("Lattice Visualization & Corrective Items")
        if st.button("Generate Lattice and Corrective Items"):
            with st.spinner("Generating lattice data..."):
                try:
                    lattice_payload = {"itemset_indices": st.session_state.selected_itemset_indices}
                    response = requests.post(f"{BACKEND_URL}/get_lattice_data/", json=lattice_payload)

                    if response.status_code == 200:
                        lattice_results = response.json()
                        corrective_suggestions = lattice_results.get("corrective_suggestions", [])
                        lattice_graph_data = lattice_results.get("lattice_graph", {"nodes": [], "edges": []})

                        if corrective_suggestions:
                            st.write("**Corrective Suggestions:**")
                            for suggestion in corrective_suggestions:
                                st.markdown(f"- **{suggestion['feature']}**: {suggestion['suggestion']} (Potential Impact: {suggestion['potential_impact']:.2f})")
                        else:
                            st.info("No specific corrective suggestions found for this itemset (mock).")

                        st.write("**Lattice Graph:**")
                        if lattice_graph_data["nodes"] and lattice_graph_data["edges"]:
                            G = nx.Graph()
                            for node in lattice_graph_data["nodes"]:
                                G.add_node(node["id"], label=node["label"])
                            for edge in lattice_graph_data["edges"]:
                                G.add_edge(edge["source"], edge["target"])

                            fig, ax = plt.subplots(figsize=(8, 6))
                            pos = nx.spring_layout(G)
                            nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=3000, ax=ax)
                            nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray')
                            nx.draw_networkx_labels(G, pos, labels={node_id: G.nodes[node_id]['label'] for node_id in G.nodes()}, font_size=8, ax=ax)
                            ax.set_title("Itemset Relationship Lattice")
                            st.pyplot(fig)
                        else:
                            st.info("Lattice graph data not available (mock).")

                    else:
                        st.error(f"Error getting lattice data: {response.status_code} - {response.json().get('detail', 'Unknown error')}")
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to backend. Is it running?")
                except Exception as e:
                    st.error(f"An unexpected error occurred during lattice generation: {e}")

    st.header("Global Item Influence")
    if "global_influence" in st.session_state and st.session_state.global_influence:
        global_influence_df = pd.DataFrame(
            {"Feature": list(st.session_state.global_influence.keys()),
             "Influence Score": list(st.session_state.global_influence.values())}
        )
        fig_global = px.bar(
            global_influence_df.sort_values(by="Influence Score", ascending=False),
            x="Influence Score",
            y="Feature",
            orientation="h",
            title="Global Feature Influence on Divergence"
        )
        st.plotly_chart(fig_global, use_container_width=True)
    else:
        st.info("Run analysis to see global item influence.")


