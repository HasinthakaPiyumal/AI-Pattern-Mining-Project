import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import shap
import plotly.express as px
import networkx as nx
import matplotlib.pyplot as plt


def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    data = pd.DataFrame({
        'Age': np.random.randint(20, 80, n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'BMI': np.random.uniform(18, 35, n_samples),
        'Glucose': np.random.uniform(70, 200, n_samples),
        'BloodPressure': np.random.uniform(90, 180, n_samples),
        'Cholesterol': np.random.uniform(150, 300, n_samples),
        'Smoking': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'Alcohol': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'Diagnosis_A': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        'Diagnosis_B': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
        'Medication_X': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'Medication_Y': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
    })

    # Generate a target variable with some bias
    data['Target'] = 0
    data.loc[(data['Age'] > 60) & (data['Glucose'] > 150) & (data['Gender'] == 'Female'), 'Target'] = 1
    data.loc[(data['BMI'] > 30) & (data['BloodPressure'] > 140), 'Target'] = 1
    data['Target'] = data['Target'].apply(lambda x: np.random.choice([x, 1-x], p=[0.8, 0.2])) # Add some noise

    return data

def train_mock_model(data, target_column='Target'):
    features = [col for col in data.columns if col != target_column]
    X = pd.get_dummies(data[features], drop_first=True)
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    #st.write(f"Mock Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    return model, X

def run_divexplorer(data, model, features_df, target_column='Target', divergence_threshold=0.1, min_subgroup_size=20):
    predictions = model.predict_proba(features_df)[:, 1]
    overall_avg_prediction = np.mean(predictions)

    divergent_subgroups = []
    
    # Mock divergence detection based on simple rules for demonstration
    # Rule 1: Elderly females with high glucose
    subgroup_1_indices = features_df[(data['Age'] > 60) & (features_df['Gender_Male'] == 0) & (data['Glucose'] > 150)].index
    if len(subgroup_1_indices) >= min_subgroup_size:
        subgroup_1_predictions = predictions[subgroup_1_indices]
        subgroup_1_avg_prediction = np.mean(subgroup_1_predictions)
        divergence_1 = abs(subgroup_1_avg_prediction - overall_avg_prediction)
        if divergence_1 > divergence_threshold:
            divergent_subgroups.append({
                'definition': 'Age > 60 & Gender == Female & Glucose > 150',
                'divergence_score': divergence_1,
                'subgroup_size': len(subgroup_1_indices),
                'model_outcome': subgroup_1_avg_prediction,
                'indices': subgroup_1_indices.tolist()
            })
    
    # Rule 2: High BMI and high blood pressure
    subgroup_2_indices = features_df[(data['BMI'] > 30) & (data['BloodPressure'] > 140)].index
    if len(subgroup_2_indices) >= min_subgroup_size:
        subgroup_2_predictions = predictions[subgroup_2_indices]
        subgroup_2_avg_prediction = np.mean(subgroup_2_predictions)
        divergence_2 = abs(subgroup_2_avg_prediction - overall_avg_prediction)
        if divergence_2 > divergence_threshold:
            divergent_subgroups.append({
                'definition': 'BMI > 30 & BloodPressure > 140',
                'divergence_score': divergence_2,
                'subgroup_size': len(subgroup_2_indices),
                'model_outcome': subgroup_2_avg_prediction,
                'indices': subgroup_2_indices.tolist()
            })

    # Rule 3: Patients on Medication_Y with Diagnosis_A
    subgroup_3_indices = features_df[(features_df['Medication_Y'] == 1) & (features_df['Diagnosis_A'] == 1)].index
    if len(subgroup_3_indices) >= min_subgroup_size:
        subgroup_3_predictions = predictions[subgroup_3_indices]
        subgroup_3_avg_prediction = np.mean(subgroup_3_predictions)
        divergence_3 = abs(subgroup_3_avg_prediction - overall_avg_prediction)
        if divergence_3 > divergence_threshold:
            divergent_subgroups.append({
                'definition': 'Medication_Y == 1 & Diagnosis_A == 1',
                'divergence_score': divergence_3,
                'subgroup_size': len(subgroup_3_indices),
                'model_outcome': subgroup_3_avg_prediction,
                'indices': subgroup_3_indices.tolist()
            })

    return divergent_subgroups

def calculate_shap_values(model, data_processed, subgroup_indices, feature_names):
    if len(subgroup_indices) == 0:
        return pd.Series()
    
    # For black-box models, KernelExplainer is suitable
    explainer = shap.KernelExplainer(model.predict_proba, data_processed.sample(100, random_state=42))
    shap_values_raw = explainer.shap_values(data_processed.loc[subgroup_indices])
    
    # shap_values_raw will be a list of arrays for multi-output models (e.g., predict_proba)
    # We typically care about the SHAP values for the positive class (index 1)
    shap_values = shap_values_raw[1]

    # Calculate mean absolute SHAP values for the subgroup
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    return pd.Series(mean_abs_shap, index=feature_names)


def create_subgroup_lattice(divergent_subgroups):
    G = nx.DiGraph()
    
    if not divergent_subgroups:
        return G

    # Add nodes for each divergent subgroup
    for i, subgroup in enumerate(divergent_subgroups):
        G.add_node(i, label=f"Subgroup {i+1}\n({subgroup['subgroup_size']} patients)", definition=subgroup['definition'])

    # Simplified: Add some random edges for demonstration if enough subgroups exist
    # In a real scenario, this would involve parsing definitions and checking subset relationships
    if len(divergent_subgroups) > 1:
        num_edges = min(len(divergent_subgroups) * (len(divergent_subgroups) - 1) // 2, 5)
        for _ in range(num_edges):
            u, v = np.random.choice(range(len(divergent_subgroups)), 2, replace=False)
            # Ensure u -> v implies v is a 'subset' or 'more specific' conceptually
            if divergent_subgroups[u]['subgroup_size'] > divergent_subgroups[v]['subgroup_size']:
                 G.add_edge(u, v, relationship="more specific")
            else:
                 G.add_edge(v, u, relationship="more specific")

    return G

st.set_page_config(layout="wide", page_title="MediBias Investigator")

st.title("🩺 MediBias Investigator")
st.markdown("Explore and analyze divergent behaviors of clinical prediction models.")

# Initialize session state variables
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'model' not in st.session_state:
    st.session_state['model'] = None
if 'features_df' not in st.session_state:
    st.session_state['features_df'] = None
if 'feature_names' not in st.session_state:
    st.session_state['feature_names'] = None
if 'divergent_subgroups' not in st.session_state:
    st.session_state['divergent_subgroups'] = []
if 'selected_subgroup_index' not in st.session_state:
    st.session_state['selected_subgroup_index'] = None

with st.sidebar:
    st.header("Configuration")

    st.subheader("Data & Model Loading")
    data_file = st.file_uploader("Upload patient data (CSV)", type=["csv"])
    model_file = st.file_uploader("Upload trained model (pickle)", type=["pkl"])

    if data_file is not None:
        st.session_state['data'] = pd.read_csv(data_file)
        st.success("Data loaded successfully!")
    elif st.button("Generate Synthetic Data"):
        st.session_state['data'] = generate_synthetic_data()
        st.success("Synthetic data generated!")

    if st.session_state['data'] is not None and st.session_state['model'] is None:
        if st.button("Train Mock Model on Synthetic Data"):
            if 'data' in st.session_state and st.session_state['data'] is not None:
                model, features_df = train_mock_model(st.session_state['data'])
                st.session_state['model'] = model
                st.session_state['features_df'] = features_df
                st.session_state['feature_names'] = features_df.columns.tolist()
                st.success("Mock model trained!")
            else:
                st.warning("Please load or generate data first.")
    
    if model_file is not None:
        try:
            st.session_state['model'] = pickle.load(model_file)
            st.success("Model loaded successfully!")
            # If model is loaded, assume features_df needs to be created from data
            if st.session_state['data'] is not None:
                # Dummy X to get feature names and structure for the model
                dummy_features = [col for col in st.session_state['data'].columns if col != 'Target']
                st.session_state['features_df'] = pd.get_dummies(st.session_state['data'][dummy_features], drop_first=True)
                st.session_state['feature_names'] = st.session_state['features_df'].columns.tolist()
            else:
                st.warning("Please upload data as well to use the loaded model.")
        except Exception as e:
            st.error(f"Error loading model: {e}")

    st.subheader("DivExplorer Parameters")
    divergence_threshold = st.slider("Divergence Threshold", 0.01, 0.5, 0.1, 0.01)
    min_subgroup_size = st.slider("Minimum Subgroup Size", 5, 100, 20, 5)

    if st.button("Run DivExplorer Analysis"):
        if st.session_state['data'] is not None and st.session_state['model'] is not None and st.session_state['features_df'] is not None:
            with st.spinner("Running DivExplorer..."):
                st.session_state['divergent_subgroups'] = run_divexplorer(
                    st.session_state['data'],
                    st.session_state['model'],
                    st.session_state['features_df'],
                    divergence_threshold=divergence_threshold,
                    min_subgroup_size=min_subgroup_size
                )
            if st.session_state['divergent_subgroups']:
                st.success(f"Found {len(st.session_state['divergent_subgroups'])} divergent subgroups!")
            else:
                st.info("No divergent subgroups found with current parameters.")
        else:
            st.warning("Please load data and a model (or generate synthetic data and train mock model) first.")


tabs = st.tabs(["Divergent Subgroups", "Subgroup Details", "Global Insights", "Lattice View"])

with tabs[0]: # Divergent Subgroups
    st.header("Divergent Patient Subgroups")
    if st.session_state['divergent_subgroups']:
        subgroup_display_df = pd.DataFrame(st.session_state['divergent_subgroups'])
        subgroup_display_df = subgroup_display_df.drop(columns=['indices'])
        
        st.dataframe(subgroup_display_df, use_container_width=True)

        st.subheader("Select a Subgroup for Detailed Analysis")
        definitions = [s['definition'] for s in st.session_state['divergent_subgroups']]
        selected_definition = st.selectbox("Choose a subgroup:", definitions, index=0 if definitions else None)
        
        if selected_definition:
            st.session_state['selected_subgroup_index'] = definitions.index(selected_definition)
        else:
            st.session_state['selected_subgroup_index'] = None
    else:
        st.info("No divergent subgroups to display. Run DivExplorer analysis in the sidebar.")

with tabs[1]: # Subgroup Details
    st.header("Selected Subgroup Details")
    if st.session_state['selected_subgroup_index'] is not None and st.session_state['divergent_subgroups']:
        selected_subgroup = st.session_state['divergent_subgroups'][st.session_state['selected_subgroup_index']]
        st.subheader(f"Subgroup Definition: {selected_subgroup['definition']}")
        st.write(f"Divergence Score: {selected_subgroup['divergence_score']:.3f}")
        st.write(f"Subgroup Size: {selected_subgroup['subgroup_size']}")
        st.write(f"Average Model Outcome: {selected_subgroup['model_outcome']:.3f}")

        st.subheader("Local Feature Contributions (SHAP Values)")
        if st.session_state['model'] is not None and st.session_state['features_df'] is not None and st.session_state['feature_names'] is not None:
            with st.spinner("Calculating SHAP values..."):
                shap_values_series = calculate_shap_values(
                    st.session_state['model'],
                    st.session_state['features_df'],
                    selected_subgroup['indices'],
                    st.session_state['feature_names']
                )
            if not shap_values_series.empty:
                fig = px.bar(shap_values_series.sort_values(ascending=True), 
                             orientation='h', 
                             title=f"Mean Absolute SHAP Values for '{selected_subgroup['definition']}'")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Could not calculate SHAP values for this subgroup.")
        else:
            st.warning("Model and features data required for SHAP calculations.")
    else:
        st.info("Please select a divergent subgroup from the 'Divergent Subgroups' tab.")

with tabs[2]: # Global Insights
    st.header("Global Feature Influence")
    if st.session_state['divergent_subgroups'] and st.session_state['model'] is not None and st.session_state['features_df'] is not None and st.session_state['feature_names'] is not None:
        all_shap_values = []
        for subgroup in st.session_state['divergent_subgroups']:
            if subgroup['indices']:
                shap_vals_sub = calculate_shap_values(
                    st.session_state['model'],
                    st.session_state['features_df'],
                    subgroup['indices'],
                    st.session_state['feature_names']
                )
                if not shap_vals_sub.empty:
                    all_shap_values.append(shap_vals_sub)
        
        if all_shap_values:
            global_feature_influence = pd.concat(all_shap_values, axis=1).mean(axis=1).sort_values(ascending=False)
            fig_global = px.bar(global_feature_influence, 
                                title="Average Feature Influence Across Divergent Subgroups (Mean Absolute SHAP)")
            st.plotly_chart(fig_global, use_container_width=True)
        else:
            st.info("No SHAP values computed for any divergent subgroup yet.")
    else:
        st.info("Run DivExplorer and ensure data/model are loaded to see global insights.")

with tabs[3]: # Lattice View
    st.header("Subgroup Relationship Lattice")
    if st.session_state['divergent_subgroups']:
        G = create_subgroup_lattice(st.session_state['divergent_subgroups'])
        if G.nodes:
            pos = nx.spring_layout(G)
            fig_lattice, ax = plt.subplots(figsize=(10, 7))
            
            node_labels = {node: G.nodes[node]['label'] for node in G.nodes()}
            nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=3000, ax=ax)
            nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, ax=ax)
            nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, ax=ax)
            
            # Optional: Highlight selected subgroup
            if st.session_state['selected_subgroup_index'] is not None and st.session_state['selected_subgroup_index'] in G.nodes():
                nx.draw_networkx_nodes(G, pos, nodelist=[st.session_state['selected_subgroup_index']], node_color='lightcoral', node_size=3500, ax=ax)
            
            ax.set_title("Conceptual Lattice of Divergent Subgroups")
            st.pyplot(fig_lattice)
            
            st.markdown("**Note:** This is a simplified, conceptual lattice visualization. A full implementation would involve robust analysis of subgroup definitions for accurate subset/superset relationships.")
            
            if st.session_state['selected_subgroup_index'] is not None:
                st.subheader("Corrective Items (Conceptual)")
                st.info(f"For subgroup '{st.session_state['divergent_subgroups'][st.session_state['selected_subgroup_index']]['definition']}', consider how adjusting 'Medication_Y' or 'Diagnosis_A' might influence its divergence. (This is a placeholder for a more complex analysis.)")
        else:
            st.info("No subgroups to display in the lattice.")
    else:
        st.info("Run DivExplorer analysis to generate subgroups for the lattice view.")