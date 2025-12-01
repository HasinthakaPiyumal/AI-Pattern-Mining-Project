import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- Mock Data Generation ---
def generate_mock_healthcare_data(num_samples=1000):
    np.random.seed(42)
    data = {
        'patient_id': range(num_samples),
        'age': np.random.randint(20, 90, num_samples),
        'gender': np.random.choice(['Male', 'Female', 'Other'], num_samples, p=[0.48, 0.50, 0.02]),
        'diagnosis_group': np.random.choice(['Cardio', 'Onco', 'Neuro', 'Metabolic'], num_samples, p=[0.3, 0.25, 0.25, 0.2]),
        'comorbidity_count': np.random.randint(0, 5, num_samples),
        'treatment_plan_model': np.random.choice(['Plan A', 'Plan B', 'Plan C', 'Plan D'], num_samples),
        'actual_treatment_outcome': np.random.choice(['Good', 'Poor', 'Neutral'], num_samples, p=[0.6, 0.2, 0.2]),
        'model_predicted_outcome': np.random.choice(['Good', 'Poor', 'Neutral'], num_samples, p=[0.55, 0.25, 0.2]) # Slightly different from actual
    }
    df = pd.DataFrame(data)
    return df

# --- Mock DivExplorer Algorithm ---
def run_mock_divexplorer(dataframe):
    st.info("Simulating DivExplorer algorithm to find divergent subgroups...")

    divergent_itemsets = []
    
    # Example 1: Older females with poor model prediction vs actual good outcome
    divergent_itemsets.append({
        'itemset_id': 1,
        'features': {'age_group': '60+', 'gender': 'Female', 'model_predicted_outcome': 'Poor'},
        'divergence_score': 0.85,
        'subgroup_size': 50,
        'actual_outcome_distribution': {'Good': 0.7, 'Poor': 0.15, 'Neutral': 0.15}
    })

    # Example 2: Patients with 'Onco' diagnosis, high comorbidities, model predicts good but actual is poor
    divergent_itemsets.append({
        'itemset_id': 2,
        'features': {'diagnosis_group': 'Onco', 'comorbidity_count_group': '3+', 'model_predicted_outcome': 'Good'},
        'divergence_score': 0.78,
        'subgroup_size': 35,
        'actual_outcome_distribution': {'Good': 0.3, 'Poor': 0.5, 'Neutral': 0.2}
    })
    
    # Example 3: Young males with 'Cardio' diagnosis, model predicts poor but actual is good
    divergent_itemsets.append({
        'itemset_id': 3,
        'features': {'age_group': '20-40', 'gender': 'Male', 'diagnosis_group': 'Cardio', 'model_predicted_outcome': 'Poor'},
        'divergence_score': 0.60,
        'subgroup_size': 70,
        'actual_outcome_distribution': {'Good': 0.8, 'Poor': 0.1, 'Neutral': 0.1}
    })

    # Example 4: Patients with 'Neuro' diagnosis, model predicts neutral but actual is poor
    divergent_itemsets.append({
        'itemset_id': 4,
        'features': {'diagnosis_group': 'Neuro', 'model_predicted_outcome': 'Neutral'},
        'divergence_score': 0.72,
        'subgroup_size': 45,
        'actual_outcome_distribution': {'Good': 0.1, 'Poor': 0.6, 'Neutral': 0.3}
    })

    # Convert features dict to a readable string for display
    for itemset in divergent_itemsets:
        itemset['features_str'] = ", ".join([f"{k}: {v}" for k, v in itemset['features'].items()])

    return pd.DataFrame(divergent_itemsets)

# --- Mock Shapley Value Calculation ---
def get_mock_shapley_values(selected_itemset_features_str):
    if "age_group: 60+" in selected_itemset_features_str and "gender: Female" in selected_itemset_features_str:
        return pd.DataFrame({
            'feature': ['age_group', 'gender', 'model_predicted_outcome', 'diagnosis_group', 'comorbidity_count'],
            'contribution': [0.4, 0.3, 0.2, 0.05, 0.05]
        })
    elif "diagnosis_group: Onco" in selected_itemset_features_str and "comorbidity_count_group: 3+" in selected_itemset_features_str:
        return pd.DataFrame({
            'feature': ['diagnosis_group', 'comorbidity_count_group', 'model_predicted_outcome', 'age_group', 'gender'],
            'contribution': [0.45, 0.35, 0.1, 0.05, 0.05]
        })
    elif "age_group: 20-40" in selected_itemset_features_str and "gender: Male" in selected_itemset_features_str:
        return pd.DataFrame({
            'feature': ['age_group', 'gender', 'diagnosis_group', 'model_predicted_outcome', 'comorbidity_count'],
            'contribution': [0.4, 0.25, 0.2, 0.1, 0.05]
        })
    elif "diagnosis_group: Neuro" in selected_itemset_features_str:
        return pd.DataFrame({
            'feature': ['diagnosis_group', 'model_predicted_outcome', 'comorbidity_count', 'age_group', 'gender'],
            'contribution': [0.5, 0.3, 0.1, 0.05, 0.05]
        })
    else:
        return pd.DataFrame({
            'feature': ['feature_X', 'feature_Y', 'feature_Z'],
            'contribution': [0.5, 0.3, 0.2]
        })


# --- Streamlit Application ---
def main():
    st.set_page_config(layout="wide", page_title="Healthcare Treatment Bias Explorer")

    st.title("🏥 Healthcare Treatment Bias Explorer")
    st.markdown("""
        This interactive system helps healthcare providers and researchers explore potential biases in AI models
        used for treatment plan recommendations. It leverages the DivExplorer concept to identify and
        visualize patient subgroups where model predictions diverge from actual outcomes or expected guidelines.
    """)

    st.sidebar.header("Configuration")
    num_samples = st.sidebar.slider("Number of mock patient samples", 100, 5000, 1000)
    
    st.sidebar.subheader("Data Loading")
    if st.sidebar.button("Generate Mock Healthcare Data"):
        st.session_state['data'] = generate_mock_healthcare_data(num_samples)
        st.sidebar.success(f"Generated {num_samples} mock patient records.")

    if 'data' not in st.session_state:
        st.info("Please generate mock data from the sidebar to start.")
        return

    st.subheader("Raw Healthcare Data (Sample)")
    st.dataframe(st.session_state['data'].head())
    st.write(f"Total records: {len(st.session_state['data'])}")

    st.sidebar.subheader("DivExplorer Analysis")
    if st.sidebar.button("Run DivExplorer Analysis"):
        with st.spinner("Running DivExplorer to identify divergent subgroups..."):
            st.session_state['divergent_itemsets'] = run_mock_divexplorer(st.session_state['data'])
            st.sidebar.success("DivExplorer analysis complete!")

    if 'divergent_itemsets' in st.session_state and not st.session_state['divergent_itemsets'].empty:
        st.markdown("---")
        st.header("🔍 Divergent Patient Subgroups (Identified by DivExplorer)")
        st.write("Below is a table of patient subgroups where the model's predictions show significant divergence.")

        divergent_df_display = st.session_state['divergent_itemsets'][['itemset_id', 'features_str', 'divergence_score', 'subgroup_size']]
        
        st.dataframe(divergent_df_display.sort_values(by='divergence_score', ascending=False), 
                     use_container_width=True, 
                     hide_index=True)

        selected_itemset_id = st.selectbox(
            "Select a Divergent Subgroup for detailed analysis:",
            options=[None] + divergent_df_display['itemset_id'].tolist(),
            format_func=lambda x: f"Subgroup ID: {x}" if x else "Select a subgroup"
        )

        if selected_itemset_id is not None:
            selected_itemset_row = st.session_state['divergent_itemsets'][st.session_state['divergent_itemsets']['itemset_id'] == selected_itemset_id].iloc[0]
            
            st.markdown(f"### Details for Subgroup ID: {selected_itemset_id}")
            st.write(f"**Features:** `{selected_itemset_row['features_str']}`")
            st.write(f"**Divergence Score:** `{selected_itemset_row['divergence_score']:.2f}` (Higher indicates more divergence)")
            st.write(f"**Subgroup Size:** `{selected_itemset_row['subgroup_size']}` patients")
            
            st.subheader("Local Feature Contributions (Shapley Values) for this Subgroup")
            st.write("These values indicate how much each feature contributes to the model's divergent behavior within this specific subgroup.")
            
            shapley_df = get_mock_shapley_values(selected_itemset_row['features_str'])
            fig = px.bar(shapley_df.sort_values(by='contribution', ascending=False), 
                         x='feature', y='contribution', 
                         title=f"Feature Contributions for Subgroup {selected_itemset_id}",
                         labels={'contribution': 'Contribution (Shapley Value)', 'feature': 'Feature'},
                         height=400)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Actual Outcome Distribution in this Subgroup")
            outcome_dist_df = pd.DataFrame(selected_itemset_row['actual_outcome_distribution'].items(), columns=['Outcome', 'Proportion'])
            fig_outcome = px.pie(outcome_dist_df, values='Proportion', names='Outcome', 
                                 title=f"Actual Treatment Outcomes for Subgroup {selected_itemset_id}")
            st.plotly_chart(fig_outcome, use_container_width=True)

            st.subheader("Additional Visualizations (Placeholders)")
            st.info("In a full implementation, this section would include a lattice visualization for exploring subset relationships, "
                    "corrective items, and global item influence visualizations.")
            st.write("---")
            st.text("Lattice Visualization: [Interactive graph showing feature subsets and supersets]")
            st.text("Global Item Influence: [Bar chart/heatmap showing overall impact of features on divergence]")
            st.text("Search and Drilldown: [Functionality to filter and explore raw data belonging to selected subgroups]")

    elif 'divergent_itemsets' in st.session_state and st.session_state['divergent_itemsets'].empty:
        st.warning("No divergent subgroups found by DivExplorer (mock run). Try adjusting parameters.")

if __name__ == "__main__":
    main()