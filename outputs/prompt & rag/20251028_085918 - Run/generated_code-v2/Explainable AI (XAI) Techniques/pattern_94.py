import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- Backend: Data Generator Module ---

def generate_synthetic_medical_data(num_patients=1000):
    np.random.seed(42)
    data = {
        "patient_id": np.arange(num_patients),
        "age": np.random.randint(20, 80, num_patients),
        "gender": np.random.choice(["Male", "Female", "Other"], num_patients, p=[0.48, 0.50, 0.02]),
        "pre_existing_condition": np.random.choice(["None", "Hypertension", "Diabetes", "Asthma", "Obesity"], num_patients, p=[0.4, 0.2, 0.15, 0.15, 0.1]),
        "symptom_severity": np.random.uniform(1, 10, num_patients),
        "model_prediction": np.random.randint(0, 2, num_patients), # 0: No disease, 1: Disease
        "ground_truth": np.random.randint(0, 2, num_patients)
    }
    df = pd.DataFrame(data)

    # Introduce some synthetic bias for demonstration
    # Example: Model performs worse for older females with Hypertension
    biased_group_mask = (df["gender"] == "Female") & (df["age"] > 60) & (df["pre_existing_condition"] == "Hypertension")
    # Increase misclassification for this group
    df.loc[biased_group_mask, "model_prediction"] = np.where(
        df.loc[biased_group_mask, "ground_truth"] == 1, 0, 1 # Flip prediction for some for misclassification
    ) 
    # Ensure some misclassification for others too, but less pronounced
    df.loc[~biased_group_mask & (np.random.rand(len(df[~biased_group_mask])) < 0.1), "model_prediction"] = np.where(
        df.loc[~biased_group_mask & (np.random.rand(len(df[~biased_group_mask])) < 0.1), "ground_truth"] == 1, 0, 1
    )

    return df

# --- Backend: DivExplorer Mock Module ---

def calculate_divergence_score(subgroup_df):
    # Simple divergence: higher misclassification rate
    if subgroup_df.empty: return 0
    misclassified = (subgroup_df["model_prediction"] != subgroup_df["ground_truth"]).sum()
    total = len(subgroup_df)
    return (misclassified / total) if total > 0 else 0

def simulate_divexplorer(df, min_subgroup_size=30, divergence_threshold=0.2):
    divergent_itemsets = []
    features = ["age_group", "gender", "pre_existing_condition"]
    df_processed = df.copy()
    df_processed["age_group"] = pd.cut(df_processed["age"], bins=[0, 30, 50, 70, 100], labels=["young", "adult", "senior", "elderly"])

    # Generate potential itemsets (simplified for demo)
    # Single features
    for feature in features:
        for value in df_processed[feature].unique():
            subgroup = df_processed[df_processed[feature] == value]
            if len(subgroup) >= min_subgroup_size:
                score = calculate_divergence_score(subgroup)
                if score > divergence_threshold:
                    divergent_itemsets.append({
                        "itemset_id": f"{feature}_{value}",
                        "description": f"{feature}: {value}",
                        "divergence_score": score,
                        "count": len(subgroup),
                        "defining_features": {feature: value}
                    })
    # Two features combinations (simplified)
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            feat1, feat2 = features[i], features[j]
            for val1 in df_processed[feat1].unique():
                for val2 in df_processed[feat2].unique():
                    subgroup = df_processed[(df_processed[feat1] == val1) & (df_processed[feat2] == val2)]
                    if len(subgroup) >= min_subgroup_size:
                        score = calculate_divergence_score(subgroup)
                        if score > divergence_threshold:
                            divergent_itemsets.append({
                                "itemset_id": f"{feat1}_{val1}_{feat2}_{val2}",
                                "description": f"{feat1}: {val1}, {feat2}: {val2}",
                                "divergence_score": score,
                                "count": len(subgroup),
                                "defining_features": {feat1: val1, feat2: val2}
                            })

    return pd.DataFrame(divergent_itemsets).sort_values(by="divergence_score", ascending=False).reset_index(drop=True)

def calculate_mock_shapley_values(itemset_data, defining_features):
    shapley_values = {}
    total_divergence = itemset_data["divergence_score"]

    # Assign higher "Shapley" values to features that define the itemset
    for feature in defining_features:
        shapley_values[feature] = np.random.uniform(0.3, 0.7) * total_divergence / len(defining_features) # mock contribution

    # Assign smaller, random values to other features
    all_features = ["age_group", "gender", "pre_existing_condition", "symptom_severity"]
    for feature in all_features:
        if feature not in defining_features:
            shapley_values[feature] = np.random.uniform(0.01, 0.1) * total_divergence
    
    # Normalize so they sum up approximately to the divergence score for easy visualization
    if sum(shapley_values.values()) > 0:
        factor = total_divergence / sum(shapley_values.values())
        shapley_values = {k: v * factor for k, v in shapley_values.items()}

    return pd.DataFrame(list(shapley_values.items()), columns=["Feature", "Shapley_Contribution"])

def get_lattice_info(selected_itemset):
    # Simplified lattice information based on the selected itemset
    desc = selected_itemset["description"]
    info = [
        f"**Relationships for {desc}:**",
        f"- This itemset represents a specific subgroup with a high model divergence.",
        f"- Subgroups containing a subset of '{desc}' features might show similar, but potentially lower, divergence.",
        f"- Subgroups that are supersets of '{desc}' might show even higher divergence or new patterns.",
        f"- Consider exploring related subgroups by adding or removing features from '{desc}'."
    ]
    return "\n".join(info)

def get_corrective_actions(selected_itemset):
    desc = selected_itemset["description"]
    features = selected_itemset["defining_features"]
    actions = [
        f"**Potential Corrective Actions for {desc}:**",
        f"- **Data Augmentation:** Collect more diverse data for cases matching {features}.",
        f"- **Feature Engineering:** Re-evaluate or engineer new features that better capture nuances for this subgroup.",
        f"- **Model Retraining:** Retrain the model with re-weighted samples from this subgroup or using different regularization techniques.",
        f"- **Threshold Adjustment:** Adjust classification thresholds specifically for predictions related to this subgroup (if applicable).",
        f"- **Ethical Review:** Conduct a deeper ethical review of the model's impact on this specific demographic/condition."
    ]
    return "\n".join(actions)

def get_global_influence(divergent_df):
    # Mock global influence by summing up 'importance' based on how often features appear in highly divergent groups
    feature_importance = {}
    if divergent_df.empty: return pd.DataFrame()

    for _, row in divergent_df.iterrows():
        for feature in row["defining_features"].keys():
            feature_importance[feature] = feature_importance.get(feature, 0) + row["divergence_score"]

    if not feature_importance: return pd.DataFrame()
    
    df_global_influence = pd.DataFrame(list(feature_importance.items()), columns=["Feature", "Global_Divergence_Influence"])
    return df_global_influence.sort_values(by="Global_Divergence_Influence", ascending=False)


# --- Streamlit Frontend ---

st.set_page_config(layout="wide", page_title="MediBias Explorer")

st.title("🩺 MediBias Explorer: Interactive Bias Detection in Medical AI")
st.markdown(
    "An interactive system to explore and debug divergent behaviors in black-box medical diagnosis models. "
    "Identify potential biases in patient subgroups to improve fairness and model reliability."
)

if "df_patients" not in st.session_state:
    st.session_state.df_patients = None
    st.session_state.df_divergent_itemsets = None

with st.sidebar:
    st.header("Data Generation & Controls")
    num_patients = st.slider("Number of Synthetic Patients", 100, 5000, 1000)
    min_subgroup_size = st.slider("Minimum Subgroup Size for DivExplorer", 10, 200, 30)
    divergence_threshold = st.slider("Divergence Score Threshold", 0.05, 0.5, 0.2, 0.01)

    if st.button("Generate Data & Run DivExplorer Simulation"):
        with st.spinner("Generating synthetic data and simulating DivExplorer..."):
            st.session_state.df_patients = generate_synthetic_medical_data(num_patients)
            st.session_state.df_divergent_itemsets = simulate_divexplorer(
                st.session_state.df_patients, 
                min_subgroup_size=min_subgroup_size, 
                divergence_threshold=divergence_threshold
            )
        st.success("Data generation and DivExplorer simulation complete!")

if st.session_state.df_patients is None:
    st.info("Please generate data and run the DivExplorer simulation using the sidebar controls.")
else:
    st.header("1. Divergent Patient Subgroups")
    st.markdown("The table below shows patient subgroups where the medical diagnosis model exhibits significantly divergent behavior (e.g., higher misclassification rates).")

    if st.session_state.df_divergent_itemsets is not None and not st.session_state.df_divergent_itemsets.empty:
        st.dataframe(st.session_state.df_divergent_itemsets.drop(columns=["defining_features"]), use_container_width=True, height=250)

        # Itemset selection for detail view
        selected_itemset_description = st.selectbox(
            "Select a divergent subgroup for detailed analysis:",
            st.session_state.df_divergent_itemsets["description"].tolist()
        )

        selected_itemset = st.session_state.df_divergent_itemsets[
            st.session_state.df_divergent_itemsets["description"] == selected_itemset_description
        ].iloc[0]

        st.header(f"2. Detailed Analysis for: {selected_itemset_description}")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Local Item Contributions (Mock Shapley Values)")
            st.markdown("Estimates how much each feature contributes to the model's divergent behavior within this specific subgroup.")
            shapley_df = calculate_mock_shapley_values(
                selected_itemset, selected_itemset["defining_features"]
            )
            if not shapley_df.empty:
                fig_shapley = px.bar(shapley_df, x="Shapley_Contribution", y="Feature", orientation="h",
                                     title="Feature Contribution to Divergence",
                                     labels={"Shapley_Contribution": "Contribution Score", "Feature": "Feature"})
                st.plotly_chart(fig_shapley, use_container_width=True)
            else:
                st.info("No Shapley values to display for this itemset.")

        with col2:
            st.subheader("Lattice Visualization (Conceptual)")
            st.markdown("Understanding subset/superset relationships for navigating divergent patterns.")
            st.markdown(get_lattice_info(selected_itemset))

            st.subheader("Corrective Items/Actions")
            st.markdown("Suggested strategies to mitigate the identified bias or divergence.")
            st.markdown(get_corrective_actions(selected_itemset))

        st.header("3. Global Item Influence")
        st.markdown("Overall influence of features across all identified divergent subgroups. Features with higher influence might be critical areas for bias investigation across the entire model.")
        global_influence_df = get_global_influence(st.session_state.df_divergent_itemsets)
        if not global_influence_df.empty:
            fig_global = px.bar(global_influence_df, x="Global_Divergence_Influence", y="Feature", orientation="h",
                                 title="Global Feature Influence on Divergence",
                                 labels={"Global_Divergence_Influence": "Total Divergence Influence", "Feature": "Feature"})
            st.plotly_chart(fig_global, use_container_width=True)
        else:
            st.info("No global influence data to display. Generate data and simulate DivExplorer first.")

    else:
        st.info("No divergent subgroups identified based on current settings or data. Try adjusting parameters in the sidebar.")