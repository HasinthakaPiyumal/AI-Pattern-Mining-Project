import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import networkx as nx
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Healthcare Treatment Outcome Divergence Explorer")

# --- Data Simulation --- 
# In a real application, this would come from a database and a DivExplorer backend

def generate_patient_data(num_patients=1000):
    np.random.seed(42)
    data = {
        "patient_id": range(1, num_patients + 1),
        "age": np.random.randint(20, 80, num_patients),
        "gender": np.random.choice(["Male", "Female"], num_patients, p=[0.5, 0.5]),
        "comorbidity": np.random.choice(["None", "Diabetes", "Hypertension", "Asthma"], num_patients, p=[0.4, 0.3, 0.2, 0.1]),
        "treatment_protocol": np.random.choice(["A", "B", "C"], num_patients, p=[0.4, 0.3, 0.3]),
        "drug_response": np.random.choice(["Good", "Moderate", "Poor"], num_patients, p=[0.6, 0.25, 0.15]),
        "adverse_event": np.random.choice([True, False], num_patients, p=[0.15, 0.85])
    }
    df = pd.DataFrame(data)
    
    # Simulate some outcome divergence based on characteristics
    df["outcome_success"] = np.random.choice([True, False], num_patients, p=[0.7, 0.3]) # Baseline
    
    # Example of divergence: Older males with diabetes on protocol A might have lower success
    df.loc[(df["age"] > 60) & (df["gender"] == "Male") & (df["comorbidity"] == "Diabetes") & (df["treatment_protocol"] == "A"), "outcome_success"] = np.random.choice([True, False], len(df[(df["age"] > 60) & (df["gender"] == "Male") & (df["comorbidity"] == "Diabetes") & (df["treatment_protocol"] == "A")]), p=[0.3, 0.7])
    
    return df

def simulate_divexplorer_output(patient_data):
    # This function simulates the output of a DivExplorer-like algorithm
    # In a real scenario, this would involve complex analysis of the patient_data
    
    divergent_itemsets = [
        {
            "itemset": {"gender": "Male", "age_group": "60+", "comorbidity": "Diabetes"},
            "divergence_score": 0.85,
            "outcome_metric": "Success Rate",
            "baseline_rate": 0.75,
            "itemset_rate": 0.30,
            "size": 55
        },
        {
            "itemset": {"treatment_protocol": "A", "drug_response": "Poor"},
            "divergence_score": 0.70,
            "outcome_metric": "Adverse Event Rate",
            "baseline_rate": 0.15,
            "itemset_rate": 0.45,
            "size": 120
        },
        {
            "itemset": {"age_group": "20-30", "treatment_protocol": "C"},
            "divergence_score": 0.60,
            "outcome_metric": "Success Rate",
            "baseline_rate": 0.70,
            "itemset_rate": 0.88,
            "size": 80
        },
        {
            "itemset": {"gender": "Female", "comorbidity": "Hypertension"},
            "divergence_score": 0.55,
            "outcome_metric": "Adverse Event Rate",
            "baseline_rate": 0.10,
            "itemset_rate": 0.25,
            "size": 90
        }
    ]
    return pd.DataFrame(divergent_itemsets)

def simulate_shapley_values(itemset_features):
    # Simulate Shapley values for features within an itemset
    features = list(itemset_features.keys())
    values = np.random.rand(len(features))
    values = values / values.sum() # Normalize to sum to 1 (or other relevant total contribution)
    contributions = {f: v for f, v in zip(features, values)}
    return contributions

def simulate_global_influences(all_features):
    # Simulate global influence scores for all possible features
    values = np.random.rand(len(all_features))
    global_influence = {f: v for f, v in zip(all_features, values)}
    return global_influence

# --- UI Components and Logic --- 

st.title("💊 Healthcare Treatment Outcome Divergence Explorer")
st.markdown(
    "This application helps medical researchers and clinicians identify and understand "
    "divergent treatment outcomes in patient subgroups. Explore where treatment success "
    "rates or adverse event occurrences significantly differ from the baseline."
)

patient_data = generate_patient_data()
divergent_itemsets_df = simulate_divexplorer_output(patient_data)

all_possible_features = ["gender", "age_group", "comorbidity", "treatment_protocol", "drug_response"]

# Sidebar for global controls or filters (optional, expand as needed)
st.sidebar.header("Global Filters")
# Example filter: minimum divergence score
min_divergence = st.sidebar.slider(
    "Minimum Divergence Score", min_value=0.0, max_value=1.0, value=0.5, step=0.05
)
filtered_divergent_itemsets = divergent_itemsets_df[divergent_itemsets_df["divergence_score"] >= min_divergence]

st.header("1. Divergent Patient Subgroups")
st.write("Below is a table of identified patient subgroups exhibiting divergent treatment outcomes."
         "Select an itemset for a detailed exploration.")

if not filtered_divergent_itemsets.empty:
    st.dataframe(filtered_divergent_itemsets.style.background_gradient(cmap='YlOrRd', subset=['divergence_score']), use_container_width=True)

    selected_itemset_index = st.selectbox(
        "Select an itemset for detailed analysis:",
        filtered_divergent_itemsets.index,
        format_func=lambda x: f"{filtered_divergent_itemsets.loc[x, 'itemset']} (Score: {filtered_divergent_itemsets.loc[x, 'divergence_score']:.2f})"
    )

    if selected_itemset_index is not None:
        selected_itemset_data = filtered_divergent_itemsets.loc[selected_itemset_index]
        st.subheader(f"2. Detailed Exploration for: {selected_itemset_data['itemset']}")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write(f"**Outcome Metric:** {selected_itemset_data['outcome_metric']}")
            st.write(f"**Baseline Rate:** {selected_itemset_data['baseline_rate']:.2f}")
            st.write(f"**Itemset Rate:** {selected_itemset_data['itemset_rate']:.2f}")
            st.write(f"**Number of Patients in Subgroup:** {selected_itemset_data['size']}")
            
            st.markdown("### Local Item Contributions (Simulated Shapley Values)")
            shapley_values = simulate_shapley_values(selected_itemset_data["itemset"])
            shapley_df = pd.DataFrame(list(shapley_values.items()), columns=['Feature', 'Contribution'])
            fig_shapley = px.bar(shapley_df, x='Feature', y='Contribution',
                                 title='Contribution of Each Item to Divergence (Local Influence)',
                                 labels={'Contribution': 'Relative Contribution'}, 
                                 color='Contribution', color_continuous_scale=px.colors.sequential.Plasma)
            st.plotly_chart(fig_shapley, use_container_width=True)
            
        with col2:
            st.markdown("### Lattice Visualization (Subset Relationships)")
            # Simplified lattice: just showing the itemset and its direct subsets for demonstration
            G = nx.DiGraph()
            
            # The selected itemset node
            itemset_str = str(selected_itemset_data["itemset"])
            G.add_node(itemset_str, type="selected", label=itemset_str)
            
            # Add nodes for individual items (direct subsets as singletons)
            for item_key, item_value in selected_itemset_data["itemset"].items():
                single_item_str = f"{item_key}: {item_value}"
                G.add_node(single_item_str, type="subset", label=single_item_str)
                G.add_edge(itemset_str, single_item_str)

            # Simulate a few 'corrective' items not in the original itemset
            corrective_items = ["New Drug X", "Dosage Adjustment", "Combined Therapy Y"]
            for corr_item in corrective_items:
                G.add_node(corr_item, type="corrective", label=corr_item)
                G.add_edge(itemset_str, corr_item)

            pos = nx.spring_layout(G, k=0.8, iterations=50) # Layout for better visualization
            
            fig_lattice, ax_lattice = plt.subplots(figsize=(8, 6))
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, nodelist=[itemset_str], node_color='lightblue', node_size=3000, ax=ax_lattice)
            nx.draw_networkx_nodes(G, pos, nodelist=[n for n, attr in G.nodes(data=True) if attr.get('type') == 'subset'], node_color='lightgreen', node_size=2000, ax=ax_lattice)
            nx.draw_networkx_nodes(G, pos, nodelist=[n for n, attr in G.nodes(data=True) if attr.get('type') == 'corrective'], node_color='lightcoral', node_size=2000, ax=ax_lattice)

            # Draw edges
            nx.draw_networkx_edges(G, pos, ax=ax_lattice, arrowsize=20)
            
            # Draw labels
            labels = {node: data['label'] for node, data in G.nodes(data=True)}
            nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold', ax=ax_lattice)
            
            ax_lattice.set_title("Lattice of Subset & Corrective Item Relationships")
            st.pyplot(fig_lattice, use_container_width=True)
            
        st.markdown("### Global Item Influence Visualizations")
        global_influences = simulate_global_influences(all_possible_features)
        global_influence_df = pd.DataFrame(list(global_influences.items()), columns=['Feature', 'Influence'])
        fig_global = px.bar(global_influence_df, x='Feature', y='Influence',
                            title='Global Influence of Features on Treatment Outcomes',
                            labels={'Influence': 'Relative Influence Score'}, 
                            color='Influence', color_continuous_scale=px.colors.sequential.Viridis)
        st.plotly_chart(fig_global, use_container_width=True)

else:
    st.warning("No divergent itemsets found with the current filters.")