import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
import networkx as nx


def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)
    data = {
        'age': np.random.randint(20, 80, num_samples),
        'gender': np.random.choice(['Male', 'Female'], num_samples),
        'ethnicity': np.random.choice(['Caucasian', 'African American', 'Asian', 'Hispanic'], num_samples),
        'condition_A': np.random.randint(0, 2, num_samples),
        'condition_B': np.random.randint(0, 2, num_samples),
        'smoker': np.random.randint(0, 2, num_samples),
        'income_bracket': np.random.choice(['Low', 'Medium', 'High'], num_samples),
    }
    df = pd.DataFrame(data)

    # Simulate a 'true_label' (e.g., disease presence)
    df['true_label'] = ((df['age'] > 50).astype(int) +
                        (df['condition_A'] == 1).astype(int) +
                        (df['smoker'] == 1).astype(int) * 2 +
                        (df['gender'] == 'Female').astype(int) * 0.5 # Small gender factor
                       ).apply(lambda x: 1 if x > 2.5 else 0)
    return df


def dummy_model_predict(df):
    # Simulate a black-box model with some inherent biases
    X = df[['age', 'gender', 'ethnicity', 'condition_A', 'condition_B', 'smoker', 'income_bracket']]
    X_encoded = pd.get_dummies(X, columns=['gender', 'ethnicity', 'income_bracket'], drop_first=True)

    # Use a simple RandomForest for simulation
    # In a real scenario, this would be a pre-trained external model
    model = RandomForestClassifier(random_state=42, n_estimators=10)
    # Train on a subset to simulate an existing model
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, df['true_label'], test_size=0.3, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_encoded)
    probabilities = model.predict_proba(X_encoded)[:, 1]

    # Introduce synthetic bias: e.g., lower accuracy for 'African American' females over 60
    biased_indices = df[(df['ethnicity'] == 'African American') & (df['gender'] == 'Female') & (df['age'] > 60)].index
    for idx in biased_indices:
        if np.random.rand() < 0.7:  # 70% chance to flip prediction if it was correct
            if predictions[idx] == df['true_label'].iloc[idx]:
                predictions[idx] = 1 - predictions[idx]

    return predictions, probabilities


def simulate_bias_detection(df, predictions, true_labels):
    results = df.copy()
    results['prediction'] = predictions
    results['true_label'] = true_labels
    results['correct'] = (results['prediction'] == results['true_label']).astype(int)

    # Identify divergent cohorts based on accuracy difference from overall accuracy
    overall_accuracy = accuracy_score(true_labels, predictions)

    divergent_cohorts = []
    features = ['gender', 'ethnicity', 'condition_A', 'condition_B', 'smoker', 'income_bracket']

    for f in features:
        for val in results[f].unique():
            subgroup = results[results[f] == val]
            if len(subgroup) > 30: # Only consider sufficiently large subgroups
                subgroup_accuracy = accuracy_score(subgroup['true_label'], subgroup['prediction'])
                divergence = overall_accuracy - subgroup_accuracy
                if abs(divergence) > 0.05: # Threshold for divergence
                    divergent_cohorts.append({
                        'Cohort': f"{f}: {val}",
                        'Divergence Score': round(divergence, 3),
                        'Subgroup Accuracy': round(subgroup_accuracy, 3),
                        'Overall Accuracy': round(overall_accuracy, 3),
                        'Count': len(subgroup)
                    })

    # Also consider some two-feature interactions (simplified for demo)
    if 'ethnicity' in features and 'gender' in features:
        for eth in results['ethnicity'].unique():
            for gen in results['gender'].unique():
                subgroup = results[(results['ethnicity'] == eth) & (results['gender'] == gen)]
                if len(subgroup) > 20:
                    subgroup_accuracy = accuracy_score(subgroup['true_label'], subgroup['prediction'])
                    divergence = overall_accuracy - subgroup_accuracy
                    if abs(divergence) > 0.08:
                        divergent_cohorts.append({
                            'Cohort': f"Ethnicity: {eth}, Gender: {gen}",
                            'Divergence Score': round(divergence, 3),
                            'Subgroup Accuracy': round(subgroup_accuracy, 3),
                            'Overall Accuracy': round(overall_accuracy, 3),
                            'Count': len(subgroup)
                        })

    divergent_df = pd.DataFrame(divergent_cohorts).sort_values(by='Divergence Score', ascending=False)
    return divergent_df


def simulate_shap_values(cohort_name, features):
    # Simulate Shapley values for a given cohort - random for demonstration
    np.random.seed(hash(cohort_name) % (2**32 - 1))
    shap_values = {f: np.random.uniform(-0.5, 0.5) for f in features}
    # Emphasize features related to the cohort for realism
    for f in features:
        if f in cohort_name:
            shap_values[f] += np.random.uniform(0.5, 1.0) * (1 if np.random.rand() > 0.5 else -1)
    return pd.DataFrame([shap_values]).T.reset_index().rename(columns={'index': 'Feature', 0: 'Shapley Value'})


def create_cohort_lattice(divergent_df):
    G = nx.DiGraph()
    cohorts = divergent_df['Cohort'].tolist()

    for i, c1 in enumerate(cohorts):
        G.add_node(c1, color='lightblue', size=5 + abs(divergent_df['Divergence Score'].iloc[i]) * 50)
        for j, c2 in enumerate(cohorts):
            if i != j:
                # Simplified subset logic: if c1's description contains c2's description
                # This is a heuristic for demonstration
                c1_parts = set([part.strip() for part in c1.replace(':', ',').split(',') if part.strip()])
                c2_parts = set([part.strip() for part in c2.replace(':', ',').split(',') if part.strip()])
                if c1_parts.issuperset(c2_parts) and len(c1_parts) > len(c2_parts):
                     G.add_edge(c2, c1)

    pos = nx.spring_layout(G, k=0.8, iterations=50)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_text = []
    node_size = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        node_size.append(G.nodes[node]['size'])

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition='bottom center',
        marker=dict(
            showscale=False,
            color=node_size,
            size=node_size,
            colorbar=dict(
                thickness=15,
                title='Divergence Magnitude',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title='Cohort Relationship Lattice',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[ dict(
                        text="(Nodes sized by divergence magnitude)",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002 ) ],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    return fig


def simulate_global_feature_influence(df_original, predictions, true_labels):
    results = df_original.copy()
    results['prediction'] = predictions
    results['true_label'] = true_labels
    results['error'] = abs(results['prediction'] - results['true_label'])

    features = ['age', 'gender', 'ethnicity', 'condition_A', 'condition_B', 'smoker', 'income_bracket']
    global_influence = {}

    for feature in features:
        if feature in ['age']:
            # For continuous features, categorize or use correlation with error
            global_influence[feature] = results['age'].corr(results['error']) * 10 # Simulate a value
        elif df_original[feature].dtype == 'object':
            # For categorical features, look at average error rate by category
            error_rates = results.groupby(feature)['error'].mean()
            global_influence[feature] = error_rates.max() - error_rates.min() # Difference in error rates
        else:
            # For binary features
            error_rates = results.groupby(feature)['error'].mean()
            global_influence[feature] = error_rates.iloc[1] - error_rates.iloc[0]

    influence_df = pd.DataFrame.from_dict(global_influence, orient='index', columns=['Influence Score'])
    influence_df = influence_df.sort_values(by='Influence Score', ascending=False)

    fig = go.Figure(data=[go.Bar(x=influence_df.index, y=influence_df['Influence Score'])])
    fig.update_layout(title='Global Feature Influence on Model Error',
                      xaxis_title='Feature', yaxis_title='Influence Score')
    return fig


# Streamlit App
st.set_page_config(layout="wide")
st.title("🔬 MedBias Explorer: Interactive Bias Analysis in Healthcare AI")
st.markdown("Explore and understand divergent model behaviors in patient subgroups.")

if 'data' not in st.session_state:
    st.session_state.data = generate_synthetic_data()
    st.session_state.predictions, st.session_state.probabilities = dummy_model_predict(st.session_state.data)
    st.session_state.divergent_cohorts = simulate_bias_detection(st.session_state.data, st.session_state.predictions, st.session_state.data['true_label'])


st.sidebar.header("Controls")
if st.sidebar.button("Re-run Simulation"):
    st.session_state.data = generate_synthetic_data()
    st.session_state.predictions, st.session_state.probabilities = dummy_model_predict(st.session_state.data)
    st.session_state.divergent_cohorts = simulate_bias_detection(st.session_state.data, st.session_state.predictions, st.session_state.data['true_label'])
    st.experimental_rerun()

st.header("1. Divergent Patient Cohorts")
st.markdown("Table of patient subgroups where the AI model exhibits significant divergence in performance.")

search_term = st.text_input("Search cohorts by attribute (e.g., 'Ethnicity', 'Female')", "")

filtered_cohorts = st.session_state.divergent_cohorts
if search_term:
    filtered_cohorts = st.session_state.divergent_cohorts[st.session_state.divergent_cohorts['Cohort'].str.contains(search_term, case=False, na=False)]

st.dataframe(filtered_cohorts, use_container_width=True)


st.header("2. Local Feature Contributions (Simulated Shapley Values)")
st.markdown("Understand which specific features contribute most to the model's prediction within a selected divergent cohort.")

if not st.session_state.divergent_cohorts.empty:
    selected_cohort_name = st.selectbox(
        "Select a divergent cohort to analyze local feature contributions:",
        st.session_state.divergent_cohorts['Cohort'].tolist()
    )

    if selected_cohort_name:
        # Extract relevant features from the main dataframe for simulating SHAP
        all_features = ['age', 'gender', 'ethnicity', 'condition_A', 'condition_B', 'smoker', 'income_bracket']
        shap_df = simulate_shap_values(selected_cohort_name, all_features)

        fig_shap = go.Figure(data=[go.Bar(x=shap_df['Shapley Value'], y=shap_df['Feature'], orientation='h')])
        fig_shap.update_layout(title=f'Simulated Feature Contributions for {selected_cohort_name}',
                              xaxis_title='Contribution to Prediction (Simulated)',
                              yaxis_title='Feature')
        st.plotly_chart(fig_shap, use_container_width=True)
else:
    st.info("No divergent cohorts found to display local feature contributions.")


st.header("3. Cohort Relationship Lattice")
st.markdown("Visualize how different patient attributes combine to form divergent subgroups and their relationships.")

if not st.session_state.divergent_cohorts.empty and len(st.session_state.divergent_cohorts) > 1:
    lattice_fig = create_cohort_lattice(st.session_state.divergent_cohorts.head(10)) # Limit to top 10 for readability
    st.plotly_chart(lattice_fig, use_container_width=True)
else:
    st.info("Not enough divergent cohorts to build a meaningful lattice visualization.")

st.header("4. Global Feature Influence")
st.markdown("See which features generally contribute to model errors or divergent behavior across all cohorts.")

global_influence_fig = simulate_global_feature_influence(
    st.session_state.data,
    st.session_state.predictions,
    st.session_state.data['true_label']
)
st.plotly_chart(global_influence_fig, use_container_width=True)


st.markdown("--- Developed for AI Project based on DivExplorer Pattern")