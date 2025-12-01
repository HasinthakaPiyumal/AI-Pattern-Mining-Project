import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import random

class DataGenerator:
    def generate_synthetic_data(self, num_samples=1000):
        np.random.seed(42)
        data = {
            'Age': np.random.randint(20, 80, num_samples),
            'Gender': np.random.choice(['Male', 'Female'], num_samples, p=[0.5, 0.5]),
            'BloodPressure': np.random.randint(90, 180, num_samples),
            'Cholesterol': np.random.randint(150, 300, num_samples),
            'Smoking': np.random.choice([0, 1], num_samples, p=[0.7, 0.3]),
            'Diabetes': np.random.choice([0, 1], num_samples, p=[0.85, 0.15]),
            'FamilyHistory': np.random.choice([0, 1], num_samples, p=[0.6, 0.4]),
            'ExerciseHoursPerWeek': np.random.randint(1, 10, num_samples),
            'Diagnosis': np.zeros(num_samples, dtype=int)
        }

        df = pd.DataFrame(data)

        # Introduce some correlation for diagnosis
        df.loc[(df['Age'] > 60) & (df['Cholesterol'] > 250) & (df['Smoking'] == 1), 'Diagnosis'] = 1
        df.loc[(df['Diabetes'] == 1) & (df['BloodPressure'] > 140), 'Diagnosis'] = 1
        df.loc[(df['Gender'] == 'Female') & (df['Age'] > 50) & (df['FamilyHistory'] == 1) & (np.random.rand(len(df[(df['Gender'] == 'Female') & (df['Age'] > 50) & (df['FamilyHistory'] == 1)])) < 0.3), 'Diagnosis'] = 1 # Introduce some gender/age bias
        df.loc[df['Diagnosis'] == 0, 'Diagnosis'] = np.random.choice([0, 1], len(df[df['Diagnosis'] == 0]), p=[0.95, 0.05]) # Add some baseline diagnoses

        return df

class MockMedicalDiagnosisModel:
    def __init__(self):
        self.model = RandomForestClassifier(random_state=42)
        self.label_encoders = {}
        self.features = []
        self.target = ''

    def preprocess(self, df, target_column):
        df_processed = df.copy()
        self.features = [col for col in df.columns if col != target_column]
        self.target = target_column

        for col in self.features:
            if df_processed[col].dtype == 'object':
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col])
                self.label_encoders[col] = le
        return df_processed[self.features], df_processed[self.target]

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return self.model.predict(X)

class DivExplorerSimulator:
    def __init__(self, data_original, features, true_labels, model_predictions, model_probabilities):
        self.data_original = data_original.copy()
        self.features = features
        self.true_labels = true_labels
        self.model_predictions = model_predictions
        self.model_probabilities = model_probabilities

        self.data_original['TrueLabel'] = self.true_labels
        self.data_original['ModelPrediction'] = self.model_predictions
        self.data_original['IsMisclassified'] = (self.data_original['TrueLabel'] != self.data_original['ModelPrediction']).astype(int)

    def simulate_divergent_subgroups(self, num_subgroups=10):
        misclassified_data = self.data_original[self.data_original['IsMisclassified'] == 1].copy()
        if misclassified_data.empty:
            return pd.DataFrame(columns=['Itemset', 'DivergenceScore', 'MisclassificationCount', 'SubgroupSize'])

        divergent_subgroups = []
        unique_features = self.features.copy()
        random.shuffle(unique_features)

        for _ in range(num_subgroups):
            num_items = random.randint(1, min(3, len(unique_features)))
            itemset_features = random.sample(unique_features, num_items)

            # Create a simple itemset by picking a random row from misclassified data and its feature values
            sample_row = misclassified_data.sample(1).iloc[0]
            itemset = {f: sample_row[f] for f in itemset_features}

            # Filter data for this itemset
            subgroup_filter = pd.Series([True] * len(self.data_original))
            for k, v in itemset.items():
                subgroup_filter &= (self.data_original[k] == v)

            subgroup_data = self.data_original[subgroup_filter]

            if not subgroup_data.empty:
                misclassification_count = subgroup_data['IsMisclassified'].sum()
                subgroup_size = len(subgroup_data)
                divergence_score = (misclassification_count / subgroup_size) if subgroup_size > 0 else 0
                
                if divergence_score > 0.1: # Only consider substantially divergent subgroups
                    divergent_subgroups.append({
                        'Itemset': str(itemset),
                        'DivergenceScore': divergence_score,
                        'MisclassificationCount': misclassification_count,
                        'SubgroupSize': subgroup_size
                    })
        
        df_divergent = pd.DataFrame(divergent_subgroups)
        return df_divergent.sort_values(by='DivergenceScore', ascending=False).drop_duplicates(subset=['Itemset']).reset_index(drop=True)

    def simulate_local_shapley_values(self, itemset_str, top_n=5):
        itemset_dict = eval(itemset_str) # Convert string back to dict
        
        # Simulate Shapley-like contributions for features in the itemset
        contributions = {feature: random.uniform(0.05, 0.4) for feature in itemset_dict.keys()}
        
        # Add some random contributions for other features to make it more realistic
        other_features = [f for f in self.features if f not in itemset_dict.keys()]
        for _ in range(min(top_n - len(contributions), len(other_features))):
            feature = random.choice(other_features)
            contributions[feature] = random.uniform(0.01, 0.2)
            if feature in other_features:
                other_features.remove(feature)

        df_shapley = pd.DataFrame(list(contributions.items()), columns=['Feature', 'Contribution'])
        return df_shapley.sort_values(by='Contribution', ascending=False)

    def simulate_lattice_visualization(self, selected_itemset_str):
        # A highly simplified simulation of a lattice
        itemset_dict = eval(selected_itemset_str)
        
        lattice_data = []
        lattice_data.append(f"Root: All Data (Error Rate: {self.data_original['IsMisclassified'].mean():.2f})")
        lattice_data.append(f"  | ")
        lattice_data.append(f"  \-- {selected_itemset_str} (Divergence Score: {random.uniform(0.1, 0.5):.2f})")

        # Simulate related subsets/supersets
        for _ in range(random.randint(1, 3)):
            modified_itemset = itemset_dict.copy()
            if len(modified_itemset) > 1 and random.random() < 0.5: # Remove an item for subset
                modified_itemset.pop(random.choice(list(modified_itemset.keys())))
                lattice_data.append(f"      \-- Subset: {str(modified_itemset)} (Divergence Score: {random.uniform(0.05, 0.3):.2f})")
            else: # Add an item for superset or just another related itemset
                remaining_features = [f for f in self.features if f not in modified_itemset.keys()]
                if remaining_features:
                    added_feature = random.choice(remaining_features)
                    modified_itemset[added_feature] = random.choice(self.data_original[added_feature].unique().tolist())
                    lattice_data.append(f"      \-- Related: {str(modified_itemset)} (Divergence Score: {random.uniform(0.1, 0.6):.2f})")

        return "\n".join(lattice_data)

    def simulate_global_item_influence(self, top_n=7):
        # Simulate global influence scores for all features
        influence = {feature: random.uniform(0.0, 1.0) for feature in self.features}
        df_influence = pd.DataFrame(list(influence.items()), columns=['Feature', 'Influence'])
        return df_influence.sort_values(by='Influence', ascending=False).head(top_n)

st.set_page_config(layout="wide")
st.title("MediBias Explorer: Interactive System for Medical Diagnosis Model Bias Analysis")

# --- Data Generation ---
@st.cache_data
def load_data():
    return DataGenerator().generate_synthetic_data(num_samples=2000)

df = load_data()

# --- Model Training ---
mock_model = MockMedicalDiagnosisModel()
X, y = mock_model.preprocess(df, 'Diagnosis')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
mock_model.train(X_train, y_train)

model_predictions = mock_model.predict(X_test)
model_probabilities = mock_model.predict_proba(X_test)

# --- DivExplorer Simulation ---
div_explorer = DivExplorerSimulator(df.loc[X_test.index], mock_model.features, y_test, model_predictions, model_probabilities)
divergent_subgroups_df = div_explorer.simulate_divergent_subgroups(num_subgroups=50)

st.sidebar.header("Configuration")
min_divergence_score = st.sidebar.slider("Minimum Divergence Score", 0.0, 1.0, 0.2, 0.05)
filtered_subgroups = divergent_subgroups_df[divergent_subgroups_df['DivergenceScore'] >= min_divergence_score]

# --- Main Content Area ---
st.header("1. Divergent Subgroups")
st.write("Explore patient subgroups where the model exhibits significantly different (often incorrect) behavior.")

search_query = st.text_input("Search divergent itemsets (e.g., 'Gender', 'Age')")
if search_query:
    filtered_subgroups = filtered_subgroups[filtered_subgroups['Itemset'].str.contains(search_query, case=False)]

st.dataframe(filtered_subgroups.sort_values(by='DivergenceScore', ascending=False), use_container_width=True)

if not filtered_subgroups.empty:
    st.header("2. Detailed Subgroup Analysis")
    selected_itemset_str = st.selectbox(
        "Select an itemset for detailed analysis:",
        filtered_subgroups['Itemset'].tolist()
    )

    if selected_itemset_str:
        st.subheader(f"Analysis for Itemset: {selected_itemset_str}")

        col1, col2 = st.columns(2)
        with col1:
            st.write("#### Local Item Contributions (Shapley-like Values)")
            shapley_df = div_explorer.simulate_local_shapley_values(selected_itemset_str)
            fig_shapley, ax_shapley = plt.subplots(figsize=(8, 4))
            ax_shapley.barh(shapley_df['Feature'], shapley_df['Contribution'], color='skyblue')
            ax_shapley.set_xlabel("Contribution")
            ax_shapley.set_ylabel("Feature")
            ax_shapley.set_title("Feature Contributions to Divergence")
            plt.tight_layout()
            st.pyplot(fig_shapley)
        
        with col2:
            st.write("#### Lattice Visualization (Subset Relationships)")
            st.code(div_explorer.simulate_lattice_visualization(selected_itemset_str), language='text')
            st.info("This is a simplified textual representation of related subgroups.")

st.header("3. Global Feature Influence")
st.write("Understand which features generally have the most influence on model decisions or divergence across all data.")

global_influence_df = div_explorer.simulate_global_item_influence()
fig_global, ax_global = plt.subplots(figsize=(10, 5))
ax_global.barh(global_influence_df['Feature'], global_influence_df['Influence'], color='lightcoral')
ax_global.set_xlabel("Influence Score")
ax_global.set_ylabel("Feature")
ax_global.set_title("Global Feature Influence on Model Behavior")
plt.tight_layout()
st.pyplot(fig_global)