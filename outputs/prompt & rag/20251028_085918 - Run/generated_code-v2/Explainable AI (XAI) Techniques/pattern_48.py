import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from mlxtend.frequent_patterns import apriori

class DivergenceExplorer:
    def __init__(self, data, model_predictions, actual_outcomes, feature_cols):
        self.data = data.copy()
        self.data["predictions"] = model_predictions
        self.data["actuals"] = actual_outcomes
        self.feature_cols = feature_cols
        self.overall_fpr, self.overall_fnr = self._calculate_performance(self.data["predictions"], self.data["actuals"])

    def _calculate_performance(self, predictions, actuals):
        tn, fp, fn, tp = confusion_matrix(actuals, predictions).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        return fpr, fnr

    def h_divergence(self, subgroup_data, metric="fpr"):
        if subgroup_data.empty or len(subgroup_data) < 2:
            return 0, False

        subgroup_fpr, subgroup_fnr = self._calculate_performance(subgroup_data["predictions"], subgroup_data["actuals"])

        if metric == "fpr":
            divergence = subgroup_fpr - self.overall_fpr
            is_divergent = divergence > (0.2 * self.overall_fpr + 0.05) and subgroup_data.shape[0] >= 50
            return divergence, is_divergent
        elif metric == "fnr":
            divergence = subgroup_fnr - self.overall_fnr
            is_divergent = divergence > (0.2 * self.overall_fnr + 0.05) and subgroup_data.shape[0] >= 50
            return divergence, is_divergent
        else:
            raise ValueError("Metric must be 'fpr' or 'fnr'")

    def find_divergent_subgroups(self, min_support=0.05, metric="fpr"):
        df_encoded = pd.get_dummies(self.data[self.feature_cols].astype(str), columns=self.feature_cols, prefix_sep='=')

        frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)

        divergent_subgroups_raw = []
        for _, row in frequent_itemsets.iterrows():
            itemset = frozenset(row["itemsets"])
            
            query_parts = []
            for item in itemset:
                feature, value = item.split("=", 1)
                query_parts.append(f"`{feature}` == '{value}'")
            query = " and ".join(query_parts)

            subgroup_data = self.data.query(query)

            divergence, is_divergent = self.h_divergence(subgroup_data, metric=metric)

            if is_divergent:
                divergent_subgroups_raw.append({
                    "itemset": itemset,
                    "size": len(subgroup_data),
                    "divergence": divergence,
                    "subgroup_data": subgroup_data
                })
        
        return self.prune_redundant_subgroups(divergent_subgroups_raw, metric=metric)

    def attribute_contributions(self, divergent_subgroup_info):
        itemset = divergent_subgroup_info["itemset"]
        
        contributions = {}
        for item in itemset:
            contributions[item] = 1.0

        corrective_items = list(itemset)

        return {"local_contributions": contributions, "corrective_items": corrective_items}

    def prune_redundant_subgroups(self, divergent_subgroups, divergence_tolerance=0.05):
        pruned_subgroups = []
        
        divergent_subgroups.sort(key=lambda x: (x["divergence"], x["size"]), reverse=True)

        for current_sg in divergent_subgroups:
            is_redundant = False
            for kept_sg in pruned_subgroups:
                if current_sg["itemset"].issubset(kept_sg["itemset"]) and \
                   abs(current_sg["divergence"] - kept_sg["divergence"]) < divergence_tolerance:
                    is_redundant = True
                    break
            
            if not is_redundant:
                pruned_subgroups.append(current_sg)
        
        return pruned_subgroups

def main():
    np.random.seed(42)
    data_size = 1000
    data = pd.DataFrame({
        "age_group": np.random.choice(["<40", "40-60", ">60"], data_size),
        "gender": np.random.choice(["Male", "Female"], data_size),
        "comorbidity_diabetes": np.random.choice([0, 1], data_size, p=[0.7, 0.3]).astype(str),
        "comorbidity_hypertension": np.random.choice([0, 1], data_size, p=[0.6, 0.4]).astype(str),
        "treatment_protocol_A": np.random.choice([0, 1], data_size, p=[0.8, 0.2]).astype(str),
        "actual_readmission": np.random.choice([0, 1], data_size, p=[0.85, 0.15])
    })

    predictions = data["actual_readmission"].apply(lambda x: 1 - x if np.random.rand() < 0.1 else x)

    predictions[(data["age_group"] == ">60") & (data["actual_readmission"] == 0) & (np.random.rand(len(data[(data["age_group"] == ">60") & (data["actual_readmission"] == 0)])) < 0.4)] = 1
    predictions[(data["gender"] == "Female") & (data["actual_readmission"] == 1) & (np.random.rand(len(data[(data["gender"] == "Female") & (data["actual_readmission"] == 1)])) < 0.5)] = 0
    
    predictions = (predictions > 0.5).astype(int)

    feature_cols = ["age_group", "gender", "comorbidity_diabetes", "comorbidity_hypertension", "treatment_protocol_A"]

    explorer = DivergenceExplorer(data.copy(), predictions, data["actual_readmission"], feature_cols)

    print(f"Overall FPR: {explorer.overall_fpr:.4f}")
    print(f"Overall FNR: {explorer.overall_fnr:.4f}\n")

    print("Finding divergent subgroups (FPR metric)...")
    divergent_fpr_subgroups = explorer.find_divergent_subgroups(min_support=0.03, metric="fpr")
    
    if divergent_fpr_subgroups:
        print("\nDivergent Subgroups (FPR):")
        for i, sg in enumerate(divergent_fpr_subgroups):
            print(f"  Subgroup {i+1}:")
            print(f"    Itemset: {list(sg['itemset'])}")
            print(f"    Size: {sg['size']}")
            print(f"    Divergence (vs overall FPR): {sg['divergence']:.4f}")

            contributions_info = explorer.attribute_contributions(sg)
            print(f"    Local Feature Contributions (Conceptual): {contributions_info['local_contributions']}")
            print(f"    Corrective Items (Conceptual): {contributions_info['corrective_items']}")
            print("-" * 30)
    else:
        print("No significant divergent subgroups found for FPR.")

    print("\nFinding divergent subgroups (FNR metric)...")
    divergent_fnr_subgroups = explorer.find_divergent_subgroups(min_support=0.03, metric="fnr")

    if divergent_fnr_subgroups:
        print("\nDivergent Subgroups (FNR):")
        for i, sg in enumerate(divergent_fnr_subgroups):
            print(f"  Subgroup {i+1}:")
            print(f"    Itemset: {list(sg['itemset'])}")
            print(f"    Size: {sg['size']}")
            print(f"    Divergence (vs overall FNR): {sg['divergence']:.4f}")

            contributions_info = explorer.attribute_contributions(sg)
            print(f"    Local Feature Contributions (Conceptual): {contributions_info['local_contributions']}")
            print(f"    Corrective Items (Conceptual): {contributions_info['corrective_items']}")
            print("-" * 30)
    else:
        print("No significant divergent subgroups found for FNR.")

if __name__ == "__main__":
    main()