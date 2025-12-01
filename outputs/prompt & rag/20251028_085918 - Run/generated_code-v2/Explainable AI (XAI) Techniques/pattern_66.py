import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from collections import defaultdict, Counter
import itertools


class DataHandler:
    def load_patient_data(self, filepath="dummy_patient_data.csv"):
        df = pd.read_csv(filepath)
        return df

    def preprocess_data(self, df):
        df = df.copy()
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].fillna(df[col].mode()[0])
        for col in df.select_dtypes(include=["number"]).columns:
            df[col] = df[col].fillna(df[col].mean())
        return df

    def binarize_features(self, df, categorical_cols, numerical_cols_to_binarize):
        df_binarized = df.copy()
        
        # One-hot encode categorical features
        if categorical_cols:
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            encoded_cats = encoder.fit_transform(df_binarized[categorical_cols])
            encoded_cat_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_cols), index=df_binarized.index)
            df_binarized = df_binarized.drop(columns=categorical_cols).join(encoded_cat_df)

        # Binarize numerical features (simple thresholding for demonstration)
        for col, (threshold, operator) in numerical_cols_to_binarize.items():
            if operator == '>':
                df_binarized[f'{col}_gt_{threshold}'] = (df_binarized[col] > threshold).astype(int)
            elif operator == '<':
                df_binarized[f'{col}_lt_{threshold}'] = (df_binarized[col] < threshold).astype(int)
            elif operator == '>=':
                df_binarized[f'{col}_ge_{threshold}'] = (df_binarized[col] >= threshold).astype(int)
            elif operator == '<=':
                df_binarized[f'{col}_le_{threshold}'] = (df_binarized[col] <= threshold).astype(int)
            df_binarized = df_binarized.drop(columns=[col])
            
        return df_binarized


class SubgroupMiner:
    def find_frequent_itemsets(self, binarized_df, min_support):
        min_count = int(min_support * len(binarized_df))
        
        item_counts = Counter()
        for _, row in binarized_df.iterrows():
            for item in row.index[row == 1]:
                item_counts[item] += 1
        
        frequent_itemsets = {frozenset([item]) for item, count in item_counts.items() if count >= min_count}
        
        k = 1
        while True:
            k += 1
            candidate_itemsets = self._generate_candidates(frequent_itemsets, k)
            if not candidate_itemsets:
                break
            
            candidate_counts = defaultdict(int)
            for _, row in binarized_df.iterrows():
                items_in_row = frozenset(row.index[row == 1])
                for candidate in candidate_itemsets:
                    if candidate.issubset(items_in_row):
                        candidate_counts[candidate] += 1
            
            new_frequent_itemsets = {itemset for itemset, count in candidate_counts.items() if count >= min_count}
            if not new_frequent_itemsets:
                break
            frequent_itemsets.update(new_frequent_itemsets)

        return [list(itemset) for itemset in frequent_itemsets]

    def _generate_candidates(self, frequent_itemsets, k):
        candidates = set()
        list_frequent_itemsets = list(frequent_itemsets)
        for i in range(len(list_frequent_itemsets)):
            for j in range(i + 1, len(list_frequent_itemsets)):
                itemset1 = list_frequent_itemsets[i]
                itemset2 = list_frequent_itemsets[j]
                
                union = itemset1.union(itemset2)
                if len(union) == k:
                    is_valid = True
                    for subset in itertools.combinations(union, k - 1):
                        if frozenset(subset) not in frequent_itemsets:
                            is_valid = False
                            break
                    if is_valid:
                        candidates.add(union)
        return candidates


class DivergenceAnalyzer:
    def calculate_h_divergence(self, model, data, subgroup_mask, target_column, divergence_metric='fpr'):
        predictions = model.predict(data.drop(columns=[target_column]))
        true_labels = data[target_column]

        # Overall population metrics
        tn, fp, fn, tp = self._get_confusion_matrix_elements(true_labels, predictions)
        overall_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        overall_fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        # Subgroup metrics
        subgroup_true_labels = true_labels[subgroup_mask]
        subgroup_predictions = predictions[subgroup_mask]
        
        if len(subgroup_true_labels) == 0:
            return 0.0

        sub_tn, sub_fp, sub_fn, sub_tp = self._get_confusion_matrix_elements(subgroup_true_labels, subgroup_predictions)
        subgroup_fpr = sub_fp / (sub_fp + sub_tn) if (sub_fp + sub_tn) > 0 else 0
        subgroup_fnr = sub_fn / (sub_fn + sub_tp) if (sub_fn + sub_tp) > 0 else 0

        if divergence_metric == 'fpr':
            divergence = abs(subgroup_fpr - overall_fpr)
        elif divergence_metric == 'fnr':
            divergence = abs(subgroup_fnr - overall_fnr)
        else:
            raise ValueError("divergence_metric must be 'fpr' or 'fnr'")
        
        return divergence

    def _get_confusion_matrix_elements(self, true_labels, predictions):
        tn = ((predictions == 0) & (true_labels == 0)).sum()
        fp = ((predictions == 1) & (true_labels == 0)).sum()
        fn = ((predictions == 0) & (true_labels == 1)).sum()
        tp = ((predictions == 1) & (true_labels == 1)).sum()
        return tn, fp, fn, tp

    def bayesian_significance_test(self, baseline_metric, subgroup_metric, baseline_n, subgroup_n):
        return "Conceptual: A full Bayesian significance test would compare posterior distributions. Assuming significant if divergence > a threshold." # Placeholder

    def identify_corrective_items(self, model, data, subgroup_mask, divergent_itemset, target_column, metric):
        original_divergence = self.calculate_h_divergence(model, data, subgroup_mask, target_column, metric)
        potential_corrective_items = []

        for item in divergent_itemset:
            temp_data = data.copy()
            
            # Attempt to 'correct' the item within the subgroup
            # This is a very simplified perturbation: flip the item's value for the subgroup
            # For binary features, 0 becomes 1 and 1 becomes 0.
            
            # Find the column in temp_data corresponding to the item
            # This handles both one-hot encoded and binarized numerical features
            if item in temp_data.columns:
                original_values = temp_data.loc[subgroup_mask, item].copy()
                # Simple flip for binary features
                temp_data.loc[subgroup_mask, item] = 1 - original_values
            else:
                # If item is not a direct column, it might be a complex numerical binarization
                # This simplification assumes direct binary features or one-hot encoded ones.
                continue

            new_divergence = self.calculate_h_divergence(model, temp_data, subgroup_mask, target_column, metric)
            if new_divergence < original_divergence:
                potential_corrective_items.append((item, new_divergence))
        
        potential_corrective_items.sort(key=lambda x: x[1])
        return potential_corrective_items


class ShapleyExplainer:
    def calculate_local_shapley_divergence(self, model, data, itemset, target_column, divergence_function, n_samples=50):
        if not itemset:
            return {}

        shapley_values = defaultdict(float)
        num_features = len(itemset)
        
        # Convert itemset to a list for indexing
        itemset_list = list(itemset)

        # Create a base mask for the itemset members in the full data
        # This means a row is 1 if it has ALL items in the itemset
        # However, for Shapley, we need to add/remove features from the context.
        # This simplification assumes we're calculating contribution to the *subgroup's* divergence.
        # The actual subgroup mask for divergence_function will be built based on permutations.

        # Simplified Monte Carlo approximation for Shapley values
        for _ in range(n_samples):
            permutation = np.random.permutation(itemset_list)
            
            for i, feature in enumerate(permutation):
                # Marginal contribution of 'feature'
                
                # Value with feature
                coalition_with_feature = frozenset(permutation[:i+1])
                mask_with_feature = pd.Series(True, index=data.index)
                for f in coalition_with_feature:
                    # This assumes itemset features are directly in data as binary columns
                    if f in data.columns:
                        mask_with_feature &= (data[f] == 1)
                    else:
                        # If a feature in the itemset isn't a direct column, skip or handle appropriately.
                        # For this simplified implementation, we assume direct correspondence.
                        mask_with_feature = pd.Series(False, index=data.index) # Effectively makes the mask empty
                        break
                
                val_with_feature = divergence_function(model, data, mask_with_feature, target_column)

                # Value without feature
                coalition_without_feature = frozenset(permutation[:i])
                mask_without_feature = pd.Series(True, index=data.index)
                for f in coalition_without_feature:
                    if f in data.columns:
                        mask_without_feature &= (data[f] == 1)
                    else:
                        mask_without_feature = pd.Series(False, index=data.index)
                        break
                
                val_without_feature = divergence_function(model, data, mask_without_feature, target_column)

                marginal_contribution = val_with_feature - val_without_feature
                shapley_values[feature] += marginal_contribution

        for feature in shapley_values:
            shapley_values[feature] /= n_samples
            
        return dict(shapley_values)

    def calculate_global_item_divergence(self, all_divergent_subgroups_with_shapley):
        global_shapley = defaultdict(float)
        for subgroup_info in all_divergent_subgroups_with_shapley:
            shapley_values = subgroup_info['local_shapley_values']
            for item, value in shapley_values.items():
                global_shapley[item] += value
        return dict(global_shapley)


class ClinicalFairnessAuditor:
    def __init__(self, model, data, target_column, categorical_features, numerical_features_to_binarize):
        self.model = model
        self.original_data = data
        self.target_column = target_column
        self.categorical_features = categorical_features
        self.numerical_features_to_binarize = numerical_features_to_binarize

        self.data_handler = DataHandler()
        self.subgroup_miner = SubgroupMiner()
        self.divergence_analyzer = DivergenceAnalyzer()
        self.shapley_explainer = ShapleyExplainer()

        self.processed_data = self.data_handler.preprocess_data(data.drop(columns=[target_column]))
        self.processed_data[target_column] = data[target_column] # Re-add target for divergence calc

        self.binarized_data_for_fpm = self.data_handler.binarize_features(
            self.original_data.drop(columns=[target_column]), 
            self.categorical_features, 
            self.numerical_features_to_binarize
        )

    def audit_model(self, min_support, divergence_metric='fpr', min_divergence=0.1):
        print("\n--- Starting Clinical Fairness Audit ---")
        
        print("1. Finding frequent itemsets (patient subgroups)...")
        frequent_itemsets = self.subgroup_miner.find_frequent_itemsets(self.binarized_data_for_fpm, min_support)
        print(f"Found {len(frequent_itemsets)} frequent itemsets.")

        divergent_subgroups = []
        print("2. Analyzing divergence for each subgroup...")
        for itemset in frequent_itemsets:
            # Create a mask for the current itemset in the processed_data (original features + target)
            subgroup_mask_data = pd.Series(True, index=self.original_data.index)
            for item in itemset:
                # Need to map binarized itemset features back to original data or its processed form
                # This mapping can be complex depending on binarization rules.
                # For simplicity, we assume itemset features are directly present as binary in binarized_data_for_fpm
                # and that we can use these to filter the processed_data.
                if item in self.binarized_data_for_fpm.columns:
                    subgroup_mask_data &= (self.binarized_data_for_fpm[item] == 1)
                else:
                    # Handle cases where an item in itemset might not directly map to a binary column
                    # This part needs careful design depending on feature engineering.
                    # For now, if we can't find it, the mask will become false for that item's contribution
                    pass # If item not found, it won't contribute to mask, might reduce subgroup size
            
            if subgroup_mask_data.sum() == 0:
                continue

            divergence_score = self.divergence_analyzer.calculate_h_divergence(
                self.model, self.processed_data, subgroup_mask_data, self.target_column, divergence_metric
            )
            
            if divergence_score >= min_divergence:
                significance = self.divergence_analyzer.bayesian_significance_test(
                    None, divergence_score, len(self.original_data), subgroup_mask_data.sum()
                )
                divergent_subgroups.append({
                    'itemset': itemset,
                    'size': subgroup_mask_data.sum(),
                    'divergence': divergence_score,
                    'metric': divergence_metric,
                    'significance': significance,
                    'mask': subgroup_mask_data # Store mask for later use
                })
        
        divergent_subgroups.sort(key=lambda x: x['divergence'], reverse=True)
        print(f"Found {len(divergent_subgroups)} divergent subgroups.")

        all_divergent_subgroups_with_shapley = []
        print("3. Calculating local Shapley values and identifying corrective items...")
        for i, subgroup_info in enumerate(divergent_subgroups):
            print(f"  Processing subgroup {i+1}/{len(divergent_subgroups)}: {subgroup_info['itemset']}")
            
            # For Shapley, we need to pass the full binarized_data_for_fpm to evaluate feature presence/absence
            local_shapley_values = self.shapley_explainer.calculate_local_shapley_divergence(
                self.model, self.processed_data, subgroup_info['itemset'], self.target_column,
                lambda model, data_inner, mask_inner, target_col: self.divergence_analyzer.calculate_h_divergence(
                    model, data_inner, mask_inner, target_col, divergence_metric
                )
            )
            subgroup_info['local_shapley_values'] = local_shapley_values

            corrective_items = self.divergence_analyzer.identify_corrective_items(
                self.model, self.processed_data, subgroup_info['mask'], subgroup_info['itemset'],
                self.target_column, divergence_metric
            )
            subgroup_info['corrective_items'] = corrective_items
            all_divergent_subgroups_with_shapley.append(subgroup_info)

        print("4. Calculating global item divergence...")
        global_item_divergence = self.shapley_explainer.calculate_global_item_divergence(all_divergent_subgroups_with_shapley)

        print("5. Pruning redundant subgroups...")
        final_divergent_subgroups = self._prune_redundant_subgroups(all_divergent_subgroups_with_shapley)

        results = {
            'final_divergent_subgroups': final_divergent_subgroups,
            'global_item_divergence': global_item_divergence
        }
        self.generate_report(results)
        return results

    def _prune_redundant_subgroups(self, divergent_subgroups):
        # Simple pruning heuristic: if a subgroup is a subset of another subgroup with similar divergence
        # and the superset is also divergent, we might consider the subset redundant.
        # For demonstration, we'll keep it simple: just return sorted by divergence for now.
        # A more sophisticated approach would involve checking overlaps and divergence similarity.
        return divergent_subgroups

    def generate_report(self, results):
        print("\n--- Clinical Fairness Audit Report ---")
        
        print("\nIdentified Divergent Subgroups (Top 5):")
        for i, subgroup in enumerate(results['final_divergent_subgroups'][:5]):
            print(f"  Subgroup {i+1}: {', '.join(subgroup['itemset'])}")
            print(f"    Size: {subgroup['size']} patients")
            print(f"    Divergence ({subgroup['metric']}): {subgroup['divergence']:.4f}")
            print(f"    Significance: {subgroup['significance']}")
            
            if subgroup['local_shapley_values']:
                print("    Local Item Contributions (Shapley Values):")
                sorted_shapley = sorted(subgroup['local_shapley_values'].items(), key=lambda x: abs(x[1]), reverse=True)
                for item, value in sorted_shapley[:3]:
                    print(f"      - {item}: {value:.4f}")
            
            if subgroup['corrective_items']:
                print("    Potential Corrective Items (reduce divergence):")
                for item, new_div in subgroup['corrective_items'][:3]:
                    print(f"      - Changing '{item}' could reduce divergence to {new_div:.4f}")
            print("----------------------------------------")

        print("\nGlobal Item Divergence (Top 5 overall influential items):")
        sorted_global_shapley = sorted(results['global_item_divergence'].items(), key=lambda x: abs(x[1]), reverse=True)
        for item, value in sorted_global_shapley[:5]:
            print(f"  - {item}: {value:.4f}")
        
        print("\n--- Audit Complete ---")


# --- Example Usage --- #
if __name__ == "__main__":
    # 1. Create dummy patient data
    data = pd.DataFrame({
        'age': np.random.randint(20, 80, 1000),
        'gender': np.random.choice(['Male', 'Female'], 1000),
        'smoker': np.random.choice([0, 1], 1000),
        'bmi': np.random.uniform(18, 40, 1000),
        'blood_pressure': np.random.randint(90, 180, 1000),
        'disease_risk': np.random.choice([0, 1], 1000, p=[0.7, 0.3]) # Target variable
    })

    # Introduce a bias: Older male smokers have higher disease risk and model might misclassify them more.
    data.loc[(data['age'] > 60) & (data['gender'] == 'Male') & (data['smoker'] == 1), 'disease_risk'] = np.random.choice([0, 1], sum((data['age'] > 60) & (data['gender'] == 'Male') & (data['smoker'] == 1)), p=[0.2, 0.8])
    
    # Make some data points where model performs poorly for a specific group
    # For instance, younger females with high BMI might be misclassified as low risk by model, but they are actually high risk
    data.loc[(data['age'] < 40) & (data['gender'] == 'Female') & (data['bmi'] > 30), 'disease_risk'] = np.random.choice([0, 1], sum((data['age'] < 40) & (data['gender'] == 'Female') & (data['bmi'] > 30)), p=[0.1, 0.9])

    # Save dummy data for DataHandler to load
    data.to_csv("dummy_patient_data.csv", index=False)

    # 2. Train a simple black-box classification model
    X = data.drop(columns=['disease_risk'])
    y = data['disease_risk']

    # Preprocessing for the model itself (not FPM binarization)
    categorical_features_model = ['gender']
    numerical_features_model = ['age', 'bmi', 'blood_pressure', 'smoker']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features_model),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features_model)
        ], remainder='passthrough'
    )

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(solver='liblinear', random_state=42))
    ])
    
    # Fit the model on the full data for the auditor to use
    model_pipeline.fit(X, y)

    # Define features for the ClinicalFairnessAuditor
    target_column = 'disease_risk'
    categorical_features_for_auditor = ['gender', 'smoker'] # Use original categories for auditor to binarize
    numerical_features_to_binarize_for_auditor = {
        'age': (60, '>'), 
        'bmi': (30, '>'),
        'blood_pressure': (140, '>')
    }

    # 3. Instantiate and run the ClinicalFairnessAuditor
    auditor = ClinicalFairnessAuditor(
        model=model_pipeline,
        data=data,
        target_column=target_column,
        categorical_features=categorical_features_for_auditor,
        numerical_features_to_binarize=numerical_features_to_binarize_for_auditor
    )

    audit_results = auditor.audit_model(
        min_support=0.05,        # Minimum support for frequent itemsets
        divergence_metric='fpr', # Metric to quantify divergence (False Positive Rate)
        min_divergence=0.05      # Minimum divergence to consider a subgroup significant
    )
