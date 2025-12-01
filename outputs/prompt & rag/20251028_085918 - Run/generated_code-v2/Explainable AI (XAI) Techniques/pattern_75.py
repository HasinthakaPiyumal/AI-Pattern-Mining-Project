import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import joblib
import itertools
from scipy.stats import beta


class DataProcessor:
    def __init__(self, categorical_features, numerical_features):
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.imputer = SimpleImputer(strategy='most_frequent') # For both for simplicity
        self.one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.scaler = StandardScaler()
        self.feature_names_out = None

    def fit_transform(self, df):
        df_copy = df.copy()

        # Impute missing values
        if self.categorical_features or self.numerical_features:
            all_features = self.categorical_features + self.numerical_features
            df_copy[all_features] = self.imputer.fit_transform(df_copy[all_features])

        # One-hot encode categorical features
        encoded_features = None
        if self.categorical_features:
            self.one_hot_encoder.fit(df_copy[self.categorical_features])
            encoded_features = self.one_hot_encoder.transform(df_copy[self.categorical_features])
            encoded_df = pd.DataFrame(encoded_features, columns=self.one_hot_encoder.get_feature_names_out(self.categorical_features), index=df_copy.index)
        else:
            encoded_df = pd.DataFrame(index=df_copy.index)

        # Scale numerical features
        scaled_features = None
        if self.numerical_features:
            self.scaler.fit(df_copy[self.numerical_features])
            scaled_features = self.scaler.transform(df_copy[self.numerical_features])
            scaled_df = pd.DataFrame(scaled_features, columns=self.numerical_features, index=df_copy.index)
        else:
            scaled_df = pd.DataFrame(index=df_copy.index)
        
        # Combine all features
        processed_df = pd.concat([encoded_df, scaled_df], axis=1)
        self.feature_names_out = processed_df.columns.tolist()
        return processed_df

    def transform(self, df):
        df_copy = df.copy()

        if self.categorical_features or self.numerical_features:
            all_features = self.categorical_features + self.numerical_features
            df_copy[all_features] = self.imputer.transform(df_copy[all_features])

        encoded_df = pd.DataFrame(index=df_copy.index)
        if self.categorical_features:
            encoded_features = self.one_hot_encoder.transform(df_copy[self.categorical_features])
            encoded_df = pd.DataFrame(encoded_features, columns=self.one_hot_encoder.get_feature_names_out(self.categorical_features), index=df_copy.index)

        scaled_df = pd.DataFrame(index=df_copy.index)
        if self.numerical_features:
            scaled_features = self.scaler.transform(df_copy[self.numerical_features])
            scaled_df = pd.DataFrame(scaled_features, columns=self.numerical_features, index=df_copy.index)
        
        processed_df = pd.concat([encoded_df, scaled_df], axis=1)
        return processed_df


class BlackBoxModelWrapper:
    def __init__(self, model_path=None, model=None):
        if model_path:
            self.model = joblib.load(model_path)
        elif model is not None:
            self.model = model
        else:
            raise ValueError("Either model_path or a model object must be provided.")

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class DivExplorerAnalyzer:
    def __init__(self, black_box_model_wrapper, data_processor, reference_data, reference_labels, reference_predictions):
        self.model_wrapper = black_box_model_wrapper
        self.data_processor = data_processor
        self.reference_data = reference_data
        self.reference_labels = reference_labels
        self.reference_predictions = reference_predictions

        self.reference_metrics = self._calculate_performance_metrics(reference_labels, reference_predictions)

    def _calculate_performance_metrics(self, true_labels, predicted_labels):
        tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        return {"fpr": fpr, "fnr": fnr, "accuracy": accuracy}

    def calculate_h_divergence(self, subgroup_labels, subgroup_predictions, metric="fpr"):
        subgroup_metrics = self._calculate_performance_metrics(subgroup_labels, subgroup_predictions)
        return subgroup_metrics[metric] - self.reference_metrics[metric]

    def is_significant(self, subgroup_size, divergence_value, alpha=0.05, metric="fpr"):
        # Simplified Bayesian significance test placeholder.
        # In a real scenario, this would involve comparing posterior distributions
        # For demonstration, we use a simple heuristic based on sample size and divergence magnitude.
        if subgroup_size < 30:
            return False

        # A very basic Bayesian-like check: if divergence is large enough relative to uncertainty
        # using a Beta distribution for rates (like FPR/FNR)
        ref_rate = self.reference_metrics[metric]
        sub_rate = ref_rate + divergence_value

        # Assuming uniform priors (alpha=1, beta=1) for simplicity
        # Calculate credible intervals for reference and subgroup rates
        # This is a highly simplified proxy, not a full Bayesian test
        a_ref, b_ref = 1 + int(ref_rate * len(self.reference_labels)), 1 + int((1 - ref_rate) * len(self.reference_labels))
        a_sub, b_sub = 1 + int(sub_rate * subgroup_size), 1 + int((1 - sub_rate) * subgroup_size)

        # Check if the credible intervals are sufficiently separated
        # This is a heuristic and needs proper statistical backing in a real application
        if beta.ppf(1 - alpha/2, a_ref, b_ref) < beta.ppf(alpha/2, a_sub, b_sub) and divergence_value > 0.05:
             return True
        if beta.ppf(alpha/2, a_ref, b_ref) > beta.ppf(1 - alpha/2, a_sub, b_sub) and divergence_value < -0.05:
             return True
        
        return False

    def find_frequent_subgroups(self, processed_data, min_support=0.1, max_k=3):
        encoded_data_bool = (processed_data > 0).astype(int)
        
        frequent_itemsets = {}
        # Frequent 1-itemsets
        for col in encoded_data_bool.columns:
            support = encoded_data_bool[col].sum() / len(encoded_data_bool)
            if support >= min_support:
                frequent_itemsets[frozenset([col])] = support

        # Generate k-itemsets from (k-1)-itemsets
        current_frequent_k_minus_1 = frequent_itemsets
        for k in range(2, max_k + 1):
            next_frequent_k = {}
            itemsets_k_minus_1_list = list(current_frequent_k_minus_1.keys())
            for i in range(len(itemsets_k_minus_1_list)):
                for j in range(i + 1, len(itemsets_k_minus_1_list)):
                    itemset1 = itemsets_k_minus_1_list[i]
                    itemset2 = itemsets_k_minus_1_list[j]
                    
                    union_itemset = itemset1.union(itemset2)
                    if len(union_itemset) == k:
                        # Check if all (k-1)-subsets are frequent (Apriori principle)
                        is_candidate = True
                        for subset in itertools.combinations(union_itemset, k - 1):
                            if frozenset(subset) not in current_frequent_k_minus_1:
                                is_candidate = False
                                break
                        
                        if is_candidate:
                            # Calculate support for the candidate k-itemset
                            mask = pd.Series(True, index=encoded_data_bool.index)
                            for item in union_itemset:
                                if item in encoded_data_bool.columns:
                                    mask = mask & (encoded_data_bool[item] == 1)
                                else:
                                    mask = False # Item not found, mask becomes false
                            
                            support = mask.sum() / len(encoded_data_bool) if len(encoded_data_bool) > 0 else 0

                            if support >= min_support:
                                next_frequent_k[union_itemset] = support
            
            if not next_frequent_k:
                break
            frequent_itemsets.update(next_frequent_k)
            current_frequent_k_minus_1 = next_frequent_k

        return frequent_itemsets

    def analyze_divergence(self, full_data, full_labels, full_predictions, min_support=0.05, max_k=2, metric="fpr", alpha=0.05):
        processed_data_for_fpm = self.data_processor.transform(full_data)

        frequent_subgroups = self.find_frequent_subgroups(processed_data_for_fpm, min_support=min_support, max_k=max_k)

        divergent_subgroups_report = []

        for subgroup_items, support in frequent_subgroups.items():
            mask = pd.Series(True, index=processed_data_for_fpm.index)
            for item in subgroup_items:
                if item in processed_data_for_fpm.columns:
                    mask = mask & (processed_data_for_fpm[item] == 1)
                else:
                    mask = False # Should not happen if items come from processed_data_for_fpm.columns

            subgroup_data_original_indices = full_data[mask].index
            
            if len(subgroup_data_original_indices) == 0: # Ensure subgroup is not empty
                continue

            subgroup_labels = full_labels.loc[subgroup_data_original_indices]
            subgroup_predictions = full_predictions.loc[subgroup_data_original_indices]

            divergence = self.calculate_h_divergence(subgroup_labels, subgroup_predictions, metric=metric)
            is_sig = self.is_significant(len(subgroup_labels), divergence, alpha=alpha, metric=metric)

            if is_sig:
                divergent_subgroups_report.append({
                    "subgroup": list(subgroup_items),
                    "support": support,
                    f"h_divergence_{metric}": divergence,
                    "subgroup_size": len(subgroup_labels),
                    "is_significant": is_sig,
                    "subgroup_metrics": self._calculate_performance_metrics(subgroup_labels, subgroup_predictions)
                })
        return divergent_subgroups_report

    def calculate_local_shapley_contribution(self, subgroup, data_point, feature_values, target_metric_fn):
        # Placeholder for actual Shapley value calculation.
        # This would typically involve permuting feature values and observing changes in the target_metric_fn.
        # A full implementation requires a library like 'shap' or a more complex custom function.
        
        # For demonstration, we'll return mock contributions
        mock_contributions = {feature: np.random.uniform(-0.1, 0.1) for feature in subgroup}
        return mock_contributions

    def calculate_global_item_divergence(self, divergent_subgroups_report):
        # Placeholder for global item divergence calculation.
        # This would aggregate local Shapley values or other feature importance metrics across all divergent subgroups.
        # It aims to identify features that consistently contribute to divergence.

        global_contributions = {}
        for report in divergent_subgroups_report:
            for item in report["subgroup"]:
                # This is a highly simplified aggregation. Real implementation would be more nuanced.
                global_contributions[item] = global_contributions.get(item, 0) + report[f"h_divergence_{report['metric']}"]
        
        # Sort and return top items contributing to global divergence
        sorted_global = sorted(global_contributions.items(), key=lambda item: abs(item[1]), reverse=True)
        return sorted_global
    
    def identify_corrective_items(self, divergent_subgroup_report, target_metric="fpr"): 
        # Placeholder for identifying corrective actions. 
        # This would involve analyzing features that, if changed, would reduce divergence.
        # For example, by looking at feature distributions within divergent vs. non-divergent parts of the subgroup.
        
        corrective_suggestions = []
        for report in divergent_subgroup_report:
            subgroup = report["subgroup"]
            current_divergence = report[f"h_divergence_{target_metric}"]
            
            if current_divergence > 0.05: # High FPR, suggest reducing it
                # Mock suggestion: if 'age_old' is in subgroup, suggest interventions for older patients
                for item in subgroup:
                    if "age" in item and "_old" in item:
                        corrective_suggestions.append(f"Consider tailored interventions for {item.replace('_', ' ')} patients to reduce {target_metric} divergence.")
                    elif "medication" in item:
                         corrective_suggestions.append(f"Review '{item.replace('medication_', '')}' usage in {subgroup} to reduce {target_metric} divergence.")

        return list(set(corrective_suggestions)) # Return unique suggestions



# --- Example Usage --- #
if __name__ == "__main__":
    # 1. Generate Dummy Data
    np.random.seed(42)
    num_samples = 1000
    data = pd.DataFrame({
        "age": np.random.randint(20, 80, num_samples),
        "gender": np.random.choice(["Male", "Female", "Other"], num_samples),
        "disease_severity": np.random.choice(["Low", "Medium", "High"], num_samples, p=[0.5, 0.3, 0.2]),
        "treatment_type": np.random.choice(["A", "B", "C"], num_samples),
        "comorbidity": np.random.randint(0, 3, num_samples), # Number of comorbidities
        "pre_existing_condition_X": np.random.choice([0, 1], num_samples, p=[0.7, 0.3]),
        "pre_existing_condition_Y": np.random.choice([0, 1], num_samples, p=[0.8, 0.2]),
        "outcome": np.random.randint(0, 2, num_samples) # 0: Failure, 1: Success
    })

    # Simulate a bias: higher failure rate for 'Female' and 'disease_severity_High'
    data.loc[(data["gender"] == "Female") & (data["disease_severity"] == "High"), "outcome"] = np.random.choice([0, 1], sum((data["gender"] == "Female") & (data["disease_severity"] == "High")), p=[0.7, 0.3])
    data.loc[(data["age"] > 60) & (data["treatment_type"] == "C"), "outcome"] = np.random.choice([0, 1], sum((data["age"] > 60) & (data["treatment_type"] == "C")), p=[0.6, 0.4])

    X = data.drop("outcome", axis=1)
    y = data["outcome"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    categorical_features = ["gender", "disease_severity", "treatment_type"]
    numerical_features = ["age", "comorbidity", "pre_existing_condition_X", "pre_existing_condition_Y"]

    # 2. Data Ingestion and Preprocessing
    data_processor = DataProcessor(categorical_features, numerical_features)
    X_train_processed = data_processor.fit_transform(X_train)
    X_test_processed = data_processor.transform(X_test)

    # Align columns between processed training and test data if they differ
    # This is important for consistent model input
    train_cols = set(X_train_processed.columns)
    test_cols = set(X_test_processed.columns)

    missing_in_test = list(train_cols - test_cols)
    for col in missing_in_test:
        X_test_processed[col] = 0
    
    missing_in_train = list(test_cols - train_cols)
    for col in missing_in_train:
        X_train_processed[col] = 0
    
    X_test_processed = X_test_processed[X_train_processed.columns] # Ensure column order is the same


    # 3. Train a Black-box Model (Logistic Regression for simplicity)
    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X_train_processed, y_train)
    # Save model for demonstration of loading
    model_path = "black_box_model.joblib"
    joblib.dump(model, model_path)

    # 4. Black-box Model Integration
    model_wrapper = BlackBoxModelWrapper(model_path=model_path)
    
    # Predictions for the entire test set (reference data)
    y_pred_test = pd.Series(model_wrapper.predict(X_test_processed), index=y_test.index)
    y_proba_test = pd.DataFrame(model_wrapper.predict_proba(X_test_processed), index=y_test.index)

    # 5. DivExplorer Core Logic
    analyzer = DivExplorerAnalyzer(model_wrapper, data_processor, 
                                   X_test, y_test, y_pred_test)

    print("\n--- Analyzing Divergent Subgroups (FPR) ---")
    divergent_subgroups_fpr = analyzer.analyze_divergence(
        full_data=X_test, 
        full_labels=y_test, 
        full_predictions=y_pred_test, 
        min_support=0.03, 
        max_k=2,
        metric="fpr",
        alpha=0.1 # Increased alpha for easier demonstration of significance
    )

    if divergent_subgroups_fpr:
        for sg in divergent_subgroups_fpr:
            print(f"Subgroup: {sg['subgroup']}, Support: {sg['support']:.2f}, FPR Divergence: {sg['h_divergence_fpr']:.3f}, Subgroup Size: {sg['subgroup_size']}, Subgroup FPR: {sg['subgroup_metrics']['fpr']:.3f}, Significant: {sg['is_significant']}")
        
        print("\n--- Global Item Divergence (FPR) ---")
        global_divergence_items = analyzer.calculate_global_item_divergence(divergent_subgroups_fpr)
        for item, div in global_divergence_items[:5]:
            print(f"Item: {item}, Aggregated Divergence: {div:.3f}")
        
        print("\n--- Corrective Item Suggestions ---")
        corrective_actions = analyzer.identify_corrective_items(divergent_subgroups_fpr, target_metric="fpr")
        for action in corrective_actions:
            print(f"- {action}")

    else:
        print("No significant divergent subgroups found for FPR.")

    print("\n--- Analyzing Divergent Subgroups (FNR) ---")
    divergent_subgroups_fnr = analyzer.analyze_divergence(
        full_data=X_test, 
        full_labels=y_test, 
        full_predictions=y_pred_test, 
        min_support=0.03, 
        max_k=2,
        metric="fnr",
        alpha=0.1 # Increased alpha for easier demonstration of significance
    )

    if divergent_subgroups_fnr:
        for sg in divergent_subgroups_fnr:
            print(f"Subgroup: {sg['subgroup']}, Support: {sg['support']:.2f}, FNR Divergence: {sg['h_divergence_fnr']:.3f}, Subgroup Size: {sg['subgroup_size']}, Subgroup FNR: {sg['subgroup_metrics']['fnr']:.3f}, Significant: {sg['is_significant']}")
        
        print("\n--- Corrective Item Suggestions ---")
        corrective_actions = analyzer.identify_corrective_items(divergent_subgroups_fnr, target_metric="fnr")
        for action in corrective_actions:
            print(f"- {action}")
    else:
        print("No significant divergent subgroups found for FNR.")


