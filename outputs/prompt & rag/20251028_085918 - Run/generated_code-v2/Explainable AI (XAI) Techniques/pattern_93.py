class DataLoader:
    def __init__(self, data, labels, model):
        self.data = data
        self.labels = labels
        self.model = model

    def get_predictions(self):
        if callable(self.model):
            return [self.model(row) for row in self.data]
        else:
            return [0 if i % 2 == 0 else 1 for i in range(len(self.data))]

    def get_features_and_labels(self):
        return self.data, self.labels

class DataTransformer:
    def __init__(self, feature_names):
        self.feature_names = feature_names
        self.unique_values = {f: [] for f in feature_names}

    def fit(self, data):
        for row in data:
            for feature in self.feature_names:
                if feature in row and row[feature] not in self.unique_values[feature]:
                    self.unique_values[feature].append(row[feature])

    def transform_to_binary(self, data):
        binary_data = []
        encoded_feature_names = []
        for feature in self.feature_names:
            for val in self.unique_values[feature]:
                encoded_feature_names.append(f"{feature}_{val}")

        for row in data:
            binary_row = []
            for feature in self.feature_names:
                for val in self.unique_values[feature]:
                    binary_row.append(1 if row.get(feature) == val else 0)
            binary_data.append(binary_row)
        return binary_data, encoded_feature_names

class DivergenceCalculator:
    def calculate_metrics(self, true_labels, predictions):
        tp = sum(1 for t, p in zip(true_labels, predictions) if t == 1 and p == 1)
        tn = sum(1 for t, p in zip(true_labels, predictions) if t == 0 and p == 0)
        fp = sum(1 for t, p in zip(true_labels, predictions) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(true_labels, predictions) if t == 1 and p == 0)

        total = len(true_labels)
        accuracy = (tp + tn) / total if total > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        return {"accuracy": accuracy, "fpr": fpr, "fnr": fnr, "tp": tp, "tn": tn, "fp": fp, "fn": fn}

    def calculate_h_divergence(self, subgroup_metrics, overall_metrics, metric_name="fpr"):
        return abs(subgroup_metrics.get(metric_name, 0) - overall_metrics.get(metric_name, 0))

class FrequentSubgroupMiner:
    def __init__(self, min_support=0.1):
        self.min_support = min_support

    def find_frequent_itemsets(self, binary_data, encoded_feature_names):
        transactions = [tuple(f_name for i, val in enumerate(row) if val == 1) for row, f_name in zip(binary_data, [encoded_feature_names] * len(binary_data))]
        
        item_counts = {}
        for transaction in transactions:
            for item in transaction:
                item_counts[item] = item_counts.get(item, 0) + 1

        frequent_1_itemsets = {frozenset([item]) for item, count in item_counts.items() if count / len(binary_data) >= self.min_support}

        k = 1
        current_frequent_itemsets = frequent_1_itemsets
        all_frequent_itemsets = set(current_frequent_itemsets)

        while current_frequent_itemsets:
            k += 1
            candidate_itemsets = self._generate_candidates(current_frequent_itemsets, k)
            
            candidate_counts = {}
            for transaction in transactions:
                for candidate in candidate_itemsets:
                    if candidate.issubset(transaction):
                        candidate_counts[candidate] = candidate_counts.get(candidate, 0) + 1
            
            next_frequent_itemsets = set()
            for candidate, count in candidate_counts.items():
                if count / len(binary_data) >= self.min_support:
                    next_frequent_itemsets.add(candidate)
            
            current_frequent_itemsets = next_frequent_itemsets
            all_frequent_itemsets.update(current_frequent_itemsets)
            
        return all_frequent_itemsets

    def _generate_candidates(self, frequent_itemsets, k):
        candidates = set()
        list_frequent_itemsets = list(frequent_itemsets)
        n = len(list_frequent_itemsets)
        for i in range(n):
            for j in range(i + 1, n):
                itemset1 = list_frequent_itemsets[i]
                itemset2 = list_frequent_itemsets[j]
                
                union_set = itemset1.union(itemset2)
                if len(union_set) == k:
                    is_valid_candidate = True
                    for subset in self._get_subsets(union_set, k - 1):
                        if subset not in frequent_itemsets:
                            is_valid_candidate = False
                            break
                    if is_valid_candidate:
                        candidates.add(union_set)
        return candidates

    def _get_subsets(self, itemset, size):
        if size == 0:
            return {frozenset()}
        if size > len(itemset):
            return set()
        
        subsets = set()
        items = list(itemset)
        
        import itertools
        for combination in itertools.combinations(items, size):
            subsets.add(frozenset(combination))
        return subsets


class SignificanceTester:
    def is_significant(self, divergence_value, threshold=0.1):
        return divergence_value > threshold

class ShapleyContributor:
    def __init__(self, data, labels, model_predictions, feature_names, performance_metric_func):
        self.data = data
        self.labels = labels
        self.model_predictions = model_predictions
        self.feature_names = feature_names
        self.performance_metric_func = performance_metric_func

    def _get_subgroup_data_and_labels(self, original_data, true_labels, subgroup_features_dict):
        filtered_data = []
        filtered_labels = []
        for i, row in enumerate(original_data):
            match = True
            for feature, value in subgroup_features_dict.items():
                if row.get(feature) != value:
                    match = False
                    break
            if match:
                filtered_data.append(row)
                filtered_labels.append(true_labels[i])
        return filtered_data, filtered_labels


    def calculate_local_shapley(self, subgroup_itemset, overall_metrics, metric_name="fpr"):
        subgroup_features_dict = {}
        for item in subgroup_itemset:
            parts = item.split('_', 1)
            if len(parts) == 2:
                feature_name, value = parts
                subgroup_features_dict[feature_name] = value

        if not subgroup_features_dict:
            return {f: 0 for f in subgroup_itemset}

        feature_contributions = {item: 0.0 for item in subgroup_itemset}
        
        subgroup_data, subgroup_labels = self._get_subgroup_data_and_labels(self.data, self.labels, subgroup_features_dict)
        subgroup_predictions = [self.model_predictions[self.data.index(row)] for row in subgroup_data]

        if not subgroup_labels:
            return feature_contributions

        full_subgroup_metrics = self.performance_metric_func(subgroup_labels, subgroup_predictions)
        full_subgroup_divergence = self.performance_metric_func.calculate_h_divergence(full_subgroup_metrics, overall_metrics, metric_name)

        for target_feature_item in subgroup_itemset:
            temp_features_dict = {k: v for k, v in subgroup_features_dict.items() if f"{k}_{v}" != target_feature_item}

            if not temp_features_dict:
                feature_contributions[target_feature_item] = full_subgroup_divergence
                continue

            reduced_subgroup_data, reduced_subgroup_labels = self._get_subgroup_data_and_labels(self.data, self.labels, temp_features_dict)
            
            if not reduced_subgroup_labels:
                feature_contributions[target_feature_item] = full_subgroup_divergence
                continue
            
            reduced_subgroup_predictions = [self.model_predictions[self.data.index(row)] for row in reduced_subgroup_data]
            reduced_subgroup_metrics = self.performance_metric_func(reduced_subgroup_labels, reduced_subgroup_predictions)
            reduced_subgroup_divergence = self.performance_metric_func.calculate_h_divergence(reduced_subgroup_metrics, overall_metrics, metric_name)

            feature_contributions[target_feature_item] = full_subgroup_divergence - reduced_subgroup_divergence
            
        return feature_contributions

    def calculate_global_shapley(self, all_local_contributions):
        global_contributions = {}
        for subgroup_contributions in all_local_contributions:
            for feature, contribution in subgroup_contributions.items():
                global_contributions[feature] = global_contributions.get(feature, 0) + contribution
        return global_contributions

class CorrectiveSuggester:
    def suggest_corrective_items(self, divergent_subgroup, local_contributions, global_contributions, overall_metrics, data, model_predictions, labels, performance_metric_func, metric_name="fpr"):
        suggestions = []
        
        highly_contributing_features = sorted([
            (feat, contrib) for feat, contrib in local_contributions.items() if contrib > 0
        ], key=lambda x: x[1], reverse=True)

        for feature_item, contribution in highly_contributing_features:
            suggestions.append(f"Consider addressing '{feature_item}' which contributed {contribution:.4f} to the divergence in this subgroup. It's a high contributor.")
            
            feature_name = feature_item.split('_', 1)[0]
            current_value = feature_item.split('_', 1)[1]
            
            all_values_for_feature = [row.get(feature_name) for row in data if feature_name in row]
            unique_other_values = sorted(list(set(all_values_for_feature) - {current_value}))

            if unique_other_values:
                suggestions.append(f"  - Explore how changing '{feature_name}' from '{current_value}' to one of {unique_other_values} might impact divergence.")
        
        for feature, contrib in global_contributions.items():
            if contrib < 0:
                suggestions.append(f"Globally, '{feature}' tends to reduce divergence. Its presence/absence might be beneficial depending on the context.")

        return suggestions

class SubgroupPruner:
    def prune_redundant_subgroups(self, significant_subgroups_info, divergence_threshold_for_pruning=0.05):
        pruned_subgroups = []
        sorted_subgroups = sorted(significant_subgroups_info, key=lambda x: x['h_divergence'], reverse=True)

        for i, current_subgroup in enumerate(sorted_subgroups):
            is_redundant = False
            current_itemset = current_subgroup['subgroup_itemset']
            
            for j, existing_subgroup in enumerate(pruned_subgroups):
                existing_itemset = existing_subgroup['subgroup_itemset']
                
                if current_itemset.issubset(existing_itemset) or existing_itemset.issubset(current_itemset):
                    if abs(current_subgroup['h_divergence'] - existing_subgroup['h_divergence']) < divergence_threshold_for_pruning:
                        is_redundant = True
                        break
            
            if not is_redundant:
                pruned_subgroups.append(current_subgroup)
        
        return pruned_subgroups

class ReportGenerator:
    def generate_report(self, overall_metrics, pruned_subgroups_info, global_contributions):
        report = []
        report.append("--- Healthcare AI Model Fairness Auditor Report ---")
        report.append("\nOverall Model Performance:")
        for metric, value in overall_metrics.items():
            if isinstance(value, float):
                report.append(f"- {metric.replace('_', ' ').capitalize()}: {value:.4f}")
            else:
                report.append(f"- {metric.replace('_', ' ').capitalize()}: {value}")

        report.append("\nIdentified Divergent Subgroups (Pruned):")
        if not pruned_subgroups_info:
            report.append("No significant divergent subgroups found after pruning.")
        else:
            for i, subgroup in enumerate(pruned_subgroups_info):
                report.append(f"\nSubgroup {i+1}: {', '.join(subgroup['subgroup_itemset'])}")
                report.append(f"  - H-Divergence (FPR): {subgroup['h_divergence']:.4f} (Threshold: {subgroup['significance_threshold']:.4f})")
                report.append("  - Subgroup Metrics:")
                for metric, value in subgroup['subgroup_metrics'].items():
                    if isinstance(value, float):
                        report.append(f"    - {metric.replace('_', ' ').capitalize()}: {value:.4f}")
                    else:
                        report.append(f"    - {metric.replace('_', ' ').capitalize()}: {value}")
                report.append("  - Local Feature Contributions to Divergence:")
                if subgroup['local_contributions']:
                    for feature, contrib in sorted(subgroup['local_contributions'].items(), key=lambda item: item[1], reverse=True):
                        report.append(f"    - {feature}: {contrib:.4f}")
                else:
                    report.append("    No specific local contributions calculated for this subgroup.")
                report.append("  - Corrective Suggestions:")
                if subgroup['corrective_suggestions']:
                    for suggestion in subgroup['corrective_suggestions']:
                        report.append(f"    - {suggestion}")
                else:
                    report.append("    No specific corrective suggestions for this subgroup.")
        
        report.append("\nGlobal Feature Influence on Divergence:")
        if global_contributions:
            for feature, contrib in sorted(global_contributions.items(), key=lambda item: item[1], reverse=True):
                report.append(f"- {feature}: {contrib:.4f}")
        else:
            report.append("No global feature contributions calculated.")
            
        return "\n".join(report)

class VisualizationTool:
    def visualize_divergence(self, pruned_subgroups_info):
        print("\n--- Visualizations (Text-based for now) ---")
        if not pruned_subgroups_info:
            print("No data to visualize.")
            return

        print("\nDivergence Levels per Subgroup:")
        max_len = max(len(', '.join(s['subgroup_itemset'])) for s in pruned_subgroups_info) if pruned_subgroups_info else 0
        for subgroup in pruned_subgroups_info:
            subgroup_str = ', '.join(subgroup['subgroup_itemset'])
            divergence = subgroup['h_divergence']
            bar = "#" * int(divergence * 50)
            print(f"{subgroup_str.ljust(max_len)} |{bar} {divergence:.4f}")

    def visualize_global_contributions(self, global_contributions):
        print("\nGlobal Feature Contributions:")
        if not global_contributions:
            print("No global contributions to visualize.")
            return
        
        max_len = max(len(f) for f in global_contributions.keys()) if global_contributions else 0
        for feature, contrib in sorted(global_contributions.items(), key=lambda item: abs(item[1]), reverse=True):
            sign = "+" if contrib >= 0 else ""
            bar = "=" * int(abs(contrib) * 20)
            print(f"{feature.ljust(max_len)} | {sign}{contrib:.4f} {bar}")

class HealthcareFairnessAuditor:
    def __init__(self, model_interface, data, labels, feature_names, min_support=0.1, divergence_threshold=0.1, pruning_threshold=0.05):
        self.model_interface = model_interface
        self.raw_data = data
        self.true_labels = labels
        self.feature_names = feature_names
        self.min_support = min_support
        self.divergence_threshold = divergence_threshold
        self.pruning_threshold = pruning_threshold

        self.data_loader = DataLoader(self.raw_data, self.true_labels, self.model_interface)
        self.data_transformer = DataTransformer(self.feature_names)
        self.divergence_calculator = DivergenceCalculator()
        self.fpm_miner = FrequentSubgroupMiner(min_support=self.min_support)
        self.significance_tester = SignificanceTester()
        self.report_generator = ReportGenerator()
        self.visualization_tool = VisualizationTool()

    def run_auditor(self):
        predictions = self.data_loader.get_predictions()
        data, labels = self.data_loader.get_features_and_labels()
        
        self.data_transformer.fit(data)
        binary_data, encoded_feature_names = self.data_transformer.transform_to_binary(data)

        overall_metrics = self.divergence_calculator.calculate_metrics(labels, predictions)
        print(f"Overall Model FPR: {overall_metrics['fpr']:.4f}")

        frequent_itemsets = self.fpm_miner.find_frequent_itemsets(binary_data, encoded_feature_names)
        print(f"Found {len(frequent_itemsets)} frequent itemsets.")

        significant_subgroups_info = []
        shapley_contributor = ShapleyContributor(data, labels, predictions, self.feature_names, self.divergence_calculator)
        
        all_local_contributions = []

        for itemset in frequent_itemsets:
            if not itemset:
                continue

            subgroup_filter_dict = {}
            for item in itemset:
                parts = item.split('_', 1)
                if len(parts) == 2:
                    subgroup_filter_dict[parts[0]] = parts[1]

            subgroup_data_indices = []
            for i, row in enumerate(data):
                match = True
                for feat, val in subgroup_filter_dict.items():
                    if row.get(feat) != val:
                        match = False
                        break
                if match:
                    subgroup_data_indices.append(i)

            if not subgroup_data_indices:
                continue

            subgroup_true_labels = [labels[i] for i in subgroup_data_indices]
            subgroup_predictions = [predictions[i] for i in subgroup_data_indices]

            if not subgroup_true_labels:
                continue

            subgroup_metrics = self.divergence_calculator.calculate_metrics(subgroup_true_labels, subgroup_predictions)
            h_divergence = self.divergence_calculator.calculate_h_divergence(subgroup_metrics, overall_metrics, metric_name="fpr")
            
            if self.significance_tester.is_significant(h_divergence, threshold=self.divergence_threshold):
                local_contributions = shapley_contributor.calculate_local_shapley(itemset, overall_metrics, metric_name="fpr")
                all_local_contributions.append(local_contributions)
                
                corrective_suggester = CorrectiveSuggester()
                corrective_suggestions = corrective_suggester.suggest_corrective_items(
                    itemset, local_contributions, {}, data, predictions, labels, self.divergence_calculator, metric_name="fpr"
                )

                significant_subgroups_info.append({
                    "subgroup_itemset": itemset,
                    "h_divergence": h_divergence,
                    "subgroup_metrics": subgroup_metrics,
                    "local_contributions": local_contributions,
                    "corrective_suggestions": corrective_suggestions,
                    "significance_threshold": self.divergence_threshold
                })
        
        global_contributions = shapley_contributor.calculate_global_shapley(all_local_contributions)

        for subgroup_info in significant_subgroups_info:
            corrective_suggester = CorrectiveSuggester()
            subgroup_info["corrective_suggestions"] = corrective_suggester.suggest_corrective_items(
                subgroup_info["subgroup_itemset"], subgroup_info["local_contributions"], global_contributions, 
                overall_metrics, data, predictions, labels, self.divergence_calculator, metric_name="fpr"
            )

        subgroup_pruner = SubgroupPruner()
        pruned_subgroups_info = subgroup_pruner.prune_redundant_subgroups(significant_subgroups_info, self.pruning_threshold)
        print(f"Found {len(pruned_subgroups_info)} pruned significant divergent subgroups.")

        report = self.report_generator.generate_report(overall_metrics, pruned_subgroups_info, global_contributions)
        print(report)

        self.visualization_tool.visualize_divergence(pruned_subgroups_info)
        self.visualization_tool.visualize_global_contributions(global_contributions)

        return report

def dummy_model_predict(patient_data_row):
    if patient_data_row.get("Age") > 60 and patient_data_row.get("ConditionA") == "Yes":
        return 1
    elif patient_data_row.get("Gender") == "Female" and patient_data_row.get("ConditionB") == "No":
        return 0
    else:
        return 0

if __name__ == "__main__":
    patient_data = [
        {"PatientID": "P001", "Age": 70, "Gender": "Male", "ConditionA": "Yes", "ConditionB": "No", "Ethnicity": "Group1"},
        {"PatientID": "P002", "Age": 55, "Gender": "Female", "ConditionA": "No", "ConditionB": "Yes", "Ethnicity": "Group2"},
        {"PatientID": "P003", "Age": 72, "Gender": "Female", "ConditionA": "Yes", "ConditionB": "No", "Ethnicity": "Group1"},
        {"PatientID": "P004", "Age": 40, "Gender": "Male", "ConditionA": "No", "ConditionB": "No", "Ethnicity": "Group3"},
        {"PatientID": "P005", "Age": 68, "Gender": "Male", "ConditionA": "Yes", "ConditionB": "Yes", "Ethnicity": "Group1"},
        {"PatientID": "P006", "Age": 50, "Gender": "Female", "ConditionA": "No", "ConditionB": "No", "Ethnicity": "Group2"},
        {"PatientID": "P007", "Age": 75, "Gender": "Female", "ConditionA": "No", "ConditionB": "Yes", "Ethnicity": "Group1"},
        {"PatientID": "P008", "Age": 62, "Gender": "Male", "ConditionA": "Yes", "ConditionB": "No", "Ethnicity": "Group2"},
        {"PatientID": "P009", "Age": 30, "Gender": "Female", "ConditionA": "Yes", "ConditionB": "No", "Ethnicity": "Group3"},
        {"PatientID": "P010", "Age": 65, "Gender": "Male", "ConditionA": "No", "ConditionB": "Yes", "Ethnicity": "Group1"},
        {"PatientID": "P011", "Age": 71, "Gender": "Female", "ConditionA": "Yes", "ConditionB": "Yes", "Ethnicity": "Group2"},
        {"PatientID": "P012", "Age": 45, "Gender": "Male", "ConditionA": "No", "ConditionB": "No", "Ethnicity": "Group1"},
        {"PatientID": "P013", "Age": 63, "Gender": "Female", "ConditionA": "Yes", "ConditionB": "No", "Ethnicity": "Group3"},
        {"PatientID": "P014", "Age": 58, "Gender": "Male", "ConditionA": "No", "ConditionB": "Yes", "Ethnicity": "Group2"},
        {"PatientID": "P015", "Age": 78, "Gender": "Male", "ConditionA": "Yes", "ConditionB": "No", "Ethnicity": "Group1"},
    ]

    true_labels = [
        1,
        0,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        1
    ]

    feature_names = ["Age", "Gender", "ConditionA", "ConditionB", "Ethnicity"]

    auditor = HealthcareFairnessAuditor(
        model_interface=dummy_model_predict,
        data=patient_data,
        labels=true_labels,
        feature_names=feature_names,
        min_support=0.2,
        divergence_threshold=0.2,
        pruning_threshold=0.05
    )
    auditor.run_auditor()
