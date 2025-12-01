import random
import itertools
import math

class HealthcareModelFairnessAuditor:
    def __init__(self, model_predict_fn, min_support=0.05, significance_threshold=0.05):
        """
        Initializes the Healthcare Model Fairness Auditor.

        Args:
            model_predict_fn (callable): A function that takes a patient record (dict) and returns a model prediction (0 or 1).
            min_support (float): Minimum support for frequent pattern mining.
            significance_threshold (float): Alpha level for statistical significance.
        """
        self.model_predict_fn = model_predict_fn
        self.min_support = min_support
        self.significance_threshold = significance_threshold
        self.base_rates = {'FPR': 0.0, 'FNR': 0.0, 'Accuracy': 0.0}

    def _simulate_healthcare_data(self, num_patients=1000):
        """
        Simulates structured healthcare data.
        Features: age_group, gender, pre_existing_condition, treatment_protocol.
        Target: actual_outcome (e.g., disease progression: 1=yes, 0=no).
        """
        data = []
        age_groups = ['child', 'adult', 'senior']
        genders = ['male', 'female']
        conditions = ['none', 'diabetes', 'heart_disease', 'asthma']
        treatments = ['protocol_A', 'protocol_B', 'protocol_C']

        for i in range(num_patients):
            patient = {
                'id': i,
                'age_group': random.choice(age_groups),
                'gender': random.choice(genders),
                'pre_existing_condition': random.choice(conditions),
                'treatment_protocol': random.choice(treatments),
                'actual_outcome': random.choices([0, 1], weights=[0.8, 0.2], k=1)[0] # 20% positive outcome (e.g., disease progressed)
            }
            data.append(patient)
        return data

    def _evaluate_model_on_data(self, data):
        """
        Evaluates the black-box model on the given data and adds predictions.
        Also calculates true positives, negatives, false positives, negatives.
        """
        evaluated_data = []
        for patient in data:
            prediction = self.model_predict_fn(patient) # Assume model_predict_fn returns 0 or 1
            patient_eval = patient.copy()
            patient_eval['model_prediction'] = prediction

            tp = 1 if patient['actual_outcome'] == 1 and prediction == 1 else 0
            tn = 1 if patient['actual_outcome'] == 0 and prediction == 0 else 0
            fp = 1 if patient['actual_outcome'] == 0 and prediction == 1 else 0
            fn = 1 if patient['actual_outcome'] == 1 and prediction == 0 else 0

            patient_eval['TP'] = tp
            patient_eval['TN'] = tn
            patient_eval['FP'] = fp
            patient_eval['FN'] = fn
            evaluated_data.append(patient_eval)

        return evaluated_data

    def _calculate_metrics(self, subgroup_data):
        """
        Calculates FPR, FNR, and Accuracy for a given subgroup of data.
        """
        total_actual_positive = sum(d['actual_outcome'] for d in subgroup_data)
        total_actual_negative = len(subgroup_data) - total_actual_positive

        fp_count = sum(d['FP'] for d in subgroup_data)
        fn_count = sum(d['FN'] for d in subgroup_data)
        correct_predictions = sum(d['TP'] + d['TN'] for d in subgroup_data)

        fpr = fp_count / total_actual_negative if total_actual_negative > 0 else 0.0
        fnr = fn_count / total_actual_positive if total_actual_positive > 0 else 0.0
        accuracy = correct_predictions / len(subgroup_data) if len(subgroup_data) > 0 else 0.0

        return {'FPR': fpr, 'FNR': fnr, 'Accuracy': accuracy, 'Count': len(subgroup_data)}

    def _calculate_h_divergence(self, subgroup_metrics):
        """
        Calculates 'h-divergence' (here, simply the absolute difference from base rates)
        for key metrics: FPR and FNR.
        """
        divergences = {}
        if self.base_rates['FPR'] > 0: # Avoid division by zero for relative divergence
            divergences['FPR_h_div'] = abs(subgroup_metrics['FPR'] - self.base_rates['FPR']) / self.base_rates['FPR']
        else:
             divergences['FPR_h_div'] = abs(subgroup_metrics['FPR'] - self.base_rates['FPR']) # Absolute difference if base is zero

        if self.base_rates['FNR'] > 0:
            divergences['FNR_h_div'] = abs(subgroup_metrics['FNR'] - self.base_rates['FNR']) / self.base_rates['FNR']
        else:
            divergences['FNR_h_div'] = abs(subgroup_metrics['FNR'] - self.base_rates['FNR'])

        # You could also add a combined divergence or other metrics
        divergences['Combined_h_div'] = divergences['FPR_h_div'] + divergences['FNR_h_div']
        return divergences

    def _bayesian_significance_test(self, subgroup_metric_value, base_metric_value, subgroup_count, total_count):
        """
        A simplified conceptual Bayesian-like significance test.
        Here, we use a Z-test for proportions difference for illustration purposes,
        which is a frequentist approach, but serves the purpose of 'significance'.
        A full Bayesian approach would involve posterior distributions.
        """
        # This is a simplification. For actual Bayesian significance, one would compare
        # posterior distributions of metrics in subgroup vs. overall.
        # Here, we'll use a simple proportion difference test for illustration.
        # Example: comparing FPRs

        # Assume a standard error for the difference in proportions
        # This requires counts, not just rates. Let's assume subgroup_metric_value
        # is based on 'err_count_subgroup' and subgroup_count for 'total_relevant_subgroup'
        # and base_metric_value is based on 'err_count_base' and total_relevant_base

        # Let's just compare if the subgroup metric is 'sufficiently' different given its size
        # A simpler heuristic for this example: if the absolute difference is large AND subgroup size is reasonable.
        abs_diff = abs(subgroup_metric_value - base_metric_value)
        if subgroup_count < 30: # Heuristic for small sample size
            return False
        
        # A simple threshold for significant difference for this example
        # In a real scenario, this would be a p-value or Bayes Factor from a proper statistical test.
        return abs_diff > (2 * math.sqrt(base_metric_value * (1 - base_metric_value) / subgroup_count))

    def _find_frequent_itemsets(self, evaluated_data):
        """
        Simplified Frequent Pattern Mining to find frequent combinations of features.
        This is a brute-force approach for illustration, not efficient for large datasets.
        Identifies single-item and two-item frequent patterns.
        """
        features = ['age_group', 'gender', 'pre_existing_condition', 'treatment_protocol']
        all_items = set()
        for patient in evaluated_data:
            for feature in features:
                all_items.add(f"{feature}={patient[feature]}")

        frequent_itemsets = {}

        # Single itemsets
        for item in all_items:
            count = sum(1 for patient in evaluated_data if item.split('=')[1] == patient[item.split('=')[0]])
            if count / len(evaluated_data) >= self.min_support:
                frequent_itemsets[frozenset([item])] = count

        # Two-item itemsets (combinations of items)
        for itemset_size in range(2, 3): # Only up to 2 for simplicity
            current_level_itemsets = list(frequent_itemsets.keys())
            for i in range(len(current_level_itemsets)):
                for j in range(i + 1, len(current_level_itemsets)):
                    itemset1 = current_level_itemsets[i]
                    itemset2 = current_level_itemsets[j]
                    # Only combine if they share all but one item (Apriori candidate generation principle, simplified)
                    combined_itemset = itemset1.union(itemset2)
                    if len(combined_itemset) == itemset_size:
                        count = 0
                        for patient in evaluated_data:
                            match = True
                            for item in combined_itemset:
                                feature, value = item.split('=')
                                if patient[feature] != value:
                                    match = False
                                    break
                            if match:
                                count += 1
                        if count / len(evaluated_data) >= self.min_support:
                            frequent_itemsets[combined_itemset] = count

        return frequent_itemsets

    def _get_subgroup_data(self, data, itemset):
        """
        Filters the dataset to get records belonging to a specific itemset subgroup.
        """
        subgroup_data = []
        for patient in data:
            match = True
            for item in itemset:
                feature, value = item.split('=')
                if patient[feature] != value:
                    match = False
                    break
            if match:
                subgroup_data.append(patient)
        return subgroup_data

    def _calculate_local_item_contribution(self, itemset, subgroup_metrics, base_metrics):
        """
        Conceptual local item contribution (Shapley-like).
        For simplicity, this attributes contribution based on how much each item
        in the itemset deviates a metric from the base.
        A full Shapley implementation requires evaluating subsets and permutations.
        """
        contributions = {}
        for item in itemset:
            # Heuristic: How much does this item's presence likely contribute to the divergence?
            # In a real scenario, this would involve comparing metrics with/without the item
            # or its interaction within the itemset using a proper Shapley kernel.
            contributions[item] = {
                'FPR_contribution': subgroup_metrics['FPR_h_div'] / len(itemset),
                'FNR_contribution': subgroup_metrics['FNR_h_div'] / len(itemset)
            }
        return contributions

    def _calculate_global_item_divergence(self, all_divergent_subgroups):
        """
        Calculates 'global item divergence' by aggregating contributions across
        all identified divergent subgroups.
        """
        global_contributions = {}
        for subgroup_info in all_divergent_subgroups:
            for item, contribs in subgroup_info['local_contributions'].items():
                if item not in global_contributions:
                    global_contributions[item] = {'FPR_total_div': 0.0, 'FNR_total_div': 0.0}
                global_contributions[item]['FPR_total_div'] += contribs['FPR_contribution']
                global_contributions[item]['FNR_total_div'] += contribs['FNR_contribution']
        return global_contributions

    def _identify_corrective_items(self, global_item_divergence):
        """
        Identifies 'corrective items' - items whose absence or modification
        might reduce divergence. This is highly conceptual here.
        """
        # This would typically involve domain knowledge or counterfactual explanations.
        # For this example, we'll assume items with high negative correlation to divergence
        # (i.e., when they are present, divergence is *lower* than expected) are corrective.
        # Or, items that, if removed from a divergent subgroup, reduce divergence significantly.
        # Here, a placeholder: items with very low global divergence contribution.
        corrective_candidates = []
        for item, div_scores in global_item_divergence.items():
            if div_scores['FPR_total_div'] < 0.1 and div_scores['FNR_total_div'] < 0.1: # Arbitrary threshold
                corrective_candidates.append(item)
        return corrective_candidates

    def _prune_redundant_subgroups(self, divergent_subgroups):
        """
        Redundancy pruning for summarization.
        Removes subgroups that are subsets of other more divergent or larger subgroups.
        """
        pruned_subgroups = []
        divergent_subgroups.sort(key=lambda x: x['divergence']['Combined_h_div'], reverse=True)

        for i, sg1 in enumerate(divergent_subgroups):
            is_redundant = False
            for j, sg2 in enumerate(divergent_subgroups):
                if i != j and sg2['itemset'].issuperset(sg1['itemset']) and \
                   sg2['divergence']['Combined_h_div'] >= sg1['divergence']['Combined_h_div']:
                    # sg1 is a subset of sg2 and sg2 is at least as divergent
                    is_redundant = True
                    break
            if not is_redundant:
                pruned_subgroups.append(sg1)
        return pruned_subgroups

    def audit_model(self, num_patients=1000):
        """
        Main function to audit the black-box model for fairness and divergent behaviors.
        """
        print("\n--- Starting Healthcare Model Fairness Auditor ---")

        print("1. Simulating Healthcare Data...")
        raw_data = self._simulate_healthcare_data(num_patients)
        print(f"   Generated {len(raw_data)} patient records.")

        print("2. Evaluating Model on Data...")
        evaluated_data = self._evaluate_model_on_data(raw_data)

        print("3. Calculating Base Rates (Overall Model Metrics)...")
        overall_metrics = self._calculate_metrics(evaluated_data)
        self.base_rates = overall_metrics
        print(f"   Overall FPR: {self.base_rates['FPR']:.4f}, FNR: {self.base_rates['FNR']:.4f}, Accuracy: {self.base_rates['Accuracy']:.4f}")

        print("4. Finding Frequent Itemsets (Subgroups)...")
        frequent_itemsets = self._find_frequent_itemsets(evaluated_data)
        print(f"   Found {len(frequent_itemsets)} frequent itemsets (subgroups).")

        divergent_subgroups = []
        print("5. Analyzing Subgroup Divergence and Significance...")
        for itemset, count in frequent_itemsets.items():
            subgroup_data = self._get_subgroup_data(evaluated_data, itemset)
            if len(subgroup_data) == 0: # Should not happen if itemset is frequent
                continue

            subgroup_metrics = self._calculate_metrics(subgroup_data)
            h_divergence = self._calculate_h_divergence(subgroup_metrics)

            # Check significance for FPR and FNR divergences
            is_fpr_significant = self._bayesian_significance_test(
                subgroup_metrics['FPR'], self.base_rates['FPR'],
                sum(1 for d in subgroup_data if d['actual_outcome'] == 0), # Total actual negatives in subgroup
                sum(1 for d in evaluated_data if d['actual_outcome'] == 0)  # Total actual negatives overall
            )
            is_fnr_significant = self._bayesian_significance_test(
                subgroup_metrics['FNR'], self.base_rates['FNR'],
                sum(1 for d in subgroup_data if d['actual_outcome'] == 1), # Total actual positives in subgroup
                sum(1 for d in evaluated_data if d['actual_outcome'] == 1)  # Total actual positives overall
            )

            if (is_fpr_significant and h_divergence['FPR_h_div'] > 0.1) or \
               (is_fnr_significant and h_divergence['FNR_h_div'] > 0.1): # Also requiring a threshold for divergence
                
                local_contributions = self._calculate_local_item_contribution(itemset, h_divergence, self.base_rates)

                divergent_subgroups.append({
                    'itemset': itemset,
                    'count': count,
                    'subgroup_metrics': subgroup_metrics,
                    'divergence': h_divergence,
                    'is_fpr_significant': is_fpr_significant,
                    'is_fnr_significant': is_fnr_significant,
                    'local_contributions': local_contributions
                })
        print(f"   Found {len(divergent_subgroups)} potentially divergent subgroups.")

        print("6. Pruning Redundant Divergent Subgroups...")
        pruned_divergent_subgroups = self._prune_redundant_subgroups(divergent_subgroups)
        print(f"   {len(pruned_divergent_subgroups)} non-redundant divergent subgroups remaining.")

        print("7. Calculating Global Item Divergence...")
        global_item_divergence = self._calculate_global_item_divergence(pruned_divergent_subgroups)

        print("8. Identifying Corrective Items (Conceptual)...")
        corrective_items = self._identify_corrective_items(global_item_divergence)

        print("\n--- Audit Results ---")
        print("\nOverall Model Performance (Base Rates):")
        for metric, value in self.base_rates.items():
            print(f"  {metric}: {value:.4f}")

        print("\nDivergent Subgroups (Pruned):")
        if not pruned_divergent_subgroups:
            print("  No significantly divergent subgroups found.")
        for sg in pruned_divergent_subgroups:
            print(f"  Subgroup: {', '.join(sg['itemset'])}")
            print(f"    Count: {sg['count']} ({sg['count']/num_patients:.2%})")
            print(f"    Subgroup Metrics - FPR: {sg['subgroup_metrics']['FPR']:.4f}, FNR: {sg['subgroup_metrics']['FNR']:.4f}")
            print(f"    Divergence (vs. Base) - FPR_h_div: {sg['divergence']['FPR_h_div']:.4f} (Sig: {sg['is_fpr_significant']}), FNR_h_div: {sg['divergence']['FNR_h_div']:.4f} (Sig: {sg['is_fnr_significant']})")
            print(f"    Local Item Contributions (FPR/FNR): {sg['local_contributions']}")
            print("    ---")

        print("\nGlobal Item Divergence:")
        if not global_item_divergence:
            print("  No global item divergence calculated.")
        for item, scores in global_item_divergence.items():
            print(f"  Item '{item}': Total FPR Divergence: {scores['FPR_total_div']:.4f}, Total FNR Divergence: {scores['FNR_total_div']:.4f}")

        print("\nConceptual Corrective Items:")
        if not corrective_items:
            print("  No conceptual corrective items identified.")
        else:
            for item in corrective_items:
                print(f"  - {item}")

        print("\n--- Audit Complete ---")
        return {
            "overall_metrics": self.base_rates,
            "divergent_subgroups": pruned_divergent_subgroups,
            "global_item_divergence": global_item_divergence,
            "corrective_items": corrective_items
        }

# --- Example Usage ---

# 1. Define a simulated black-box model prediction function
def simulated_blackbox_model(patient):
    """
    A dummy black-box model that makes predictions based on patient features.
    This model has a bias: it performs poorly for 'senior' males with 'heart_disease'
    and 'protocol_C', leading to higher FNR (missing positive cases).
    """
    # Default prediction: often 0 (no disease progression)
    prediction = 0

    # Simulate some complexity/bias
    if patient['age_group'] == 'adult' and patient['gender'] == 'female' and patient['pre_existing_condition'] == 'none':
        if random.random() < 0.3: # Higher chance of predicting 1 (false positive risk for this group)
            prediction = 1
    elif patient['age_group'] == 'senior' and patient['gender'] == 'male' and patient['pre_existing_condition'] == 'heart_disease' and patient['treatment_protocol'] == 'protocol_C':
        if random.random() < 0.7: # High chance of predicting 0 even if actual_outcome is 1 (high FNR risk for this group)
            prediction = 0
        else:
            prediction = 1 # Sometimes correct
    elif patient['actual_outcome'] == 1 and random.random() < 0.1: # Catch some actual positives generally
        prediction = 1
    elif patient['actual_outcome'] == 0 and random.random() < 0.05: # Some false positives generally
        prediction = 1
    
    return prediction

# 2. Instantiate and run the auditor
if __name__ == "__main__":
    auditor = HealthcareModelFairnessAuditor(model_predict_fn=simulated_blackbox_model, min_support=0.01, significance_threshold=0.05)
    audit_results = auditor.audit_model(num_patients=2000)

    # You can further process audit_results here
    # For instance, saving to a file or visualizing specific subgroups.
