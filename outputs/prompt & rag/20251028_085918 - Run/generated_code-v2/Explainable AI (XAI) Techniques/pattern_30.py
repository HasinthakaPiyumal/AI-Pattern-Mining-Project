import pandas as pd
import numpy as np
from scipy.stats import beta

class DivergenceCalculator:
    """
    Calculates 'h-divergence' and its statistical significance for identifying
    subgroups with peculiar divergent behaviors in a black-box classification model.
    """
    def __init__(self, target_column: str, prediction_column: str):
        self.target_column = target_column
        self.prediction_column = prediction_column

    def _calculate_performance_metric(self, df: pd.DataFrame, metric: str) -> float:
        """
        Calculates a specified performance metric (e.g., FPR, FNR).
        Assumes binary classification (0/1).
        """
        if df.empty:
            return 0.0

        true_positives = ((df[self.prediction_column] == 1) & (df[self.target_column] == 1)).sum()
        true_negatives = ((df[self.prediction_column] == 0) & (df[self.target_column] == 0)).sum()
        false_positives = ((df[self.prediction_column] == 1) & (df[self.target_column] == 0)).sum()
        false_negatives = ((df[self.prediction_column] == 0) & (df[self.target_column] == 1)).sum()

        if metric == "FPR":
            if (false_positives + true_negatives) == 0: return 0.0
            return false_positives / (false_positives + true_negatives)
        elif metric == "FNR":
            if (false_negatives + true_positives) == 0: return 0.0
            return false_negatives / (false_negatives + true_positives)
        elif metric == "Accuracy":
            total = len(df)
            if total == 0: return 0.0
            return (true_positives + true_negatives) / total
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    def calculate_h_divergence(self,
                                subgroup_df: pd.DataFrame,
                                global_df: pd.DataFrame,
                                metric: str = "FPR") -> float:
        """
        Calculates the h-divergence for a given subgroup and a global dataset.
        h-divergence is the absolute difference in a performance metric.
        """
        subgroup_metric = self._calculate_performance_metric(subgroup_df, metric)
        global_metric = self._calculate_performance_metric(global_df, metric)
        return abs(subgroup_metric - global_metric)

    def _get_beta_parameters(self, successes: int, failures: int) -> tuple:
        """
        Returns parameters for a Beta distribution (alpha, beta).
        Using Jeffrey's prior: alpha = successes + 0.5, beta = failures + 0.5
        """
        return successes + 0.5, failures + 0.5

    def calculate_bayesian_significance(
        self,
        subgroup_df: pd.DataFrame,
        global_df: pd.DataFrame,
        metric: str = "FPR",
        confidence_level: float = 0.95
    ) -> dict:
        """
        Calculates Bayesian significance for the divergence of a metric.
        Compares the posterior distributions of the metric for the subgroup and global data.
        Returns the probability that the subgroup metric is greater/less than the global metric,
        and credible intervals.
        """
        if metric == "FPR":
            # For FPR: successes = False Positives, failures = True Negatives
            sub_s = ((subgroup_df[self.prediction_column] == 1) & (subgroup_df[self.target_column] == 0)).sum()
            sub_f = ((subgroup_df[self.prediction_column] == 0) & (subgroup_df[self.target_column] == 0)).sum()
            global_s = ((global_df[self.prediction_column] == 1) & (global_df[self.target_column] == 0)).sum()
            global_f = ((global_df[self.prediction_column] == 0) & (global_df[self.target_column] == 0)).sum()
        elif metric == "FNR":
            # For FNR: successes = False Negatives, failures = True Positives
            sub_s = ((subgroup_df[self.prediction_column] == 0) & (subgroup_df[self.target_column] == 1)).sum()
            sub_f = ((subgroup_df[self.prediction_column] == 1) & (subgroup_df[self.target_column] == 1)).sum()
            global_s = ((global_df[self.prediction_column] == 0) & (global_df[self.target_column] == 1)).sum()
            global_f = ((global_df[self.prediction_column] == 1) & (global_df[self.target_column] == 1)).sum()
        elif metric == "Accuracy":
            # For Accuracy: successes = Correct Predictions, failures = Incorrect Predictions
            sub_s = (subgroup_df[self.prediction_column] == subgroup_df[self.target_column]).sum()
            sub_f = (subgroup_df[self.prediction_column] != subgroup_df[self.target_column]).sum()
            global_s = (global_df[self.prediction_column] == global_df[self.target_column]).sum()
            global_f = (global_df[self.prediction_column] != global_df[self.target_column]).sum()
        else:
            raise ValueError(f"Unsupported metric for Bayesian significance: {metric}")

        if (sub_s + sub_f == 0) or (global_s + global_f == 0):
            return {"significant": False, "probability_gt": 0.5, "probability_lt": 0.5, "subgroup_credible_interval": (0.0, 0.0), "global_credible_interval": (0.0, 0.0)}

        alpha_sub, beta_sub = self._get_beta_parameters(sub_s, sub_f)
        alpha_global, beta_global = self._get_beta_parameters(global_s, global_f)

        # Sample from posterior distributions
        np.random.seed(42) # for reproducibility
        samples_sub = beta.rvs(alpha_sub, beta_sub, size=10000)
        samples_global = beta.rvs(alpha_global, beta_global, size=10000)

        # Probability that subgroup metric is greater than global metric
        probability_gt = (samples_sub > samples_global).mean()
        probability_lt = (samples_sub < samples_global).mean()

        # Credible intervals
        lower_bound = (1 - confidence_level) / 2
        upper_bound = 1 - lower_bound
        subgroup_ci = (np.quantile(samples_sub, lower_bound), np.quantile(samples_sub, upper_bound))
        global_ci = (np.quantile(samples_global, lower_bound), np.quantile(samples_global, upper_bound))

        # A simple criterion for significance: one metric's CI does not overlap with the other's mean,
        # or the probability of being greater/less is high (e.g., > 0.95 or < 0.05 for two-tailed)
        # More robust significance can be derived from the probability_gt/lt directly.
        significant = False
        if probability_gt > confidence_level or probability_lt > confidence_level:
            significant = True

        return {
            "significant": significant,
            "probability_gt": probability_gt, # Probability subgroup metric > global metric
            "probability_lt": probability_lt, # Probability subgroup metric < global metric
            "subgroup_credible_interval": subgroup_ci,
            "global_credible_interval": global_ci
        }


if __name__ == "__main__":
    # Example Usage:
    from data_simulator import generate_patient_data
    from mock_model import mock_black_box_predict

    # 1. Generate synthetic data and predictions
    patient_df = generate_patient_data(num_patients=5000)
    patient_df["predicted_outcome"] = mock_black_box_predict(patient_df.drop(columns=["treatment_outcome"]))

    # Rename for clarity in the calculator
    patient_df = patient_df.rename(columns={
        "treatment_outcome": "true_label",
        "predicted_outcome": "prediction"
    })

    # 2. Initialize the calculator
    calculator = DivergenceCalculator(target_column="true_label", prediction_column="prediction")

    # 3. Define a global dataset (all data)
    global_data = patient_df.copy()

    # 4. Define a subgroup (e.g., older patients with diabetes)
    subgroup_data_1 = patient_df[(patient_df["age"] > 70) & (patient_df["comorbidity_diabetes"] == 1)].copy()
    print(f"\nSubgroup 1 size: {len(subgroup_data_1)}")

    # 5. Calculate h-divergence for FPR
    h_div_fpr_1 = calculator.calculate_h_divergence(subgroup_data_1, global_data, metric="FPR")
    print(f"h-divergence (FPR) for Subgroup 1: {h_div_fpr_1:.4f}")

    # 6. Calculate Bayesian significance for FPR
    bayesian_sig_fpr_1 = calculator.calculate_bayesian_significance(subgroup_data_1, global_data, metric="FPR")
    print(f"Bayesian Significance (FPR) for Subgroup 1: {bayesian_sig_fpr_1}")

    # 7. Define another subgroup (e.g., African_American patients with Surgery Y)
    subgroup_data_2 = patient_df[(patient_df["ethnicity"] == "African_American") & (patient_df["treatment_type"] == "Surgery Y")].copy()
    print(f"\nSubgroup 2 size: {len(subgroup_data_2)}")

    # 8. Calculate h-divergence for FNR
    h_div_fnr_2 = calculator.calculate_h_divergence(subgroup_data_2, global_data, metric="FNR")
    print(f"h-divergence (FNR) for Subgroup 2: {h_div_fnr_2:.4f}")

    # 9. Calculate Bayesian significance for FNR
    bayesian_sig_fnr_2 = calculator.calculate_bayesian_significance(subgroup_data_2, global_data, metric="FNR")
    print(f"Bayesian Significance (FNR) for Subgroup 2: {bayesian_sig_fnr_2}")

    # Example with a non-divergent subgroup (likely)
    subgroup_data_3 = patient_df[(patient_df["age"] < 30) & (patient_df["gender"] == "Female")].copy()
    print(f"\nSubgroup 3 size: {len(subgroup_data_3)}")
    h_div_fpr_3 = calculator.calculate_h_divergence(subgroup_data_3, global_data, metric="FPR")
    print(f"h-divergence (FPR) for Subgroup 3: {h_div_fpr_3:.4f}")
    bayesian_sig_fpr_3 = calculator.calculate_bayesian_significance(subgroup_data_3, global_data, metric="FPR")
    print(f"Bayesian Significance (FPR) for Subgroup 3: {bayesian_sig_fpr_3}")
