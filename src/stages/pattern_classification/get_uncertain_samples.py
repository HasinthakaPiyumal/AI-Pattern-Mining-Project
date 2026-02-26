import pandas as pd

data = pd.read_csv("unverified_weighted_predictions.csv")

uncertain_sample_count = 100

data["uncertainty"] = 1 - (data["weighted_score"]/3)
data = data.sort_values(by="uncertainty", ascending=False)

top_k_uncertain_samples = data.head(uncertain_sample_count)

top_k_uncertain_samples.to_csv("top_k_uncertain_samples.csv", index=False)