import pandas as pd

data = pd.read_csv("unverified_weighted_predictions.csv")

uncertain_sample_count = 100

# Filter out samples where the weighted prediction is 'Other'
data = data[data['weighted_pred']!='Other']

# Calculate uncertainty as 1 - (weighted_score / 3) to normalize it to the range [0, 1]
data["uncertainty"] = 1 - (data["weighted_score"]/3)

data = data.sort_values(by="uncertainty", ascending=False)
data['rank'] = range(1, len(data) + 1)

labeled_data = pd.read_csv("/home/hasinthaka/Documents/Projects/AI/Pattern Mining/pipeline/notebooks/result/metadata.csv")

labeled_df = labeled_data['label'].value_counts().reset_index()
labeled_df.columns = ['label', 'count']
labeled_df['label_count_rank'] = labeled_df['count'].rank(method='first', ascending=False)


merged_df = data.merge(
    labeled_df[["label", "label_count_rank"]], 
    how="left", 
    left_on="weighted_pred", 
    right_on="label"
)

merged_df['final_rank'] = merged_df['rank'] + merged_df['label_count_rank']
merged_df = merged_df.sort_values("final_rank", ascending=True)

top_k_uncertain_samples = merged_df.head(uncertain_sample_count)
top_k_uncertain_samples.to_csv("top_k_uncertain_samples.csv", index=False)