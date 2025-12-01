import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from alibi.explainers import CounterFactual

# 1. Data Simulation
pd.set_option("display.max_columns", None)
np.random.seed(42)

num_samples = 1000
data = {
    "credit_score": np.random.randint(300, 850, num_samples),
    "income": np.random.randint(20000, 150000, num_samples),
    "debt_to_income_ratio": np.random.uniform(0.1, 0.6, num_samples),
    "employment_status": np.random.choice(["Employed", "Self-Employed", "Unemployed", "Retired"], num_samples, p=[0.6, 0.2, 0.1, 0.1]),
    "loan_amount": np.random.randint(5000, 500000, num_samples),
}
df = pd.DataFrame(data)

# Simple rule for loan approval for synthetic data
df["loan_approved"] = (
    (df["credit_score"] > 650) & 
    (df["income"] > 40000) & 
    (df["debt_to_income_ratio"] < 0.4) & 
    (df["employment_status"].isin(["Employed", "Self-Employed"]))
).astype(int)

X = df.drop("loan_approved", axis=1)
y = df["loan_approved"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Feature Preprocessing
numerical_features = ["credit_score", "income", "debt_to_income_ratio", "loan_amount"]
categorical_features = ["employment_status"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ],
    remainder="passthrough"
)

# 2. Loan Prediction Model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

model.fit(X_train, y_train)

# Select a denied instance for explanation
denied_applications = X_test[model.predict(X_test) == 0]
if not denied_applications.empty:
    instance_to_explain = denied_applications.sample(1, random_state=42)
else:
    print("No denied applications in the test set to explain.")
    exit()

print(f"Original Denied Application:\n{instance_to_explain}")

# Predict function wrapper for Alibi
def predict_fn(x):
    x_df = pd.DataFrame(x, columns=X_train.columns) # Ensure columns match original data
    return model.predict_proba(x_df)

# 3. Counterfactual Explanation Generator
c_init = 1.0
c_steps = 10
max_iterations = 1000
feature_range = (X_train.min().values, X_train.max().values)

# Define feature ranges for counterfactual generation (can be more sophisticated)
# For simplicity, we'll use the min/max from training data for numerical features
# Categorical features are handled by one-hot encoding.
feature_ranges_alibi = {}
for col in numerical_features:
    feature_ranges_alibi[col] = (X_train[col].min(), X_train[col].max())

# Create a list of feature names that matches the preprocessor output order
processed_feature_names = numerical_features + list(model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features))

# Ensure `target_proba` is 1 for approval
explainer = CounterFactual(
    predict_fn,
    instance_to_explain.values,
    target_proba=1.0,
    target_class=1,
    feature_range=(X_train.values.min(), X_train.values.max()), # This range needs careful consideration for real data
    # For real applications, this feature_range would be more granular or defined per feature
    # For alibi, feature_range applies to the *transformed* features, or is used as a generic bound
    # Let's pass the preprocessor itself to Alibi for better handling of feature ranges
    # However, for simplicity here, we'll make a more direct call for the initial values
    c_init=c_init,
    c_steps=c_steps,
    max_iterations=max_iterations,
    # For categorical features, Alibi-CF needs to know which columns are categorical and their possible values
    # This is often done by passing `cat_vars` and `cat_perms` to the explainer
    # For this simple example, we'll assume the preprocessor handles the encoding and the search space is continuous
    # over the transformed features, and then we'll reverse-map.
)

# Alibi's CounterFactual expects input as numpy array for initialisation
explanation = explainer.explain(instance_to_explain.values)

if explanation.cf is not None:
    cf_instance_transformed = explanation.cf["X"]
    
    # Inverse transform the counterfactual instance
    # This part can be tricky as inverse_transform might not be directly available for ColumnTransformer + OneHotEncoder
    # We'll manually reconstruct for demonstration
    
    # First, get the preprocessor without the classifier
    preprocessor_only = model.named_steps['preprocessor']
    
    # Get the inverse transformed numerical features
    cf_num_transformed = cf_instance_transformed[0, :len(numerical_features)]
    cf_numerical_features = preprocessor_only.named_transformers_['num'].inverse_transform(cf_num_transformed.reshape(1, -1))
    
    # Get the inverse transformed categorical features
    cf_cat_transformed = cf_instance_transformed[0, len(numerical_features):]
    # Find the index of the highest value in each one-hot encoded block
    cat_feature_index = np.argmax(cf_cat_transformed)
    original_categories = preprocessor_only.named_transformers_['cat'].categories_[0]
    cf_categorical_feature = original_categories[cat_feature_index]
    
    cf_data = {
        numerical_features[i]: cf_numerical_features[0, i] for i in range(len(numerical_features))
    }
    cf_data[categorical_features[0]] = cf_categorical_feature

    cf_df = pd.DataFrame([cf_data], columns=X_train.columns)

    print(f"\nCounterfactual Explanation (Target: Approved):\n{cf_df}")

    print("\nMinimal Changes Required:")
    for col in X_train.columns:
        original_val = instance_to_explain[col].values[0]
        cf_val = cf_df[col].values[0]
        if isinstance(original_val, (int, float)) and isinstance(cf_val, (int, float)): # For numerical features
            if not np.isclose(original_val, cf_val):
                print(f"  {col}: From {original_val:.2f} to {cf_val:.2f} (Change: {cf_val - original_val:.2f})")
        else:
            if original_val != cf_val:
                print(f"  {col}: From \'{original_val}\' to \'{cf_val}\'")
else:
    print("Could not find a counterfactual explanation.")
