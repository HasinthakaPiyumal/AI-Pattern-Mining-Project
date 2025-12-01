import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import dice_ml

def generate_synthetic_loan_data(num_samples=1000):
    np.random.seed(42)
    data = {
        'credit_score': np.random.randint(300, 850, num_samples),
        'income': np.random.randint(30000, 150000, num_samples),
        'debt_to_income_ratio': np.random.uniform(0.1, 0.6, num_samples),
        'employment_status': np.random.choice(['employed', 'self-employed', 'unemployed'], num_samples, p=[0.7, 0.2, 0.1]),
        'loan_amount': np.random.randint(5000, 100000, num_samples)
    }
    df = pd.DataFrame(data)

    # Simple logic for 'approved' status
    # Higher credit score, higher income, lower DTI generally lead to approval
    df['approved'] = ((df['credit_score'] >= 650) & 
                      (df['income'] >= 50000) & 
                      (df['debt_to_income_ratio'] <= 0.4) & 
                      (df['employment_status'] != 'unemployed')).astype(int)

    # Add some noise to make it more realistic
    noise_factor = np.random.rand(num_samples) * 0.3
    df['approved'] = np.where(noise_factor < 0.1, 1 - df['approved'], df['approved'])
    
    return df

def train_loan_model(df):
    X = df.drop('approved', axis=1)
    y = df['approved']

    categorical_features = ['employment_status']
    numerical_features = ['credit_score', 'income', 'debt_to_income_ratio', 'loan_amount']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    model.fit(X, y)
    return model, preprocessor, X.columns

def get_prediction(applicant_data, model):
    applicant_df = pd.DataFrame([applicant_data])
    prediction = model.predict(applicant_df)
    return prediction[0]

def main():
    print("Generating synthetic loan data...")
    loan_data = generate_synthetic_loan_data(2000)
    
    print("Training loan prediction model...")
    model, preprocessor, original_features = train_loan_model(loan_data)

    # Prepare the training data for DiCE
    # Need to transform the entire dataset to get feature names for DiCE
    X_encoded = pd.DataFrame(preprocessor.transform(loan_data.drop('approved', axis=1)),
                             columns=preprocessor.get_feature_names_out())
    
    # Rename DiCE feature names to be more readable if needed
    # DiCE expects the feature names to match what the model was trained on
    feature_names_for_dice = list(original_features)
    
    # Example applicant (likely to be denied for demonstration)
    denied_applicant = {
        'credit_score': 550,
        'income': 40000,
        'debt_to_income_ratio': 0.55,
        'employment_status': 'unemployed',
        'loan_amount': 20000
    }
    
    print(f"\nApplicant's original data: {denied_applicant}")
    
    prediction = get_prediction(denied_applicant, model)
    print(f"Initial loan prediction: {'Approved' if prediction == 1 else 'Denied'}")

    if prediction == 0:  # If denied, generate counterfactuals
        print("\nGenerating counterfactual explanations...")

        # DiCE requires a callable model function and the training data.
        # We need to wrap the pipeline's predict_proba for DiCE.
        def predict_proba_wrapper(X_input):
            # DiCE provides dataframes for input, ensure consistent column order and one-hot encoding
            # For DiCE to work correctly, the input to the wrapper should be in the original feature format.
            # The preprocessor within the model pipeline will handle the encoding.
            return model.predict_proba(X_input)

        # Create a DiCE data object. Use the original dataframe before one-hot encoding for easier interpretation
        # DiCE handles the encoding internally if we pass the preprocessor in the model wrapper
        d = dice_ml.Data(dataframe=loan_data, 
                         continuous_features=['credit_score', 'income', 'debt_to_income_ratio', 'loan_amount'], 
                         outcome_name='approved')
        
        # Create a DiCE model object. The model should be the pipeline.
        m = dice_ml.Model(model=model, backend='sklearn')

        # Create a DiCE explainer object
        exp = dice_ml.Dice(d, m, method='kdtree') # Using kdtree for efficiency with numerical data

        # Convert the denied applicant to a DataFrame for DiCE
        query_instance = pd.DataFrame([denied_applicant])

        # Generate counterfactuals. Target class 1 means we want to find changes for approval.
        dice_exp = exp.generate_counterfactuals(query_instance, 
                                                total_CFs=3, 
                                                desired_class="opposite",
                                                features_to_vary=['credit_score', 'income', 'debt_to_income_ratio'])

        # Print the counterfactuals
        print("\nCounterfactual Explanations (Minimal changes to get approved):")
        explanation_df = dice_exp.cf_examples_list[0].final_cfs_df
        
        # Filter out features that were not varied, or only show the changes
        original_df = pd.DataFrame([denied_applicant])
        
        print("\nOriginal Denied Application:")
        print(original_df)
        print("\nSuggested Changes for Approval:")
        
        for i, row in explanation_df.iterrows():
            print(f"\nCounterfactual {i+1}:")
            changes = {}
            for feature in original_df.columns:
                original_value = original_df[feature].iloc[0]
                cf_value = row[feature]
                if original_value != cf_value:
                    if isinstance(original_value, (int, float)) and isinstance(cf_value, (int, float)):
                        change_amount = cf_value - original_value
                        changes[feature] = f"Change by {change_amount:+.2f} (from {original_value} to {cf_value:.2f})"
                    else:
                        changes[feature] = f"Change from '{original_value}' to '{cf_value}'"
            
            if changes:
                for feature, change_desc in changes.items():
                    print(f"  - {feature}: {change_desc}")
                print(f"  - Predicted outcome: {'Approved' if row['approved'] == 1 else 'Denied'}")
            else:
                print("  No significant changes found to flip the prediction with the specified parameters.")

    else:
        print("No counterfactual explanation needed as the loan is approved.")

if __name__ == "__main__":
    main()