import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import PartialDependenceDisplay

def generate_simulated_data(num_samples=1000):
    np.random.seed(42)
    data = {
        'Age': np.random.randint(20, 90, num_samples),
        'Num_Previous_Admissions': np.random.randint(0, 5, num_samples),
        'Chronic_Conditions': np.random.randint(0, 3, num_samples),
        'Length_of_Stay': np.random.randint(1, 30, num_samples),
        'Lab_Result_A': np.random.normal(100, 15, num_samples),
        'Lab_Result_B': np.random.normal(50, 10, num_samples),
        'readmitted': np.random.randint(0, 2, num_samples)
    }
    df = pd.DataFrame(data)
    
    df['readmitted'] = (
        (df['Age'] > 60).astype(int) +
        (df['Num_Previous_Admissions'] > 1).astype(int) +
        (df['Chronic_Conditions'] > 0).astype(int) +
        (df['Length_of_Stay'] > 10).astype(int)
    > 2).astype(int)
    
    return df

if __name__ == "__main__":
    df = generate_simulated_data()

    X = df.drop('readmitted', axis=1)
    y = df['readmitted']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    features_for_pdp = ['Age', 'Num_Previous_Admissions', 'Length_of_Stay', ('Age', 'Chronic_Conditions')]

    fig, ax = plt.subplots(figsize=(12, 8))
    display = PartialDependenceDisplay.from_estimator(
        model, X_train, features_for_pdp, kind="average", ax=ax, 
        feature_names=X.columns.tolist(), 
        target_imputation="auto", 
        percentiles=(0.05, 0.95)
    )
    plt.suptitle('Partial Dependence Plots for Patient Readmission Risk')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()