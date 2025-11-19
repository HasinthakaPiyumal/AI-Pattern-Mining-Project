import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

def load_and_preprocess_data():
    """
    Loads the breast cancer dataset, preprocesses it, and splits it into training and testing sets.
    Returns preprocessed data, target, feature names, and a scaler object.
    """
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=data.feature_names)

    print("Data loaded and preprocessed successfully.")
    return X_scaled_df, y, data.feature_names, scaler

