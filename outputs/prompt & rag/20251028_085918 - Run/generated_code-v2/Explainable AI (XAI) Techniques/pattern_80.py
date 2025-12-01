import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

def load_patient_data():
    np.random.seed(42)
    n_samples = 1000
    X, y = make_classification(n_samples=n_samples, n_features=4, n_informative=3, n_redundant=0, random_state=42, n_classes=2)
    
    df = pd.DataFrame(X, columns=['Age', 'Cholesterol', 'BloodPressure', 'BMI'])
    df['Age'] = np.random.randint(20, 80, n_samples)
    df['Cholesterol'] = np.random.normal(200, 30, n_samples)
    df['BloodPressure'] = np.random.normal(120, 15, n_samples)
    df['BMI'] = np.random.normal(25, 5, n_samples)
    df['HeartDisease'] = y
    
    return df

def train_disease_prediction_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def generate_ice_data(model, data, feature_name, num_grid_points=100, instance_indices=None):
    if instance_indices is None:
        instance_indices = data.index.to_list()

    feature_min = data[feature_name].min()
    feature_max = data[feature_name].max()
    feature_grid = np.linspace(feature_min, feature_max, num_grid_points)

    ice_curves = []

    for i in instance_indices:
        original_instance = data.iloc[i].copy()
        instance_data_for_prediction = pd.DataFrame([original_instance] * num_grid_points)
        instance_data_for_prediction[feature_name] = feature_grid
        
        # Exclude the target column if it's present in the data passed to generate_ice_data
        features_for_prediction = instance_data_for_prediction.drop(columns=['HeartDisease'], errors='ignore')
        
        predictions = model.predict_proba(features_for_prediction)[:, 1] # Probability of positive class
        
        for j, val in enumerate(feature_grid):
            ice_curves.append({
                "instance_id": i,
                feature_name: val,
                "prediction": predictions[j]
            })
            
    return pd.DataFrame(ice_curves)

def plot_ice_curves(ice_data, feature_name, title='ICE Plot'):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=ice_data, x=feature_name, y="prediction", hue="instance_id", palette="tab10", legend=False, alpha=0.7)
    
    # Optionally add a PDP-like average curve
    # avg_curve = ice_data.groupby(feature_name)['prediction'].mean().reset_index()
    # sns.lineplot(data=avg_curve, x=feature_name, y="prediction", color='red', linestyle='--', linewidth=2, label='Average Prediction')

    plt.title(title)
    plt.xlabel(f'{feature_name} Value')
    plt.ylabel('Predicted Probability of Heart Disease')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def main():
    print("Loading patient data...")
    df = load_patient_data()
    
    X = df.drop(columns=['HeartDisease'])
    y = df['HeartDisease']
    
    print("Training disease prediction model...")
    model = train_disease_prediction_model(X, y)
    
    feature_of_interest = 'Cholesterol'
    # Select a few random instances to visualize, or choose specific ones
    num_instances_to_plot = 10
    np.random.seed(42) # for reproducibility of instance selection
    instance_indices = np.random.choice(X.index, num_instances_to_plot, replace=False)
    
    print(f"Generating ICE data for feature: {feature_of_interest} for {num_instances_to_plot} instances...")
    ice_data = generate_ice_data(model, df, feature_of_interest, instance_indices=instance_indices)
    
    print("Plotting ICE curves...")
    plot_ice_curves(ice_data, feature_of_interest, title=f'ICE Plot for {feature_of_interest} on Heart Disease Prediction')
    print("ICE plot displayed.")

if __name__ == "__main__":
    main()