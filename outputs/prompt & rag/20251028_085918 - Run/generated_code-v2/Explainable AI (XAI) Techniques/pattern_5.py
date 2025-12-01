import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split

def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)
    data = {
        "Fever": np.random.randint(0, 2, num_samples), # 0 or 1
        "Cough": np.random.randint(0, 2, num_samples),
        "Sore_Throat": np.random.randint(0, 2, num_samples),
        "Fatigue": np.random.randint(0, 2, num_samples),
        "Age": np.random.randint(10, 80, num_samples),
        "Headache": np.random.randint(0, 2, num_samples),
        "Muscle_Aches": np.random.randint(0, 2, num_samples),
    }
    df = pd.DataFrame(data)

    # Simulate Flu_Risk based on interpretable rules
    # Rule 1: High risk if Fever, Cough, Sore_Throat are all present (classic flu symptoms)
    # Rule 2: Moderate risk if Age > 60 and Fatigue and Muscle_Aches
    # Rule 3: Low risk otherwise
    df["Flu_Risk"] = 0 # Default to low risk
    df.loc[(df["Fever"] == 1) & (df["Cough"] == 1) & (df["Sore_Throat"] == 1), "Flu_Risk"] = 2 # High risk
    df.loc[(df["Age"] > 60) & (df["Fatigue"] == 1) & (df["Muscle_Aches"] == 1), "Flu_Risk"] = 1 # Moderate risk
    
    # Ensure some overlap and complexity for the tree
    df.loc[(df["Fever"] == 1) & (df["Headache"] == 1) & (df["Flu_Risk"] == 0), "Flu_Risk"] = 1 # Moderate risk for some fever+headache cases not covered by other rules

    return df

def train_model(df):
    X = df.drop("Flu_Risk", axis=1)
    y = df["Flu_Risk"]
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    return model, feature_names

def get_decision_path_explanation(tree_model, feature_names, sample_data):
    # Ensure sample_data is a 2D array for prediction/path
    sample_data_2d = sample_data.reshape(1, -1)
    
    node_indicator = tree_model.decision_path(sample_data_2d)
    leaf_id = tree_model.apply(sample_data_2d)

    explanation = []
    current_node = 0 # Start at the root
    path_description = []

    while current_node != leaf_id[0]:
        feature_idx = tree_model.tree_.feature[current_node]
        threshold = tree_model.tree_.threshold[current_node]
        
        if feature_idx == -2: # Leaf node reached (should not happen in loop before current_node == leaf_id[0])
            break

        feature_name = feature_names[feature_idx]
        sample_value = sample_data[feature_idx]

        if sample_value <= threshold:
            path_description.append(f"If {feature_name} <= {threshold:.2f} (Patient has {feature_name} = {sample_value})")
            current_node = tree_model.tree_.children_left[current_node]
        else:
            path_description.append(f"If {feature_name} > {threshold:.2f} (Patient has {feature_name} = {sample_value})")
            current_node = tree_model.tree_.children_right[current_node]
    
    return path_description

def main():
    print("Initializing Interpretable Medical Diagnosis Assistant...")
    df = generate_synthetic_data()
    model, feature_names = train_model(df)
    
    risk_mapping = {0: "Low Risk", 1: "Moderate Risk", 2: "High Risk"}

    print("\nModel trained successfully. Here are the general decision rules:\n")
    tree_rules = export_text(model, feature_names=feature_names)
    print(tree_rules)
    
    print("\n--- Enter Patient Data for Diagnosis (0 for No/False, 1 for Yes/True) ---")
    print("--- For Age, enter a numerical value ---")

    while True:
        try:
            patient_input = []
            patient_input.append(int(input("Fever (0/1): ")))
            patient_input.append(int(input("Cough (0/1): ")))
            patient_input.append(int(input("Sore Throat (0/1): ")))
            patient_input.append(int(input("Fatigue (0/1): ")))
            patient_input.append(int(input("Age: ")))
            patient_input.append(int(input("Headache (0/1): ")))
            patient_input.append(int(input("Muscle Aches (0/1): ")))

            patient_data = np.array(patient_input).reshape(1, -1)
            
            prediction_proba = model.predict_proba(patient_data)[0]
            predicted_risk_label = risk_mapping[np.argmax(prediction_proba)]
            
            print(f"\nDiagnosis: The patient has a {predicted_risk_label} of Flu Risk.")
            print("\nExplanation for this specific diagnosis:")
            explanation_path = get_decision_path_explanation(model, feature_names, patient_data[0])
            for i, rule in enumerate(explanation_path):
                print(f"  {i+1}. {rule}")
            print(f"  -> Final decision: {predicted_risk_label}")

        except ValueError:
            print("Invalid input. Please enter 0, 1, or a numerical age.")
        except Exception as e:
            print(f"An error occurred: {e}")
        
        another = input("\nDiagnose another patient? (yes/no): ").lower()
        if another != 'yes':
            break

    print("Exiting Medical Diagnosis Assistant. Goodbye!")

if __name__ == "__main__":
    main()