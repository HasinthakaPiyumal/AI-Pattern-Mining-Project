import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 1. Simulate Data Ingestion
np.random.seed(42)
num_samples = 1000

data = {
    'Age': np.random.randint(20, 80, num_samples),
    'Gender': np.random.choice(['Male', 'Female'], num_samples),
    'BMI': np.random.uniform(18.0, 35.0, num_samples),
    'BloodPressure': np.random.randint(90, 180, num_samples),
    'Cholesterol': np.random.randint(150, 250, num_samples),
    'Smoker': np.random.choice([0, 1], num_samples),
    'ExerciseFrequency': np.random.randint(0, 7, num_samples), # days per week
    'FamilyHistory': np.random.choice([0, 1], num_samples),
    'Disease': np.random.choice([0, 1], num_samples, p=[0.7, 0.3]) # 0: No Disease, 1: Disease
}
df = pd.DataFrame(data)

# Introduce some correlation for 'Disease'
df.loc[df['Age'] > 60, 'Disease'] = np.random.choice([0, 1], sum(df['Age'] > 60), p=[0.4, 0.6])
df.loc[df['BMI'] > 30, 'Disease'] = np.random.choice([0, 1], sum(df['BMI'] > 30), p=[0.4, 0.6])
df.loc[df['Smoker'] == 1, 'Disease'] = np.random.choice([0, 1], sum(df['Smoker'] == 1), p=[0.4, 0.6])

X = df.drop('Disease', axis=1)
y = df['Disease']

# Define categorical and numerical features
categorical_features = ['Gender']
numerical_features = ['Age', 'BMI', 'BloodPressure', 'Cholesterol', 'Smoker', 'ExerciseFrequency', 'FamilyHistory']

# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Create a column transformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Model Training and Evaluation
models = {
    'Logistic Regression': LogisticRegression(random_state=42, solver='liblinear'),
    'Support Vector Machine': SVC(random_state=42, probability=True),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

results = {}
best_model = None
best_accuracy = 0
best_model_name = ""

for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    results[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc
    }

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = pipeline
        best_model_name = name

print("Model Evaluation Results:")
for name, metrics in results.items():
    print(f"\n{name}:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")

print(f"\nBest performing model: {best_model_name} with an accuracy of {best_accuracy:.4f}")

# 4. Prediction Interface (Example)
def predict_disease_likelihood(new_patient_data: dict, model_pipeline: Pipeline):
    new_df = pd.DataFrame([new_patient_data])
    prediction = model_pipeline.predict(new_df)[0]
    prediction_proba = model_pipeline.predict_proba(new_df)[:, 1][0]
    return {"prediction": int(prediction), "likelihood": prediction_proba}

# Example usage of the prediction interface
if best_model is not None:
    new_patient = {
        'Age': 65,
        'Gender': 'Female',
        'BMI': 31.5,
        'BloodPressure': 160,
        'Cholesterol': 230,
        'Smoker': 1,
        'ExerciseFrequency': 1,
        'FamilyHistory': 1
    }
    prediction_result = predict_disease_likelihood(new_patient, best_model)
    print(f"\nPrediction for new patient: {new_patient}")
    print(f"  Predicted Disease (0=No, 1=Yes): {prediction_result['prediction']}")
    print(f"  Likelihood of Disease: {prediction_result['likelihood']:.4f}")
else:
    print("No best model found. Training might have failed.")
