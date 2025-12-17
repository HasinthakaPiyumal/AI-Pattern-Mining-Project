import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Simulate a dataset (replace with actual data loading for a real application)
# For demonstration, we'll create a synthetic dataset resembling Pima Indians Diabetes Dataset features
np.random.seed(42)
num_samples = 1000

data = {
    'Pregnancies': np.random.randint(0, 17, num_samples),
    'Glucose': np.random.randint(70, 200, num_samples),
    'BloodPressure': np.random.randint(40, 120, num_samples),
    'SkinThickness': np.random.randint(10, 60, num_samples),
    'Insulin': np.random.randint(0, 850, num_samples),
    'BMI': np.random.uniform(15, 60, num_samples),
    'DiabetesPedigreeFunction': np.random.uniform(0.078, 2.42, num_samples),
    'Age': np.random.randint(21, 81, num_samples),
    'Outcome': np.random.randint(0, 2, num_samples) # 0 for no diabetes, 1 for diabetes
}
df = pd.DataFrame(data)

# Introduce some correlation for a more realistic 'Outcome'
df['Outcome'] = ((df['Glucose'] > 140) & (df['BMI'] > 30) & (df['Age'] > 40)).astype(int)
# Add some noise to the outcome to prevent perfect classification
num_flip = int(num_samples * 0.1)
flip_indices = np.random.choice(df.index, num_flip, replace=False)
df.loc[flip_indices, 'Outcome'] = 1 - df.loc[flip_indices, 'Outcome']


print("Dataset Head:")
print(df.head())
print("\nDataset Info:")
print(df.info())

# 2. Separate features (X) and target (y)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# 3. Initialize and train a Decision Tree Classifier
# Using a reasonable depth to avoid overfitting on a synthetic dataset
dtc = DecisionTreeClassifier(max_depth=5, random_state=42)
dtc.fit(X_train, y_train)

print("\nDecision Tree Classifier trained successfully.")

# 4. Make predictions on the test data
y_pred = dtc.predict(X_test)

# 5. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Visualize the Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Diabetes', 'Diabetes'], 
            yticklabels=['No Diabetes', 'Diabetes'])
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

# 6. Visualize the Decision Tree (optional)
plt.figure(figsize=(20, 10))
plot_tree(dtc, feature_names=X.columns.tolist(), class_names=['No Diabetes', 'Diabetes'], 
          filled=True, rounded=True, fontsize=10)
plt.title('Decision Tree Visualization')
plt.show()