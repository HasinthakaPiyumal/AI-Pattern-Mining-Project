import pandas as pd
import numpy as np
import joblib
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical


class DataProcessor:
    def __init__(self):
        self.preprocessor_structured = None
        self.image_size = (64, 64)

    def _create_dummy_structured_data(self, num_samples=100):
        data = {
            "age": np.random.randint(20, 80, num_samples),
            "gender": np.random.choice(["Male", "Female"], num_samples),
            "bmi": np.random.uniform(18.0, 35.0, num_samples),
            "cholesterol": np.random.uniform(150, 250, num_samples),
            "blood_pressure_systolic": np.random.uniform(100, 180, num_samples),
            "blood_pressure_diastolic": np.random.uniform(60, 120, num_samples),
            "smoking": np.random.choice([0, 1], num_samples, p=[0.7, 0.3]),
            "diabetes": np.random.choice([0, 1], num_samples, p=[0.8, 0.2]),
            "genetic_predisposition": np.random.choice([0, 1], num_samples, p=[0.9, 0.1]),
            "label_structured": np.random.choice([0, 1], num_samples, p=[0.6, 0.4])
        }
        return pd.DataFrame(data)

    def _create_dummy_image_data(self, num_samples=100):
        images = np.random.rand(num_samples, 128, 128, 1) * 255
        labels = np.random.choice([0, 1], num_samples, p=[0.5, 0.5])
        return images.astype(np.uint8), labels

    def preprocess_structured_data(self, df, fit=True):
        numerical_features = ["age", "bmi", "cholesterol", "blood_pressure_systolic", "blood_pressure_diastolic"]
        categorical_features = ["gender"]

        if fit:
            self.preprocessor_structured = ColumnTransformer(
                transformers=[
                    ("num", StandardScaler(), numerical_features),
                    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
                ],
                remainder="passthrough"
            )
            X_processed = self.preprocessor_structured.fit_transform(df)
        else:
            X_processed = self.preprocessor_structured.transform(df)

        return X_processed

    def preprocess_image(self, image_data):
        processed_images = []
        for img in image_data:
            resized_img = cv2.resize(img, self.image_size)
            normalized_img = resized_img / 255.0
            processed_images.append(normalized_img)
        return np.array(processed_images).reshape(-1, self.image_size[0], self.image_size[1], 1)


class ClassicalModelTrainer:
    def __init__(self, model_type="RandomForest"):
        if model_type == "LogisticRegression":
            self.model = LogisticRegression(random_state=42)
        elif model_type == "XGBoost":
            self.model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss")
        else:
            self.model = RandomForestClassifier(random_state=42)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba)
        }
        return metrics, confusion_matrix(y_test, y_pred)

    def predict(self, X_new):
        return self.model.predict(X_new), self.model.predict_proba(X_new)[:, 1]

    def save_model(self, filename="classical_model.joblib"):
        joblib.dump(self.model, filename)

    def load_model(self, filename="classical_model.joblib"):
        self.model = joblib.load(filename)


class DeepLearningModelTrainer:
    def __init__(self, input_shape=(64, 64, 1), num_classes=2):
        self.model = self._build_cnn_model(input_shape, num_classes)

    def _build_cnn_model(self, input_shape, num_classes):
        model = Sequential([
            Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(num_classes, activation="softmax")
        ])
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        return model

    def train(self, X_train, y_train, epochs=10, batch_size=32):
        self.model.fit(X_train, to_categorical(y_train, num_classes=2), epochs=epochs, batch_size=batch_size, verbose=0)

    def evaluate(self, X_test, y_test):
        loss, accuracy = self.model.evaluate(X_test, to_categorical(y_test, num_classes=2), verbose=0)
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = y_test

        metrics = {
            "accuracy": accuracy,
            "precision": precision_score(y_true, y_pred, average="weighted"),
            "recall": recall_score(y_true, y_pred, average="weighted"),
            "f1_score": f1_score(y_true, y_pred, average="weighted"),
            "roc_auc": roc_auc_score(y_true, y_pred_proba[:, 1]) if len(np.unique(y_true)) == 2 else np.nan
        }
        return metrics, confusion_matrix(y_true, y_pred)

    def predict(self, X_new):
        predictions = self.model.predict(X_new, verbose=0)
        return np.argmax(predictions, axis=1), predictions[:, 1]

    def save_model(self, filename="dl_model.h5"):
        self.model.save(filename)

    def load_model(self, filename="dl_model.h5"):
        self.model = tf.keras.models.load_model(filename)


class PredictionSystem:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.classical_trainer = ClassicalModelTrainer(model_type="XGBoost")
        self.dl_trainer = DeepLearningModelTrainer()
        self.trained_structured_model = False
        self.trained_image_model = False

    def run_training_workflow(self, num_structured_samples=1000, num_image_samples=500):
        # Structured Data Workflow
        print("\n--- Structured Data Workflow ---")
        structured_df = self.data_processor._create_dummy_structured_data(num_samples=num_structured_samples)
        X_structured = structured_df.drop("label_structured", axis=1)
        y_structured = structured_df["label_structured"]

        X_structured_processed = self.data_processor.preprocess_structured_data(X_structured, fit=True)
        X_struct_train, X_struct_test, y_struct_train, y_struct_test = train_test_split(
            X_structured_processed, y_structured, test_size=0.2, random_state=42
        )

        self.classical_trainer.train(X_struct_train, y_struct_train)
        metrics_structured, cm_structured = self.classical_trainer.evaluate(X_struct_test, y_struct_test)
        print("Classical Model Metrics:", metrics_structured)
        print("Classical Model Confusion Matrix:\n", cm_structured)
        self.classical_trainer.save_model("structured_diagnosis_model.joblib")
        joblib.dump(self.data_processor.preprocessor_structured, "structured_preprocessor.joblib")
        self.trained_structured_model = True

        # Image Data Workflow
        print("\n--- Image Data Workflow ---")
        raw_images, image_labels = self.data_processor._create_dummy_image_data(num_samples=num_image_samples)
        X_image_processed = self.data_processor.preprocess_image(raw_images)
        y_image_encoded = image_labels

        X_img_train, X_img_test, y_img_train, y_img_test = train_test_split(
            X_image_processed, y_image_encoded, test_size=0.2, random_state=42
        )

        self.dl_trainer.train(X_img_train, y_img_train, epochs=5)
        metrics_image, cm_image = self.dl_trainer.evaluate(X_img_test, y_img_test)
        print("Deep Learning Model Metrics:", metrics_image)
        print("Deep Learning Model Confusion Matrix:\n", cm_image)
        self.dl_trainer.save_model("image_diagnosis_model.h5")
        self.trained_image_model = True

    def make_prediction(self, new_structured_data, new_image_data):
        if not self.trained_structured_model or not self.trained_image_model:
            print("Models are not trained. Please run run_training_workflow first.")
            return None, None, None

        # Load preprocessors and models if not already loaded (e.g., in a deployed environment)
        if self.data_processor.preprocessor_structured is None:
            self.data_processor.preprocessor_structured = joblib.load("structured_preprocessor.joblib")
        self.classical_trainer.load_model("structured_diagnosis_model.joblib")
        self.dl_trainer.load_model("image_diagnosis_model.h5")

        # Process new structured data
        processed_structured_data = self.data_processor.preprocess_structured_data(new_structured_data, fit=False)
        classical_pred_class, classical_pred_proba = self.classical_trainer.predict(processed_structured_data)

        # Process new image data
        processed_image_data = self.data_processor.preprocess_image(new_image_data)
        dl_pred_class, dl_pred_proba = self.dl_trainer.predict(processed_image_data)

        # Combine predictions (simple average of probabilities as an example)
        combined_pred_proba = (classical_pred_proba + dl_pred_proba) / 2
        combined_pred_class = (combined_pred_proba > 0.5).astype(int)

        return combined_pred_class, combined_pred_proba, (
            {"classical_prediction": classical_pred_class[0], "classical_probability": classical_pred_proba[0]},
            {"dl_prediction": dl_pred_class[0], "dl_probability": dl_pred_proba[0]}
        )


if __name__ == "__main__":
    system = PredictionSystem()
    system.run_training_workflow(num_structured_samples=1000, num_image_samples=500)

    print("\n--- Making a Prediction with New Data ---")
    # Create dummy new data for prediction
    new_structured_patient_data = pd.DataFrame({
        "age": [65],
        "gender": ["Female"],
        "bmi": [28.5],
        "cholesterol": [210],
        "blood_pressure_systolic": [145],
        "blood_pressure_diastolic": [90],
        "smoking": [0],
        "diabetes": [1],
        "genetic_predisposition": [1]
    })
    new_medical_image = np.random.rand(1, 128, 128, 1) * 255
    new_medical_image = new_medical_image.astype(np.uint8)

    combined_class, combined_proba, individual_predictions = system.make_prediction(
        new_structured_patient_data, new_medical_image
    )

    if combined_class is not None:
        print("\nIndividual Model Predictions:", individual_predictions)
        print(f"Combined Prediction (Class): {combined_class[0]}")
        print(f"Combined Prediction (Probability of Disease): {combined_proba[0]:.4f}")