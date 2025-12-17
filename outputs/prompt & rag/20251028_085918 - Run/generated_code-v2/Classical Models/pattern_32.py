import pydicom
import numpy as np
import pandas as pd
import cv2
from skimage.transform import resize
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import joblib
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
import uvicorn
import io

# --- 1. Dummy Data Generation (for demonstration without actual files/DB) ---
# Dummy DICOM image (simple numpy array)
def generate_dummy_dicom_data():
    return np.random.rand(128, 128, 3).astype(np.float32) * 255

# Dummy EHR data
def generate_dummy_ehr_data():
    data = {
        'age': np.random.randint(20, 80, 100),
        'gender': np.random.choice(['Male', 'Female'], 100),
        'blood_pressure_systolic': np.random.randint(90, 180, 100),
        'cholesterol': np.random.randint(150, 250, 100),
        'smoking': np.random.choice([0, 1], 100),
        'disease_outcome': np.random.choice([0, 1], 100) # Target variable
    }
    return pd.DataFrame(data)

# --- 2. Data Preprocessing & Feature Engineering Layer ---

class ImagePreprocessor:
    def __init__(self, target_size=(64, 64)):
        self.target_size = target_size

    def preprocess(self, image_data):
        # Simulate DICOM loading (image_data is already a numpy array in this dummy setup)
        # In a real scenario, pydicom would load from file
        # ds = pydicom.dcmread(io.BytesIO(image_bytes))
        # image_array = ds.pixel_array

        processed_image = resize(image_data, self.target_size, anti_aliasing=True)
        processed_image = (processed_image - processed_image.min()) / (processed_image.max() - processed_image.min()) # Normalize to [0, 1]
        return np.expand_dims(processed_image, axis=0) # Add batch dimension for model input

class EHRPreprocessor:
    def __init__(self):
        self.preprocessor = None
        self.numerical_cols = ['age', 'blood_pressure_systolic', 'cholesterol']
        self.categorical_cols = ['gender', 'smoking']

    def fit(self, df):
        preprocessor_pipeline = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_cols)
            ])
        self.preprocessor = preprocessor_pipeline.fit(df)

    def transform(self, df):
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        return self.preprocessor.transform(df)

# --- 3. Dummy Model Training (Simplified) ---

def build_dummy_image_model(input_shape=(64, 64, 3)):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid') # Binary classification (e.g., disease presence)
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_dummy_models():
    # Image Model
    dummy_image_data = np.array([generate_dummy_dicom_data() for _ in range(100)])
    image_preprocessor = ImagePreprocessor()
    X_img_preprocessed = np.array([image_preprocessor.preprocess(img)[0] for img in dummy_image_data])
    y_img = np.random.randint(0, 2, 100) # Dummy labels
    
    image_model = build_dummy_image_model()
    image_model.fit(X_img_preprocessed, y_img, epochs=1, verbose=0)
    image_model.save('image_model.h5')
    joblib.dump(image_preprocessor, 'image_preprocessor.pkl')

    # EHR Model
    dummy_ehr_df = generate_dummy_ehr_data()
    ehr_preprocessor = EHRPreprocessor()
    ehr_preprocessor.fit(dummy_ehr_df.drop('disease_outcome', axis=1))
    X_ehr_preprocessed = ehr_preprocessor.transform(dummy_ehr_df.drop('disease_outcome', axis=1))
    y_ehr = dummy_ehr_df['disease_outcome']

    ehr_model = LogisticRegression()
    ehr_model.fit(X_ehr_preprocessed, y_ehr)
    joblib.dump(ehr_model, 'ehr_model.pkl')
    joblib.dump(ehr_preprocessor, 'ehr_preprocessor.pkl')


# --- 4. Prediction/Inference Service Layer (FastAPI) ---

app = FastAPI()

# Load pre-trained models and preprocessors
try:
    image_model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    image_model.load_weights('image_model.h5')
    image_preprocessor = joblib.load('image_preprocessor.pkl')
    ehr_model = joblib.load('ehr_model.pkl')
    ehr_preprocessor = joblib.load('ehr_preprocessor.pkl')
except FileNotFoundError:
    print("Models or preprocessors not found. Training dummy models...")
    train_dummy_models()
    image_model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    image_model.load_weights('image_model.h5')
    image_preprocessor = joblib.load('image_preprocessor.pkl')
    ehr_model = joblib.load('ehr_model.pkl')
    ehr_preprocessor = joblib.load('ehr_preprocessor.pkl')


class EHRData(BaseModel):
    age: int
    gender: str
    blood_pressure_systolic: int
    cholesterol: int
    smoking: int

class PredictionResponse(BaseModel):
    image_prediction: float
    ehr_prediction: float
    combined_risk_score: float

@app.post("/predict", response_model=PredictionResponse)
async def predict_diagnostics(
    ehr_data: EHRData = Form(...),
    medical_image: UploadFile = File(...)
):
    # Image Preprocessing and Inference
    image_bytes = await medical_image.read()
    # In a real scenario, use pydicom.dcmread to parse DICOM
    # For this dummy, we assume image_bytes can be converted to a dummy array.
    # A more robust solution would involve reading actual DICOM files.
    
    # Dummy conversion: create a numpy array from bytes (highly simplified)
    # This part would need actual DICOM parsing with pydicom in a real app
    try:
        # Attempt to interpret as PNG/JPEG for cv2 for a slightly more realistic dummy
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_decoded = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_decoded is None:
            # Fallback to pure dummy if decoding fails
            dummy_img_array = np.random.rand(200, 200, 3).astype(np.float32) * 255
        else:
            dummy_img_array = img_decoded
    except Exception:
        dummy_img_array = np.random.rand(200, 200, 3).astype(np.float32) * 255

    processed_image = image_preprocessor.preprocess(dummy_img_array)
    image_prediction = image_model.predict(processed_image)[0][0]

    # EHR Preprocessing and Inference
    ehr_df = pd.DataFrame([ehr_data.model_dump()])
    processed_ehr_data = ehr_preprocessor.transform(ehr_df)
    ehr_prediction = ehr_model.predict_proba(processed_ehr_data)[0][1]

    # Combined Risk Score (simple average for demonstration)
    combined_risk_score = (image_prediction + ehr_prediction) / 2

    return PredictionResponse(
        image_prediction=float(image_prediction),
        ehr_prediction=float(ehr_prediction),
        combined_risk_score=float(combined_risk_score)
    )

@app.get("/health")
async def health_check():
    return {"status": "ok"}


# To run the FastAPI app:
#   1. Make sure you have uvicorn installed: pip install uvicorn
#   2. Run from your terminal: uvicorn predictive_diagnostics_platform:app --reload
# Then access the API at http://127.0.0.1:8000/docs for Swagger UI.

# Example usage for POST /predict (using httpie or curl):
# http POST http://127.0.0.1:8000/predict \
#   ehr_data:='{"age": 55, "gender": "Male", "blood_pressure_systolic": 140, "cholesterol": 220, "smoking": 1}' \
#   medical_image@./path/to/your/dummy_image.png 
# (Replace dummy_image.png with any image file for testing. The content will be treated as dummy data.)
