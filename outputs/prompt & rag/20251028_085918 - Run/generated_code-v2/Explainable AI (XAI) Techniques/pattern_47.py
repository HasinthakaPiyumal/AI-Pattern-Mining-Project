import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# --- 1. Database Setup (SQLite for simplicity in a single file) ---
DATABASE_URL = "sqlite:///./medixplain.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class PatientDB(Base):
    __tablename__ = "patients"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    age = Column(Integer)
    gender = Column(String)
    medical_history = Column(Text)
    symptoms = Column(Text)
    lab_results = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    diagnoses = relationship("DiagnosisDB", back_populates="patient")

class DiagnosisDB(Base):
    __tablename__ = "diagnoses"
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"))
    model_id = Column(String)
    predicted_class = Column(String)
    prediction_probability = Column(Float)
    actual_class = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    patient = relationship("PatientDB", back_populates="diagnoses")
    explanations = relationship("ExplanationDB", back_populates="diagnosis")

class ExplanationDB(Base):
    __tablename__ = "explanations"
    id = Column(Integer, primary_key=True, index=True)
    diagnosis_id = Column(Integer, ForeignKey("diagnoses.id"))
    explanation_type = Column(String)
    important_features = Column(Text) # JSON string of feature importance
    counterfactuals = Column(Text)    # JSON string of counterfactual examples
    local_rules = Column(Text)       # JSON string of local rules
    timestamp = Column(DateTime, default=datetime.utcnow)

    diagnosis = relationship("DiagnosisDB", back_populates="explanations")

class ModelDB(Base):
    __tablename__ = "models"
    id = Column(String, primary_key=True, index=True)
    name = Column(String)
    description = Column(Text)
    path = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class UserRuleDB(Base):
    __tablename__ = "user_rules"
    id = Column(Integer, primary_key=True, index=True)
    rule_text = Column(Text)
    is_hypothesis = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class ExplanationMetadataDB(Base):
    __tablename__ = "explanation_metadata"
    id = Column(Integer, primary_key=True, index=True)
    attribute = Column(String)
    item = Column(String) # e.g., specific symptom value
    impact_score = Column(Float)
    category = Column(String) # e.g., 'frequent_positive_contributor', 'frequent_misclassifier'
    timestamp = Column(DateTime, default=datetime.utcnow)

# Create database tables
Base.metadata.create_all(bind=engine)

# --- Pydantic Models for Data Validation (simulating FastAPI request/response bodies) ---
class PatientInput(BaseModel):
    name: str
    age: int
    gender: str
    medical_history: str
    symptoms: str
    lab_results: str

class PredictionResult(BaseModel):
    predicted_class: str
    prediction_probability: float
    diagnosis_id: int

class ExplanationResult(BaseModel):
    explanation_type: str
    important_features: Dict[str, float]
    counterfactuals: List[Dict[str, Any]]
    local_rules: List[str]

class WhatIfRequest(BaseModel):
    patient_id: int
    modified_attributes: Dict[str, Any]

class UserRuleInput(BaseModel):
    rule_text: str
    is_hypothesis: bool = True

class ExplanationMetadataOutput(BaseModel):
    attribute: str
    item: str
    impact_score: float
    category: str

# --- 2. LACE Explanation Placeholder Function ---
def generate_lace_explanation(model, instance_df: pd.DataFrame, feature_names: List[str], target_class: str = "diagnosis_A") -> Dict[str, Any]:
    # This is a simplified placeholder for LACE. A real LACE implementation is complex.
    # It would typically involve perturbation, local model fitting, and rule extraction.

    # For demonstration, we'll simulate some feature importance and a simple rule.
    feature_importance = {}
    for col in feature_names:
        # Simulate importance based on some heuristic or random values
        feature_importance[col] = np.random.uniform(0.1, 0.9)
    
    # Sort features by importance (descending)
    important_features_sorted = dict(sorted(feature_importance.items(), key=lambda item: item[1], reverse=True))

    counterfactuals = [
        {"attribute": "fever", "original_value": "high", "counterfactual_value": "low", "change_in_prediction": "diagnosis_B"},
        {"attribute": "cough", "original_value": "severe", "counterfactual_value": "mild", "change_in_prediction": "no_diagnosis"}
    ]

    local_rules = [
        f"IF {list(important_features_sorted.keys())[0]} is high AND {list(important_features_sorted.keys())[1]} is present THEN likely {target_class}",
        "IF no fever AND good appetite THEN unlikely pneumonia"
    ]

    return {
        "explanation_type": "LACE-like explanation",
        "important_features": important_features_sorted,
        "counterfactuals": counterfactuals,
        "local_rules": local_rules
    }

# --- 3. Dummy Model Training/Loading ---
def load_or_train_dummy_model():
    db = SessionLocal()
    model_id = "dummy_rf_v1"
    model_path = "dummy_rf_model.joblib"

    existing_model_meta = db.query(ModelDB).filter(ModelDB.id == model_id).first()

    if existing_model_meta and existing_model_meta.path == model_path:
        try:
            model = joblib.load(model_path)
            st.sidebar.success("Dummy model loaded from disk.")
            return model, model_id
        except FileNotFoundError:
            st.sidebar.warning("Model file not found, re-training.")
    
    # Generate some dummy data for a binary classification task
    np.random.seed(42)
    num_samples = 100
    data = {
        'age': np.random.randint(20, 80, num_samples),
        'gender_M': np.random.randint(0, 2, num_samples),
        'fever': np.random.randint(0, 2, num_samples), # 0: no, 1: yes
        'cough': np.random.randint(0, 2, num_samples),
        'fatigue': np.random.randint(0, 2, num_samples),
        'lab_A': np.random.rand(num_samples) * 100,
        'lab_B': np.random.rand(num_samples) * 50,
        'diagnosis_A': np.random.randint(0, 2, num_samples) # Target variable
    }
    df = pd.DataFrame(data)

    X = df.drop('diagnosis_A', axis=1)
    y = df['diagnosis_A']

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    joblib.dump(model, model_path)
    if not existing_model_meta:
        new_model_meta = ModelDB(id=model_id, name="Dummy Random Forest", description="A simple RF model for demonstration.", path=model_path)
        db.add(new_model_meta)
        db.commit()
    db.close()
    st.sidebar.success("Dummy model trained and saved.")
    return model, model_id, X.columns.tolist()

dummy_model, dummy_model_id, feature_names = load_or_train_dummy_model()

# --- 4. Backend Logic Functions (called directly, simulating FastAPI) ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def _patient_data_to_df_row(patient_data: PatientInput, feature_names: List[str]) -> pd.DataFrame:
    processed_data = {
        'age': patient_data.age,
        'gender_M': 1 if patient_data.gender.lower() == 'male' else 0,
        'fever': 1 if 'fever' in patient_data.symptoms.lower() else 0,
        'cough': 1 if 'cough' in patient_data.symptoms.lower() else 0,
        'fatigue': 1 if 'fatigue' in patient_data.symptoms.lower() else 0,
    }
    # Dummy lab results for demonstration, assumes simple parsing or fixed values
    if 'lab_A:' in patient_data.lab_results:
        try: processed_data['lab_A'] = float(patient_data.lab_results.split('lab_A:')[1].split(',')[0].strip()) 
        except: processed_data['lab_A'] = 50.0 # Default
    else: processed_data['lab_A'] = 50.0
    if 'lab_B:' in patient_data.lab_results:
        try: processed_data['lab_B'] = float(patient_data.lab_results.split('lab_B:')[1].split(',')[0].strip()) 
        except: processed_data['lab_B'] = 25.0 # Default
    else: processed_data['lab_B'] = 25.0
    
    # Ensure all model features are present, fill missing with defaults/0
    for feature in feature_names:
        if feature not in processed_data:
            processed_data[feature] = 0 # Or a suitable default for the feature

    return pd.DataFrame([processed_data], columns=feature_names)

def create_and_predict_patient(db, patient_data: PatientInput, model, model_id: str, feature_names: List[str]) -> PredictionResult:
    db_patient = PatientDB(
        name=patient_data.name,
        age=patient_data.age,
        gender=patient_data.gender,
        medical_history=patient_data.medical_history,
        symptoms=patient_data.symptoms,
        lab_results=patient_data.lab_results
    )
    db.add(db_patient)
    db.commit()
    db.refresh(db_patient)

    instance_df = _patient_data_to_df_row(patient_data, feature_names)
    prediction_proba = model.predict_proba(instance_df)[0]
    predicted_class_idx = np.argmax(prediction_proba)
    predicted_class = "diagnosis_A" if predicted_class_idx == 1 else "no_diagnosis"

    db_diagnosis = DiagnosisDB(
        patient_id=db_patient.id,
        model_id=model_id,
        predicted_class=predicted_class,
        prediction_probability=float(prediction_proba[predicted_class_idx])
    )
    db.add(db_diagnosis)
    db.commit()
    db.refresh(db_diagnosis)

    return PredictionResult(predicted_class=predicted_class, prediction_probability=float(prediction_proba[predicted_class_idx]), diagnosis_id=db_diagnosis.id)

def get_explanation_for_diagnosis(db, diagnosis_id: int, model, feature_names: List[str]) -> ExplanationResult:
    diagnosis = db.query(DiagnosisDB).filter(DiagnosisDB.id == diagnosis_id).first()
    if not diagnosis: return None

    patient = db.query(PatientDB).filter(PatientDB.id == diagnosis.patient_id).first()
    if not patient: return None
    
    # Reconstruct patient_input for explanation generation
    patient_input_for_explanation = PatientInput(
        name=patient.name, age=patient.age, gender=patient.gender,
        medical_history=patient.medical_history, symptoms=patient.symptoms, lab_results=patient.lab_results
    )
    instance_df = _patient_data_to_df_row(patient_input_for_explanation, feature_names)

    explanation = generate_lace_explanation(model, instance_df, feature_names, diagnosis.predicted_class)

    db_explanation = ExplanationDB(
        diagnosis_id=diagnosis.id,
        explanation_type=explanation["explanation_type"],
        important_features=str(explanation["important_features"]),
        counterfactuals=str(explanation["counterfactuals"]),
        local_rules=str(explanation["local_rules"])
    )
    db.add(db_explanation)
    db.commit()
    db.refresh(db_explanation)

    return ExplanationResult(**explanation)

def perform_what_if(db, request: WhatIfRequest, model, feature_names: List[str], model_id: str) -> PredictionResult:
    original_patient = db.query(PatientDB).filter(PatientDB.id == request.patient_id).first()
    if not original_patient: return None

    # Create a modified patient_input object
    modified_patient_data = PatientInput(
        name=original_patient.name, age=original_patient.age, gender=original_patient.gender,
        medical_history=original_patient.medical_history, symptoms=original_patient.symptoms, lab_results=original_patient.lab_results
    )
    for attr, value in request.modified_attributes.items():
        if hasattr(modified_patient_data, attr):
            setattr(modified_patient_data, attr, value)
        # Special handling for symptoms/lab_results if needed to merge changes
        if attr == "symptoms":
            modified_patient_data.symptoms = f"{original_patient.symptoms}, {value}" if original_patient.symptoms else value
        if attr == "lab_results":
            modified_patient_data.lab_results = f"{original_patient.lab_results}, {value}" if original_patient.lab_results else value


    instance_df = _patient_data_to_df_row(modified_patient_data, feature_names)
    prediction_proba = model.predict_proba(instance_df)[0]
    predicted_class_idx = np.argmax(prediction_proba)
    predicted_class = "diagnosis_A" if predicted_class_idx == 1 else "no_diagnosis"

    db_diagnosis = DiagnosisDB(
        patient_id=original_patient.id,
        model_id=model_id,
        predicted_class=predicted_class,
        prediction_probability=float(prediction_proba[predicted_class_idx]),
        timestamp=datetime.utcnow() # New timestamp for what-if scenario
    )
    db.add(db_diagnosis)
    db.commit()
    db.refresh(db_diagnosis)

    return PredictionResult(predicted_class=predicted_class, prediction_probability=float(prediction_proba[predicted_class_idx]), diagnosis_id=db_diagnosis.id)

def add_user_defined_rule(db, rule_data: UserRuleInput) -> UserRuleDB:
    db_rule = UserRuleDB(rule_text=rule_data.rule_text, is_hypothesis=rule_data.is_hypothesis)
    db.add(db_rule)
    db.commit()
    db.refresh(db_rule)
    return db_rule

def get_aggregated_explanation_metadata(db) -> List[ExplanationMetadataOutput]:
    # This is a highly simplified aggregation. A real system would analyze many explanations.
    all_explanations = db.query(ExplanationDB).all()
    metadata_list = []

    feature_impacts = {}
    for exp in all_explanations:
        try:
            features = eval(exp.important_features) # Use eval carefully, assuming trusted source
            for feature, importance in features.items():
                feature_impacts[feature] = feature_impacts.get(feature, 0) + importance
        except Exception:
            pass # Handle malformed JSON/text
    
    for feature, impact in feature_impacts.items():
        metadata_list.append(ExplanationMetadataOutput(
            attribute=feature,
            item="N/A", # More sophisticated parsing needed for specific item values
            impact_score=impact,
            category="frequent_contributor"
        ))
    
    # Add some dummy rules metadata
    metadata_list.append(ExplanationMetadataOutput(
        attribute="rule_adherence",
        item="user_rule_1_match_rate",
        impact_score=np.random.uniform(0.5, 0.9),
        category="rule_alignment"
    ))
    metadata_list.append(ExplanationMetadataOutput(
        attribute="model_consistency",
        item="model_A_vs_model_B",
        impact_score=np.random.uniform(0.3, 0.7),
        category="model_comparison"
    ))

    return metadata_list

# --- 5. Streamlit Frontend ---
st.set_page_config(layout="wide", page_title="MediXplain")
st.title("🩺 MediXplain: Interactive Diagnostic Assistant")

# --- Sidebar for Patient Input/Selection ---
st.sidebar.header("Patient Management")
with st.sidebar.form("new_patient_form"):
    st.subheader("Add New Patient")
    patient_name = st.text_input("Name")
    patient_age = st.number_input("Age", min_value=0, max_value=120, value=30)
    patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    patient_medical_history = st.text_area("Medical History (e.g., hypertension, diabetes)")
    patient_symptoms = st.text_area("Current Symptoms (comma-separated, e.g., fever, cough, fatigue)")
    patient_lab_results = st.text_area("Lab Results (e.g., lab_A: 75.2, lab_B: 30.1)")
    add_patient_button = st.form_submit_button("Add Patient & Get Prediction")

    current_db = next(get_db())
    if add_patient_button:
        if patient_name and patient_symptoms:
            patient_input = PatientInput(
                name=patient_name, age=patient_age, gender=patient_gender,
                medical_history=patient_medical_history, symptoms=patient_symptoms, lab_results=patient_lab_results
            )
            prediction = create_and_predict_patient(current_db, patient_input, dummy_model, dummy_model_id, feature_names)
            st.session_state['selected_patient_id'] = current_db.query(PatientDB).filter_by(name=patient_name, age=patient_age).order_by(PatientDB.created_at.desc()).first().id
            st.session_state['current_diagnosis_id'] = prediction.diagnosis_id
            st.sidebar.success(f"Patient '{patient_name}' added. Predicted: {prediction.predicted_class} with prob {prediction.prediction_probability:.2f}")
        else:
            st.sidebar.error("Please fill in patient name and symptoms.")
    current_db.close()

current_db = next(get_db())
all_patients = current_db.query(PatientDB).all()
patient_options = {f"{p.name} (ID: {p.id})": p.id for p in all_patients}
selected_patient_display = st.sidebar.selectbox(
    "Select Existing Patient",
    options=list(patient_options.keys()),
    index=0 if not 'selected_patient_id' in st.session_state or st.session_state['selected_patient_id'] not in patient_options.values() else list(patient_options.values()).index(st.session_state['selected_patient_id'])
) if all_patients else None

if selected_patient_display:
    st.session_state['selected_patient_id'] = patient_options[selected_patient_display]
    # Automatically fetch latest diagnosis for selected patient
    latest_diagnosis = current_db.query(DiagnosisDB).filter_by(patient_id=st.session_state['selected_patient_id']).order_by(DiagnosisDB.timestamp.desc()).first()
    if latest_diagnosis:
        st.session_state['current_diagnosis_id'] = latest_diagnosis.id
    else:
        st.session_state['current_diagnosis_id'] = None
else:
    st.session_state['selected_patient_id'] = None
    st.session_state['current_diagnosis_id'] = None
current_db.close()

# --- Main Content Area ---
if st.session_state['selected_patient_id'] is None:
    st.info("Please add a new patient or select an existing one from the sidebar.")
else:
    current_db = next(get_db())
    selected_patient = current_db.query(PatientDB).filter_by(id=st.session_state['selected_patient_id']).first()
    st.subheader(f"Patient: {selected_patient.name} (ID: {selected_patient.id})")
    current_db.close()

    tab1, tab2, tab3, tab4 = st.tabs(["Prediction & Explanation", "What-If Analysis", "User-Defined Rules", "Global Insights"])

    with tab1:
        st.header("Model Prediction and Explanation")
        if st.session_state['current_diagnosis_id']:
            current_db = next(get_db())
            diagnosis = current_db.query(DiagnosisDB).filter_by(id=st.session_state['current_diagnosis_id']).first()
            st.write(f"**Predicted Class:** {diagnosis.predicted_class}")
            st.write(f"**Prediction Probability:** {diagnosis.prediction_probability:.2f}")
            
            st.subheader("LACE-based Explanation")
            explanation_result = get_explanation_for_diagnosis(current_db, diagnosis.id, dummy_model, feature_names)
            if explanation_result:
                st.write(f"**Explanation Type:** {explanation_result.explanation_type}")
                st.write("**Important Features:**")
                for feature, importance in explanation_result.important_features.items():
                    st.write(f"- {feature}: {importance:.3f}")
                
                st.write("**Counterfactual Examples:**")
                for cf in explanation_result.counterfactuals:
                    st.write(f"- If {cf['attribute']} was '{cf['counterfactual_value']}' instead of '{cf['original_value']}', prediction might change to '{cf['change_in_prediction']}'.")

                st.write("**Local Rules:**")
                for rule in explanation_result.local_rules:
                    st.write(f"- {rule}")
            else:
                st.warning("Could not generate explanation.")
            current_db.close()
        else:
            st.info("No diagnosis available for this patient. Add a new patient or ensure prediction was made.")

    with tab2:
        st.header("What-If Analysis")
        if st.session_state['selected_patient_id']:
            st.write("Modify patient attributes below to see how the prediction changes.")
            current_db = next(get_db())
            patient_to_modify = current_db.query(PatientDB).filter_by(id=st.session_state['selected_patient_id']).first()
            current_db.close()
            
            with st.form("what_if_form"):
                modified_age = st.number_input("Age", value=patient_to_modify.age, key="what_if_age")
                modified_gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=(["Male", "Female", "Other"].index(patient_to_modify.gender) if patient_to_modify.gender in ["Male", "Female", "Other"] else 0), key="what_if_gender")
                modified_symptoms = st.text_area("Symptoms (e.g., 'fever: high', 'cough: mild')", value=patient_to_modify.symptoms, key="what_if_symptoms")
                modified_lab_results = st.text_area("Lab Results (e.g., 'lab_A: 60.0')", value=patient_to_modify.lab_results, key="what_if_lab_results")
                
                what_if_submit = st.form_submit_button("Run What-If Analysis")
            
            if what_if_submit:
                modified_attributes = {
                    "age": modified_age,
                    "gender": modified_gender,
                    "symptoms": modified_symptoms,
                    "lab_results": modified_lab_results
                }
                what_if_request = WhatIfRequest(
                    patient_id=st.session_state['selected_patient_id'],
                    modified_attributes=modified_attributes
                )
                current_db = next(get_db())
                what_if_prediction = perform_what_if(current_db, what_if_request, dummy_model, feature_names, dummy_model_id)
                current_db.close()
                if what_if_prediction:
                    st.success("What-If Analysis Result:")
                    st.write(f"**New Predicted Class:** {what_if_prediction.predicted_class}")
                    st.write(f"**New Prediction Probability:** {what_if_prediction.prediction_probability:.2f}")
                    
                    # Get explanation for the what-if prediction
                    st.subheader("Explanation for What-If Prediction")
                    current_db = next(get_db())
                    what_if_explanation = get_explanation_for_diagnosis(current_db, what_if_prediction.diagnosis_id, dummy_model, feature_names)
                    current_db.close()

                    if what_if_explanation:
                        st.write("**Important Features:**")
                        for feature, importance in what_if_explanation.important_features.items():
                            st.write(f"- {feature}: {importance:.3f}")
                    else:
                        st.warning("Could not generate explanation for what-if scenario.")
                else:
                    st.error("Failed to perform what-if analysis.")
        else:
            st.info("Select a patient to perform what-if analysis.")

    with tab3:
        st.header("User-Defined Rules & Hypotheses")
        with st.form("user_rule_form"):
            rule_text = st.text_area("Enter a medical rule or hypothesis (e.g., 'IF patient has high fever AND cough THEN likely pneumonia')")
            is_hypothesis = st.checkbox("Is this a hypothesis (vs. a confirmed rule)?", value=True)
            add_rule_button = st.form_submit_button("Add Rule")

            if add_rule_button:
                if rule_text:
                    rule_input = UserRuleInput(rule_text=rule_text, is_hypothesis=is_hypothesis)
                    current_db = next(get_db())
                    add_user_defined_rule(current_db, rule_input)
                    current_db.close()
                    st.success("Rule added successfully!")
                else:
                    st.error("Rule text cannot be empty.")
        
        st.subheader("Existing User Rules")
        current_db = next(get_db())
        all_rules = current_db.query(UserRuleDB).all()
        current_db.close()
        if all_rules:
            for rule in all_rules:
                st.markdown(f"- **{'Hypothesis' if rule.is_hypothesis else 'Rule'}:** {rule.rule_text}")
        else:
            st.info("No user-defined rules yet.")

    with tab4:
        st.header("Global Explanation Insights (Metadata)")
        st.write("Aggregated insights from multiple patient explanations.")
        current_db = next(get_db())
        metadata_results = get_aggregated_explanation_metadata(current_db)
        current_db.close()

        if metadata_results:
            for meta in metadata_results:
                st.markdown(f"- **Attribute:** {meta.attribute}, **Item:** {meta.item}, **Impact:** {meta.impact_score:.2f}, **Category:** {meta.category}")
        else:
            st.info("No explanation metadata available yet. Process more patient explanations.")

# --- How to Run ---
st.sidebar.markdown("""
---
**How to Run This App:**
1. Save this code as `medixplain_app.py`.
2. Open your terminal in the same directory.
3. Run `streamlit run medixplain_app.py`
""")
