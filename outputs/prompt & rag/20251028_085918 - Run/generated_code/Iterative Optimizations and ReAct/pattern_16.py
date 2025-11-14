from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class PatientData(BaseModel):
    patient_id: str = Field(..., description="Unique identifier for the patient.")
    symptoms: List[str] = Field(..., description="List of reported symptoms.")
    medical_history: List[str] = Field(default_factory=list, description="List of relevant medical history.")
    lab_results: Dict[str, str] = Field(default_factory=dict, description="Dictionary of lab test names and their results.")
    image_data_description: Optional[str] = Field(None, description="Description or reference to available image data (e.g., 'X-ray of chest', 'MRI brain scan').")
    current_medications: List[str] = Field(default_factory=list, description="List of medications the patient is currently taking.")

class Diagnosis(BaseModel):
    disease: str = Field(..., description="Proposed disease or condition.")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence level of the diagnosis (0.0 to 1.0).")
    justification: str = Field(..., description="Reasoning behind the diagnosis.")
    differential_diagnoses: List[str] = Field(default_factory=list, description="Other possible diagnoses considered.")

class TreatmentPlan(BaseModel):
    medication: Optional[str] = Field(None, description="Recommended medication.")
    dosage: Optional[str] = Field(None, description="Prescribed dosage.")
    duration: Optional[str] = Field(None, description="Duration of treatment.")
    instructions: str = Field(..., description="Detailed instructions for the patient.")
    follow_up_required: bool = Field(False, description="Indicates if a follow-up appointment is necessary.")
    expected_outcome: str = Field(..., description="Anticipated outcome of the treatment.")
    recommended_tests: List[str] = Field(default_factory=list, description="Additional tests recommended as part of the plan.")

class Feedback(BaseModel):
    patient_id: str = Field(..., description="Unique identifier for the patient.")
    initial_diagnosis: str = Field(..., description="The diagnosis that was initially made.")
    initial_treatment: str = Field(..., description="The treatment that was initially prescribed.")
    patient_response: str = Field(..., description="Patient's subjective feedback on treatment efficacy and symptoms.")
    outcome_observed: str = Field(..., description="Objective outcome observed (e.g., 'symptoms improved', 'no change', 'worsened').")
    efficacy_rating: float = Field(..., ge=0.0, le=1.0, description="Numeric rating of treatment efficacy (0.0 to 1.0).")
    new_lab_results: Dict[str, str] = Field(default_factory=dict, description="Any new lab results obtained after treatment.")
    new_image_data_description: Optional[str] = Field(None, description="Description of new image data after treatment.")
