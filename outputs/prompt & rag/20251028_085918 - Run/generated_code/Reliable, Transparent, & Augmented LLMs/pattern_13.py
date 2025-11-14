from pydantic import BaseModel, Field, conlist
from typing import List, Optional

class PatientData(BaseModel):
    patient_id: str
    symptoms: conlist(str, min_length=1) = Field(..., description="List of reported symptoms")
    age: int = Field(..., ge=0, description="Age of the patient in years")
    gender: str = Field(..., description="Gender of the patient")
    medical_history: List[str] = Field(default_factory=list, description="List of relevant past medical conditions")
    lab_results: List[str] = Field(default_factory=list, description="List of relevant lab test results")
    image_analysis_findings: List[str] = Field(default_factory=list, description="Findings from diagnostic image analysis")

class DiagnosisOutput(BaseModel):
    suggested_diagnoses: conlist(str, min_length=1) = Field(..., description="List of potential diagnoses suggested by the AI")
    reasoning: str = Field(..., description="Detailed explanation of the AI's reasoning for the suggested diagnoses")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="A score from 0.0 to 1.0 indicating the AI's confidence in the diagnoses")
    further_steps_recommended: Optional[List[str]] = Field(default_factory=list, description="Recommended further diagnostic steps or consultations")
    disclaimer: str = Field("This is an AI-generated suggestion and should not replace professional medical judgment.", description="Standard medical disclaimer")