"""
Data models for the Agentic and Trustworthy AI Diagnostic Assistant.
"""

from pydantic import BaseModel, Field
from typing import List, Optional

class PatientInfo(BaseModel):
    patient_id: str = Field(..., description="Unique identifier for the patient.")
    age: int = Field(..., gt=0, description="Age of the patient in years.")
    gender: str = Field(..., description="Gender of the patient.")
    symptoms: List[str] = Field(..., description="List of reported symptoms.")
    medical_history: List[str] = Field(default_factory=list, description="Relevant past medical history.")
    lab_results: Optional[dict] = Field(None, description="Dictionary of recent lab results.")

class DiagnosticSuggestion(BaseModel):
    diagnosis: str = Field(..., description="Proposed diagnosis.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for the diagnosis (0.0 to 1.0).")
    reasoning_path: List[str] = Field(..., description="Step-by-step reasoning leading to the diagnosis.")
    recommended_actions: List[str] = Field(default_factory=list, description="Recommended next steps or tests.")
    disclaimer: str = Field("This is an AI-generated suggestion and should not replace professional medical advice.", description="Standard medical disclaimer.")

class Feedback(BaseModel):
    patient_id: str
    suggested_diagnosis: str
    expert_diagnosis: str
    was_accurate: bool
    corrections: Optional[str] = None
    additional_notes: Optional[str] = None
