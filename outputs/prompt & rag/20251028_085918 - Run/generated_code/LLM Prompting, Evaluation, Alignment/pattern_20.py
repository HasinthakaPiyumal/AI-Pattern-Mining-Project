from pydantic import BaseModel
from typing import Dict, List, Optional

class PatientInfo(BaseModel):
    patient_id: str
    age: int
    gender: str  # e.g., "Male", "Female", "Other"
    medical_history: Optional[str] = None

class ImageFinding(BaseModel):
    area: str  # e.g., "Lungs", "Heart", "Abdomen"
    finding: str  # e.g., "Clear", "Cardiomegaly", "No acute abnormalities"

class ImageAnalysisResults(BaseModel):
    image_type: str  # e.g., "X-ray", "MRI", "CT Scan"
    findings: List[ImageFinding]
    impression: Optional[str] = None

class ClinicalReport(BaseModel):
    report_id: str
    patient_info: PatientInfo
    image_analysis_results: ImageAnalysisResults
    generated_text: str
    evaluation_score: Optional[float] = None
    qa_feedback: Optional[Dict[str, str]] = None
