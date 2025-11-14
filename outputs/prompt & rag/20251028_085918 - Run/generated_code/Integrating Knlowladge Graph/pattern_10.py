from pydantic import BaseModel
from typing import List, Optional

class PatientSymptoms(BaseModel):
    symptoms: str
    medical_history: Optional[str] = None
    lab_results: Optional[str] = None

class DiagnosticResult(BaseModel):
    diagnosis: str
    explanation: str
    kg_evidence: List[str]

class KGQuery(BaseModel):
    entity: Optional[str] = None
    relation: Optional[str] = None
    target_entity: Optional[str] = None
    natural_language_query: str
