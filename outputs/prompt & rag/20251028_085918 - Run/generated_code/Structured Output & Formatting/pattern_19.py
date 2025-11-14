from pydantic import BaseModel, Field
from typing import List, Optional

class MedicalSummary(BaseModel):
    patient_name: str = Field(..., description="The full name of the patient.")
    diagnosis: List[str] = Field(..., description="A list of medical diagnoses for the patient.")
    medications: List[str] = Field(..., description="A list of medications prescribed to the patient.")
    treatment_plan: str = Field(..., description="A summary of the treatment plan.")
    allergies: List[str] = Field(default_factory=list, description="A list of known allergies for the patient. Can be empty if none.")
    next_appointment: Optional[str] = Field(None, description="The date or description of the next scheduled appointment, if any.")