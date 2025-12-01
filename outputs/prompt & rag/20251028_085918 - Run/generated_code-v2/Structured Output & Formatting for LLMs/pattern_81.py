from pydantic import BaseModel, Field
from typing import List, Optional

class Medication(BaseModel):
    drug_name: str = Field(..., description="Name of the medication")
    dosage: str = Field(..., description="Dosage of the medication (e.g., '10mg', '500mg')")
    frequency: str = Field(..., description="How often the medication should be taken (e.g., 'once daily', 'twice a day')")
    timing: str = Field(..., description="Specific timing for the medication (e.g., 'morning', 'before meals', 'at bedtime')")
    duration: str = Field(..., description="Duration for which the medication should be taken (e.g., '7 days', 'as needed', 'indefinitely')")

class MedicationRegimen(BaseModel):
    patient_id: Optional[str] = Field(None, description="Optional patient identifier")
    medications: List[Medication] = Field(..., description="List of medications in the regimen")
    notes: Optional[str] = Field(None, description="Any additional notes or instructions for the regimen")

class ValidationResult(BaseModel):
    is_valid: bool = Field(..., description="True if the regimen passes all validation checks, False otherwise")
    messages: List[str] = Field(..., description="List of validation messages or errors")
