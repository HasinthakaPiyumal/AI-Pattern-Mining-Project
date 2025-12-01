import json
import requests
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# 4. Structured Output Schema Definition using Pydantic
class Medication(BaseModel):
    name: str
    dosage: str
    frequency: str
    notes: Optional[str] = None

class LifestyleRecommendation(BaseModel):
    type: str  # e.g., "Diet", "Exercise", "Stress Management"
    details: str

class Appointment(BaseModel):
    date: str
    time: str
    reason: str
    provider: Optional[str] = None

class TreatmentPlan(BaseModel):
    patient_id: str
    disease: str
    plan_id: str = Field(..., description="Unique identifier for the treatment plan")
    generated_date: str
    medications: List[Medication]
    lifestyle_recommendations: List[LifestyleRecommendation]
    follow_up_appointments: List[Appointment]
    additional_notes: Optional[str] = None


# Simulate LLM interaction
def _mock_llm_response(patient_data: dict, medical_guidelines: str) -> str:
    patient_id = patient_data.get("patient_id", "unknown_patient")
    disease = patient_data.get("diagnosis", "unknown_disease")

    # This is a mock LLM response that tries to adhere to the schema
    # In a real application, this would be an actual API call to an LLM
    mock_json_output = {
        "patient_id": patient_id,
        "disease": disease,
        "plan_id": f"TP-{patient_id}-{disease}-123",
        "generated_date": "2023-10-27",
        "medications": [
            {"name": "Metformin", "dosage": "500mg", "frequency": "Twice daily", "notes": "Take with food"},
            {"name": "Lisinopril", "dosage": "10mg", "frequency": "Once daily"}
        ],
        "lifestyle_recommendations": [
            {"type": "Diet", "details": "Reduce sugar intake, increase fiber"},
            {"type": "Exercise", "details": "30 minutes moderate exercise, 5 times a week"},
            {"type": "Stress Management", "details": "Practice mindfulness daily"}
        ],
        "follow_up_appointments": [
            {"date": "2023-11-15", "time": "10:00 AM", "reason": "Follow-up with endocrinologist"}
        ],
        "additional_notes": f"Plan generated based on {disease} guidelines and patient {patient_id}'s history."
    }
    return json.dumps(mock_json_output)


# 3. Prompt Engineering Module (simplified as a function to prepare context)
def prepare_llm_prompt_context(patient_data: dict, medical_guidelines: str) -> str:
    # In a real scenario, this would craft a sophisticated prompt for the LLM.
    # For this simulation, we just combine the data.
    context = f"Patient Data: {json.dumps(patient_data)}\nMedical Guidelines: {medical_guidelines}\n\nGenerate a personalized treatment plan in JSON format adhering to the following Pydantic schema:\n{json.dumps(TreatmentPlan.schema())}"
    return context


# 5. Post-processing and Validation Module
def validate_and_parse_treatment_plan(llm_output: str) -> TreatmentPlan:
    try:
        parsed_data = json.loads(llm_output)
        validated_plan = TreatmentPlan(**parsed_data)
        return validated_plan
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM output is not valid JSON: {e}")
    except Exception as e:
        raise ValueError(f"LLM output does not match schema: {e}")


# 6. Integration Layer (Mock functions for EHR and Monitoring Apps)
def _mock_ehr_update(treatment_plan: TreatmentPlan):
    print(f"[EHR Integration] Successfully updated EHR for patient {treatment_plan.patient_id} with plan {treatment_plan.plan_id}")
    # In a real system, this would involve calling an external EHR API.

def _mock_patient_monitoring_app_update(treatment_plan: TreatmentPlan):
    print(f"[Monitoring App Integration] Pushed plan details to monitoring app for patient {treatment_plan.patient_id}")
    # In a real system, this would involve calling an external patient monitoring app API.


# FastAPI Application
app = FastAPI("Personalized Treatment Plan Generator")

class PatientInput(BaseModel):
    patient_id: str
    medical_history: str
    current_medications: List[str]
    lab_results: dict
    lifestyle_factors: dict
    diagnosis: str

class GuidelinesInput(BaseModel):
    disease: str
    guidelines_text: str

@app.post("/generate_treatment_plan", response_model=TreatmentPlan)
async def generate_treatment_plan_api(
    patient_input: PatientInput,
    guidelines_input: GuidelinesInput
):
    # 1. Data Ingestion and Preprocessing (handled by Pydantic validation of PatientInput)
    patient_data = patient_input.dict()
    medical_guidelines = guidelines_input.guidelines_text

    # 3. Prompt Engineering Module
    llm_context = prepare_llm_prompt_context(patient_data, medical_guidelines)
    
    # 2. LLM Integration (simulated)
    llm_raw_output = _mock_llm_response(patient_data, medical_guidelines)
    
    # 5. Post-processing and Validation Module
    try:
        treatment_plan = validate_and_parse_treatment_plan(llm_raw_output)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 6. Integration Layer
    _mock_ehr_update(treatment_plan)
    _mock_patient_monitoring_app_update(treatment_plan)

    return treatment_plan
