import json
from pydantic import BaseModel, ValidationError, Field
from typing import List, Optional

class Medication(BaseModel):
    name: str
    dosage: str
    frequency: str
    notes: Optional[str] = None

class Appointment(BaseModel):
    date: str = Field(..., description="Format: YYYY-MM-DD")
    time: str = Field(..., description="Format: HH:MM")
    specialty: str
    reason: str
    location: Optional[str] = None

class DietRecommendation(BaseModel):
    type: str
    details: str

class LifestyleChange(BaseModel):
    type: str
    details: str

class FollowUp(BaseModel):
    date: str = Field(..., description="Format: YYYY-MM-DD")
    action: str

class TreatmentPlan(BaseModel):
    patient_id: str
    diagnosis: str
    medications: List[Medication]
    appointments: List[Appointment]
    diet_recommendations: List[DietRecommendation]
    lifestyle_changes: List[LifestyleChange]
    follow_ups: List[FollowUp]
    generated_by_llm: bool = True

def simulate_llm_response(patient_input: dict) -> str:
    if "invalid_case" in patient_input and patient_input["invalid_case"]:
        return "This is not a valid JSON response from the LLM."
    
    return json.dumps({
        "patient_id": patient_input.get("patient_id", "PAT001"),
        "diagnosis": patient_input.get("diagnosis", "Type 2 Diabetes"),
        "medications": [
            {
                "name": "Metformin",
                "dosage": "500mg",
                "frequency": "Twice daily",
                "notes": "Take with food"
            },
            {
                "name": "Lisinopril",
                "dosage": "10mg",
                "frequency": "Once daily"
            }
        ],
        "appointments": [
            {
                "date": "2024-03-15",
                "time": "10:00",
                "specialty": "Endocrinology",
                "reason": "Follow-up on blood sugar levels"
            },
            {
                "date": "2024-04-01",
                "time": "14:30",
                "specialty": "Nutritionist",
                "reason": "Dietary plan review"
            }
        ],
        "diet_recommendations": [
            {
                "type": "Low Carb",
                "details": "Focus on non-starchy vegetables, lean proteins, and healthy fats."
            }
        ],
        "lifestyle_changes": [
            {
                "type": "Exercise",
                "details": "Aim for 30 minutes of moderate-intensity exercise most days of the week."
            },
            {
                "type": "Stress Management",
                "details": "Practice mindfulness or meditation daily."
            }
        ],
        "follow_ups": [
            {
                "date": "2024-06-01",
                "action": "Review blood test results and medication adjustments."
            }
        ]
    })

def generate_and_validate_treatment_plan(patient_data: dict) -> Optional[TreatmentPlan]:
    llm_raw_output = simulate_llm_response(patient_data)
    
    try:
        parsed_json = json.loads(llm_raw_output)
        treatment_plan = TreatmentPlan(**parsed_json)
        return treatment_plan
    except json.JSONDecodeError as e:
        print(f"Error: LLM output is not valid JSON: {e}")
        return None
    except ValidationError as e:
        print(f"Error: LLM output does not conform to the TreatmentPlan schema: {e}")
        return None

if __name__ == "__main__":
    # Example of successful generation and validation
    print("\n--- Valid Case ---")
    patient_info_valid = {"patient_id": "P123", "diagnosis": "Hypertension"}
    valid_plan = generate_and_validate_treatment_plan(patient_info_valid)
    if valid_plan:
        print("Successfully generated and validated treatment plan:")
        print(valid_plan.json(indent=2))

    # Example of an invalid LLM response (non-JSON)
    print("\n--- Invalid JSON Case ---")
    patient_info_invalid_json = {"patient_id": "P124", "diagnosis": "Flu", "invalid_case": True}
    invalid_json_plan = generate_and_validate_treatment_plan(patient_info_invalid_json)
    if not invalid_json_plan:
        print("Handled invalid JSON output as expected.")

    # Example of a response that is JSON but might not conform to schema (simulated - would require modifying simulate_llm_response to return malformed JSON)
    # For brevity, we'll rely on the 'invalid_case' for non-JSON and assume the valid case covers schema adherence.
    # In a real scenario, the LLM might return JSON with missing required fields or incorrect types, which Pydantic would catch.