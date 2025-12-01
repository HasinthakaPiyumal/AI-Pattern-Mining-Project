from pydantic import BaseModel, Field
from typing import List, Optional
import json


class Medication(BaseModel):
    name: str = Field(..., description="Name of the medication")
    dosage: str = Field(..., description="Dosage instructions for the medication")
    frequency: str = Field(..., description="Frequency of medication intake")
    notes: Optional[str] = Field(None, description="Additional notes for the medication")


class DietaryRecommendation(BaseModel):
    item: str = Field(..., description="Specific food item or category")
    category: str = Field(..., description="Category of the recommendation (e.g., 'avoid', 'include', 'limit')")
    notes: Optional[str] = Field(None, description="Additional notes for the dietary recommendation")


class ExerciseRoutine(BaseModel):
    activity: str = Field(..., description="Type of exercise activity")
    duration_minutes: int = Field(..., description="Duration of the exercise in minutes")
    frequency: str = Field(..., description="Frequency of the exercise routine")
    notes: Optional[str] = Field(None, description="Additional notes for the exercise routine")


class FollowUpAppointment(BaseModel):
    date: str = Field(..., description="Date of the follow-up appointment (YYYY-MM-DD)")
    purpose: str = Field(..., description="Purpose of the follow-up")
    notes: Optional[str] = Field(None, description="Additional notes for the appointment")


class SymptomMonitoring(BaseModel):
    symptom: str = Field(..., description="Symptom to be monitored")
    frequency: str = Field(..., description="Frequency of symptom monitoring")
    notes: Optional[str] = Field(None, description="Additional notes for symptom monitoring")


class TreatmentPlan(BaseModel):
    patient_id: str = Field(..., description="Unique identifier for the patient")
    disease: str = Field(..., description="Name of the chronic disease")
    medications: List[Medication] = Field([], description="List of prescribed medications")
    dietary_recommendations: List[DietaryRecommendation] = Field([], description="List of dietary recommendations")
    exercise_routines: List[ExerciseRoutine] = Field([], description="List of exercise routines")
    follow_up_appointments: List[FollowUpAppointment] = Field([], description="List of follow-up appointments")
    symptom_monitoring: List[SymptomMonitoring] = Field([], description="List of symptom monitoring instructions")
    overall_notes: Optional[str] = Field(None, description="General notes about the treatment plan")


def _simulate_llm_response(prompt: str) -> str:
    # In a real application, this would involve an actual LLM API call (e.g., OpenAI, Gemini)
    # For demonstration, we return a hardcoded JSON string that conforms to the TreatmentPlan schema.
    print(f"Simulating LLM call with prompt:\n{prompt}\n")
    return json.dumps({
        "patient_id": "PAT001",
        "disease": "Type 2 Diabetes",
        "medications": [
            {"name": "Metformin", "dosage": "500mg", "frequency": "Twice daily", "notes": "Take with meals"},
            {"name": "Lisinopril", "dosage": "10mg", "frequency": "Once daily", "notes": "For blood pressure control"}
        ],
        "dietary_recommendations": [
            {"item": "Processed sugars", "category": "avoid", "notes": "Limit intake of sugary drinks and desserts"},
            {"item": "Whole grains", "category": "include", "notes": "Opt for brown rice, oats, and whole wheat bread"},
            {"item": "Lean proteins", "category": "include", "notes": "Chicken, fish, tofu"}
        ],
        "exercise_routines": [
            {"activity": "Brisk walking", "duration_minutes": 30, "frequency": "5 times a week", "notes": "Aim for moderate intensity"},
            {"activity": "Strength training", "duration_minutes": 20, "frequency": "2 times a week", "notes": "Bodyweight or light weights"}
        ],
        "follow_up_appointments": [
            {"date": "2023-11-15", "purpose": "Diabetes check-up", "notes": "Fasting blood glucose test required"},
            {"date": "2024-02-15", "purpose": "Cardiology review", "notes": "ECG might be performed"}
        ],
        "symptom_monitoring": [
            {"symptom": "Blood glucose levels", "frequency": "Daily (fasting and post-meal)", "notes": "Record readings in logbook"},
            {"symptom": "Blood pressure", "frequency": "Weekly", "notes": "Record readings"},
            {"symptom": "Foot checks", "frequency": "Daily", "notes": "Look for cuts, blisters, or sores"}
        ],
        "overall_notes": "Patient should adhere strictly to medication and lifestyle changes. Regular monitoring is crucial."
    })

def generate_treatment_plan(patient_details: dict, disease_info: dict) -> TreatmentPlan:
    """
    Generates a structured treatment plan for a chronic disease using an LLM.
    The LLM is prompted to output in a specific JSON format defined by Pydantic models.
    """
    schema_json = json.dumps(TreatmentPlan.model_json_schema(), indent=2)

    prompt = f"""
    Generate a comprehensive treatment plan for a patient with a chronic disease.
    The output MUST be a JSON object that strictly adheres to the following Pydantic schema:

    {schema_json}

    Patient Details:
    {json.dumps(patient_details, indent=2)}

    Disease Information:
    {json.dumps(disease_info, indent=2)}

    Ensure all relevant sections (medications, diet, exercise, follow-ups, symptom monitoring) are populated appropriately.
    """

    llm_output_json = _simulate_llm_response(prompt)

    try:
        parsed_data = json.loads(llm_output_json)
        treatment_plan = TreatmentPlan(**parsed_data)
        return treatment_plan
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM output is not valid JSON: {e}")
    except Exception as e:
        raise ValueError(f"Failed to parse or validate LLM output against schema: {e}")


if __name__ == "__main__":
    # Example Usage
    patient_info = {
        "name": "Jane Doe",
        "age": 55,
        "gender": "Female",
        "medical_history": ["Hypertension", "High Cholesterol"],
        "allergies": ["Penicillin"]
    }

    disease_context = {
        "disease_name": "Type 2 Diabetes",
        "severity": "Moderate",
        "current_symptoms": ["Fatigue", "Increased Thirst"],
        "recent_lab_results": {"HbA1c": "7.5%", "Fasting Glucose": "140 mg/dL"}
    }

    try:
        plan = generate_treatment_plan(patient_info, disease_context)
        print("\nSuccessfully generated and validated Treatment Plan:")
        print(plan.model_dump_json(indent=2))
    except ValueError as e:
        print(f"Error generating treatment plan: {e}")

    # Example of an invalid LLM response (would raise a validation error)
    # class MockInvalidTreatmentPlan(BaseModel):
    #     invalid_field: str
    #     medications: List[Medication]

    # def _simulate_llm_invalid_response(prompt: str) -> str:
    #     return json.dumps({"invalid_field": "some_value", "medications": [{"name": "Test", "dosage": "1", "frequency": "1", "notes": ""}]})

    # original_simulate_llm_response = _simulate_llm_response
    # globals()['_simulate_llm_response'] = _simulate_llm_invalid_response

    # try:
    #     print("\nAttempting to generate with an intentionally invalid LLM response...")
    #     plan = generate_treatment_plan(patient_info, disease_context)
    #     print(plan.model_dump_json(indent=2))
    # except ValueError as e:
    #     print(f"Successfully caught error for invalid LLM output: {e}")
    # finally:
    #     globals()['_simulate_llm_response'] = original_simulate_llm_response # Restore original

