from pydantic import BaseModel, Field
import json

# 1. Define Pydantic models for the structured care plan components
class Medication(BaseModel):
    name: str = Field(..., description="Name of the medication")
    dosage: str = Field(..., description="Dosage instructions")
    frequency: str = Field(..., description="Frequency of administration")

class TreatmentProcedure(BaseModel):
    name: str = Field(..., description="Name of the procedure")
    description: str = Field(..., description="Detailed description of the procedure")
    schedule: str = Field(..., description="Schedule for the procedure")

class Appointment(BaseModel):
    type: str = Field(..., description="Type of appointment (e.g., Follow-up, Specialist)")
    date: str = Field(..., description="Date of the appointment (YYYY-MM-DD)")
    time: str = Field(..., description="Time of the appointment (HH:MM)")
    location: str = Field(..., description="Location of the appointment")

class DietaryRestriction(BaseModel):
    restriction: str = Field(..., description="Specific dietary restriction")
    reason: str = Field(..., description="Reason for the restriction")

class ActivityRecommendation(BaseModel):
    activity: str = Field(..., description="Recommended activity")
    duration: str = Field(..., description="Recommended duration or intensity")
    frequency: str = Field(..., description="Frequency of activity")

class CarePlan(BaseModel):
    patient_id: str = Field(..., description="Unique identifier for the patient")
    plan_id: str = Field(..., description="Unique identifier for the care plan")
    medications: list[Medication] = Field(default_factory=list, description="List of prescribed medications")
    treatment_procedures: list[TreatmentProcedure] = Field(default_factory=list, description="List of treatment procedures")
    follow_up_appointments: list[Appointment] = Field(default_factory=list, description="List of follow-up appointments")
    dietary_restrictions: list[DietaryRestriction] = Field(default_factory=list, description="List of dietary restrictions")
    activity_recommendations: list[ActivityRecommendation] = Field(default_factory=list, description="List of activity recommendations")
    notes: str = Field(default="", description="Any additional notes for the care plan")

# 2. Simulate LLM interaction
def simulate_llm_response(natural_language_care_plan: str) -> str:
    # In a real application, this would involve an actual LLM API call
    # with prompt engineering to instruct JSON output.
    # For this demonstration, we'll return a hardcoded JSON string
    # that a well-behaved LLM would generate based on a prompt.

    # The LLM's prompt would look something like this:
    # "Convert the following patient care plan into a JSON object adhering to this schema:
    # { "patient_id": "string", "plan_id": "string", "medications": [...], ... }
    # Care Plan: """"""" + natural_language_care_plan + """""""
    
    # This is a simulated LLM output for demonstration purposes.
    # It closely matches the CarePlan Pydantic model.
    simulated_json_output = {
        "patient_id": "PAT-001",
        "plan_id": "CP-2023-11-08-001",
        "medications": [
            {
                "name": "Aspirin",
                "dosage": "81 mg",
                "frequency": "Once daily"
            },
            {
                "name": "Metformin",
                "dosage": "500 mg",
                "frequency": "Twice daily with meals"
            }
        ],
        "treatment_procedures": [
            {
                "name": "Blood Pressure Monitoring",
                "description": "Monitor blood pressure daily at home",
                "schedule": "Daily"
            }
        ],
        "follow_up_appointments": [
            {
                "type": "Cardiology Follow-up",
                "date": "2023-12-01",
                "time": "10:00",
                "location": "Hospital Clinic A"
            }
        ],
        "dietary_restrictions": [
            {
                "restriction": "Low Sodium",
                "reason": "Hypertension management"
            }
        ],
        "activity_recommendations": [
            {
                "activity": "Brisk Walking",
                "duration": "30 minutes",
                "frequency": "Most days of the week"
            }
        ],
        "notes": "Patient needs education on medication adherence and dietary changes."
    }
    return json.dumps(simulated_json_output, indent=2)

# Main application logic
if __name__ == "__main__":
    # Example natural language care plan (input)
    natural_language_care_plan_example = (
        "Patient PAT-001 needs a care plan. They should take 81mg Aspirin once daily "
        "and 500mg Metformin twice daily with meals. Daily blood pressure monitoring "
        "is required. A cardiology follow-up is scheduled for December 1, 2023, at "
        "10 AM at Hospital Clinic A. The patient must follow a low-sodium diet "
        "due to hypertension. Brisk walking for 30 minutes most days of the week is "
        "recommended. Important: Educate the patient on medication and diet."
    )

    print("--- Natural Language Care Plan ---")
    print(natural_language_care_plan_example)
    print("\n" + "="*40 + "\n")

    # Simulate LLM generating structured output
    print("--- Simulating LLM Structured Output Generation ---")
    llm_generated_json_string = simulate_llm_response(natural_language_care_plan_example)
    print(llm_generated_json_string)
    print("\n" + "="*40 + "\n")

    # 4. Output Parsing & Validation using Pydantic
    print("--- Parsing and Validating with Pydantic ---")
    try:
        # Parse the JSON string into a Python dictionary first
        parsed_data = json.loads(llm_generated_json_string)
        # Then validate and create a Pydantic model instance
        structured_care_plan = CarePlan(**parsed_data)
        print("Structured Care Plan successfully parsed and validated!")
        print("\n--- Validated Structured Care Plan (Pydantic Object) ---")
        print(structured_care_plan.model_dump_json(indent=2))
        print("\nPatient ID:", structured_care_plan.patient_id)
        print("Number of Medications:", len(structured_care_plan.medications))
        print("First Medication Name:", structured_care_plan.medications[0].name)
        print("\n" + "="*40 + "\n")

        # 5. Structured Output (ready for EHR integration, evaluation, etc.)
        print("--- Structured Output ready for integration --- ")
        # This Pydantic object can be easily converted back to JSON if needed
        print(json.dumps(structured_care_plan.model_dump(), indent=2))

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from LLM: {e}")
    except Exception as e:
        print(f"Error validating structured care plan with Pydantic: {e}")