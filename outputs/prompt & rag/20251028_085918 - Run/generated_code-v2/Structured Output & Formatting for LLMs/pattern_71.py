from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date
import json

class Medication(BaseModel):
    name: str = Field(..., description="Name of the medication.")
    dosage: str = Field(..., description="Dosage instructions (e.g., '1 pill twice daily').")
    frequency: str = Field(..., description="Frequency of medication (e.g., 'BID', 'Daily').")
    notes: Optional[str] = Field(None, description="Any special notes for this medication.")

class Appointment(BaseModel):
    specialty: str = Field(..., description="Specialty of the follow-up appointment (e.g., 'Cardiology').")
    date: date = Field(..., description="Date of the appointment.")
    time: str = Field(..., description="Time of the appointment (e.g., '10:00 AM').")
    location: str = Field(..., description="Location of the appointment.")
    contact: Optional[str] = Field(None, description="Contact information for scheduling/questions.")

class Instruction(BaseModel):
    category: str = Field(..., description="Category of instruction (e.g., 'Dietary', 'Activity', 'Wound Care').")
    detail: str = Field(..., description="Detailed instruction for the patient.")

class DischargePlan(BaseModel):
    patient_name: str = Field(..., description="Full name of the patient.")
    discharge_date: date = Field(..., description="Date of patient discharge.")
    diagnosis: str = Field(..., description="Primary diagnosis for which the patient was admitted.")
    medications: List[Medication] = Field([], description="List of medications to be taken post-discharge.")
    follow_up_appointments: List[Appointment] = Field([], description="List of scheduled follow-up appointments.")
    activity_restrictions: List[Instruction] = Field([], description="List of activity restrictions.")
    dietary_instructions: List[Instruction] = Field([], description="List of dietary instructions.")
    warning_signs: List[str] = Field([], description="List of warning signs requiring immediate medical attention.")
    additional_notes: Optional[str] = Field(None, description="Any other important notes for the patient or caregivers.")

def generate_discharge_plan_from_notes(clinical_notes: str) -> DischargePlan:
    # In a real application, this would involve calling an LLM with a carefully crafted prompt
    # to output JSON that conforms to the DischargePlan schema.
    # For this example, we simulate the LLM's JSON output.
    
    # Example prompt structure for an LLM (not executed here):
    # prompt = f"""Extract the following information from the clinical notes and format it as a JSON object strictly adhering to this Pydantic schema:\n\n{DischargePlan.model_json_schema()}\n\nClinical Notes: {clinical_notes}\n\nJSON Output:"""
    
    # Simulated LLM output (assuming the LLM successfully generated valid JSON)
    simulated_llm_output_json = '''
    {
        "patient_name": "John Doe",
        "discharge_date": "2023-10-26",
        "diagnosis": "Acute Myocardial Infarction",
        "medications": [
            {
                "name": "Aspirin",
                "dosage": "81 mg",
                "frequency": "Daily",
                "notes": "Take with food."
            },
            {
                "name": "Metoprolol",
                "dosage": "25 mg",
                "frequency": "BID",
                "notes": null
            }
        ],
        "follow_up_appointments": [
            {
                "specialty": "Cardiology",
                "date": "2023-11-15",
                "time": "10:00 AM",
                "location": "Main Hospital Clinic",
                "contact": "(555) 123-4567"
            }
        ],
        "activity_restrictions": [
            {
                "category": "Activity",
                "detail": "Avoid heavy lifting for 4 weeks."
            },
            {
                "category": "Activity",
                "detail": "Light walking encouraged."
            }
        ],
        "dietary_instructions": [
            {
                "category": "Dietary",
                "detail": "Follow a low-sodium diet."
            }
        ],
        "warning_signs": [
            "Chest pain",
            "Shortness of breath",
            "Sudden swelling"
        ],
        "additional_notes": "Patient educated on medication adherence and symptom monitoring."
    }
    '''
    
    # Parse and validate the LLM's JSON output using the Pydantic model
    try:
        discharge_plan = DischargePlan.model_validate_json(simulated_llm_output_json)
        return discharge_plan
    except Exception as e:
        print(f"Error validating LLM output: {e}")
        # In a real scenario, you might log the error, attempt to re-prompt, or use a fallback parser.
        raise

if __name__ == "__main__":
    sample_clinical_notes = (
        "Patient John Doe, 65 y.o., admitted for MI. Discharged today, 2023-10-26. "
        "Medications: Aspirin 81mg daily (with food), Metoprolol 25mg BID. "
        "Follow-up with Cardiology on 2023-11-15 at 10 AM, Main Hospital Clinic, call (555) 123-4567. "
        "Activity: No heavy lifting for 4 weeks, light walking okay. "
        "Diet: Low sodium. Watch for chest pain, SOB, or swelling. "
        "Patient was instructed on all discharge orders and understands the plan."
    )

    print("--- Generating Discharge Plan ---")
    try:
        generated_plan = generate_discharge_plan_from_notes(sample_clinical_notes)
        print("\n--- Structured Discharge Plan (Pydantic Object) ---")
        print(generated_plan)
        
        print("\n--- Structured Discharge Plan (JSON Output) ---")
        print(generated_plan.model_dump_json(indent=4))

    except Exception as e:
        print(f"An error occurred during plan generation: {e}")