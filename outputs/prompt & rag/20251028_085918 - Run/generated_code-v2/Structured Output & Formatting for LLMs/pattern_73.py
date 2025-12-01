import json
from datetime import date
from typing import List, Optional

from pydantic import BaseModel, Field


class Medication(BaseModel):
    name: str
    dosage: str
    instructions: str


class FollowUpAppointment(BaseModel):
    date: date
    department: str
    doctor: Optional[str] = None
    location: str


class DischargePlan(BaseModel):
    patient_id: str
    discharge_date: date
    diagnosis: str
    medications: List[Medication] = Field(default_factory=list)
    follow_up_appointments: List[FollowUpAppointment] = Field(default_factory=list)
    activity_restrictions: Optional[str] = None
    dietary_recommendations: Optional[str] = None
    warning_signs: List[str] = Field(default_factory=list)
    contact_information: str


class LLMDischargePlanner:
    def _generate_raw_plan_from_llm(self, patient_info: str) -> str:
        # This method simulates an LLM call.
        # In a real application, this would involve calling an actual LLM API
        # with a carefully crafted prompt to ensure structured JSON output.
        # For this example, we return a hardcoded JSON string.

        # Based on patient_info (which could be free-form text),
        # the LLM would extract and format the details.
        if "John Doe" in patient_info:
            return json.dumps({
                "patient_id": "PD12345",
                "discharge_date": "2023-10-27",
                "diagnosis": "Acute Bronchitis",
                "medications": [
                    {
                        "name": "Amoxicillin",
                        "dosage": "250mg",
                        "instructions": "Take one capsule every 8 hours for 7 days with food."
                    },
                    {
                        "name": "Albuterol Inhaler",
                        "dosage": "90mcg",
                        "instructions": "Two puffs every 4-6 hours as needed for shortness of breath."
                    }
                ],
                "follow_up_appointments": [
                    {
                        "date": "2023-11-10",
                        "department": "Pulmonology",
                        "doctor": "Dr. Emily White",
                        "location": "Hospital Clinic 3"
                    }
                ],
                "activity_restrictions": "Avoid strenuous activity for 1 week.",
                "dietary_recommendations": "Stay hydrated, soft diet initially.",
                "warning_signs": [
                    "Increased difficulty breathing",
                    "Persistent fever",
                    "Chest pain"
                ],
                "contact_information": "Hospital Discharge Coordinator: 555-123-4567"
            })
        else:
            # Example for another patient or generic response
            return json.dumps({
                "patient_id": "GENERIC001",
                "discharge_date": "2023-10-28",
                "diagnosis": "Observation Stay",
                "medications": [],
                "follow_up_appointments": [],
                "activity_restrictions": "No restrictions.",
                "dietary_recommendations": "Normal diet.",
                "warning_signs": ["Any new or worsening symptoms."],
                "contact_information": "On-call Doctor: 555-987-6543"
            })

    def create_structured_discharge_plan(self, patient_info: str) -> DischargePlan:
        raw_json_output = self._generate_raw_plan_from_llm(patient_info)
        parsed_data = json.loads(raw_json_output)
        # Validate and parse the data using the Pydantic model
        structured_plan = DischargePlan(**parsed_data)
        return structured_plan


def main():
    planner = LLMDischargePlanner()

    # Sample patient data
    patient_data_john_doe = (
        "Patient Name: John Doe\n" "Diagnosis: Acute Bronchitis\n" "Discharge Date: October 27, 2023\n" "Medications: Amoxicillin (250mg, 3 times a day for 7 days with food), Albuterol Inhaler (2 puffs every 4-6 hours as needed).\n" "Follow-up: Pulmonology, Dr. Emily White, Nov 10, Hospital Clinic 3.\n" "Restrictions: No strenuous activity for 1 week.\n" "Diet: Stay hydrated, soft diet initially.\n" "Warning Signs: Difficulty breathing, fever, chest pain.\n" "Contact: Discharge Coordinator at 555-123-4567."
    )

    patient_data_jane_smith = (
        "Patient Name: Jane Smith\n" "Diagnosis: Minor observation\n" "Discharge Date: October 28, 2023\n" "Medications: None.\n" "Follow-up: None.\n" "Restrictions: None.\n" "Diet: Normal.\n" "Warning Signs: Any new symptoms.\n" "Contact: On-call Doctor: 555-987-6543"
    )

    print("--- Generating Discharge Plan for John Doe ---")
    try:
        john_doe_plan = planner.create_structured_discharge_plan(patient_data_john_doe)
        print(json.dumps(john_doe_plan.model_dump(), indent=2))
        print("\nPlan successfully generated and validated for John Doe.")
    except Exception as e:
        print(f"Error generating plan for John Doe: {e}")

    print("\n--- Generating Discharge Plan for Jane Smith ---")
    try:
        jane_smith_plan = planner.create_structured_discharge_plan(patient_data_jane_smith)
        print(json.dumps(jane_smith_plan.model_dump(), indent=2))
        print("\nPlan successfully generated and validated for Jane Smith.")
    except Exception as e:
        print(f"Error generating plan for Jane Smith: {e}")


if __name__ == "__main__":
    main()