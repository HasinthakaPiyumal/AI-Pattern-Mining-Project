import re
import json
from pydantic import BaseModel, Field
from typing import List, Optional


class Medication(BaseModel):
    name: str
    dosage: str
    frequency: str


class Appointment(BaseModel):
    date: str
    specialist: str
    purpose: str


class TreatmentPlan(BaseModel):
    patient_id: str
    diagnosis: str
    medications: List[Medication] = Field(default_factory=list)
    appointments: List[Appointment] = Field(default_factory=list)
    dietary_restrictions: List[str] = Field(default_factory=list)
    exercise_recommendations: List[str] = Field(default_factory=list)
    notes: Optional[str] = None


def generate_mock_llm_plan(patient_id: str, diagnosis: str) -> str:
    return (
        f"Treatment Plan for Patient ID: {patient_id}\n"
        f"Diagnosis: {diagnosis}\n\n"
        "Medications:\n"
        "- Aspirin 81mg once daily\n"
        "- Lisinopril 10mg twice a day (morning and evening)\n"
        "\n"
        "Appointments:\n"
        "- 2024-07-15 with Dr. Smith for Cardiology Follow-up\n"
        "- 2024-08-01 with Nutritionist Emily White for Dietary Consultation\n"
        "\n"
        "Dietary Restrictions:\n"
        "- Low sodium diet\n"
        "- Avoid sugary drinks\n"
        "\n"
        "Exercise Recommendations:\n"
        "- Walk 30 minutes daily\n"
        "- Light stretching exercises thrice a week\n"
        "\n"
        "Notes: Monitor blood pressure regularly and report any unusual symptoms."
    )


def parse_natural_language_plan(llm_plan: str, patient_id: str, diagnosis: str) -> TreatmentPlan:
    medications: List[Medication] = []
    appointments: List[Appointment] = []
    dietary_restrictions: List[str] = []
    exercise_recommendations: List[str] = []
    notes: Optional[str] = None

    # Parse Medications
    med_pattern = re.compile(r"- (.*?)\s+(\d+.*?)\s+(once daily|twice a day|thrice a day|every \d+ hours|as needed)")
    for match in med_pattern.finditer(llm_plan):
        name, dosage, frequency = match.groups()
        medications.append(Medication(name=name.strip(), dosage=dosage.strip(), frequency=frequency.strip()))

    # Parse Appointments
    appt_pattern = re.compile(r"- (\d{4}-\d{2}-\d{2}) with (.*?) for (.*?)")
    for match in appt_pattern.finditer(llm_plan):
        date, specialist, purpose = match.groups()
        appointments.append(Appointment(date=date.strip(), specialist=specialist.strip(), purpose=purpose.strip()))

    # Parse Dietary Restrictions
    diet_match = re.search(r"Dietary Restrictions:\n((?:- .*\n)*)", llm_plan)
    if diet_match:
        dietary_restrictions = [item.strip() for item in diet_match.group(1).split('\n') if item.strip()]

    # Parse Exercise Recommendations
    exercise_match = re.search(r"Exercise Recommendations:\n((?:- .*\n)*)", llm_plan)
    if exercise_match:
        exercise_recommendations = [item.strip() for item in exercise_match.group(1).split('\n') if item.strip()]

    # Parse Notes
    notes_match = re.search(r"Notes: (.*)", llm_plan)
    if notes_match:
        notes = notes_match.group(1).strip()

    return TreatmentPlan(
        patient_id=patient_id,
        diagnosis=diagnosis,
        medications=medications,
        appointments=appointments,
        dietary_restrictions=dietary_restrictions,
        exercise_recommendations=exercise_recommendations,
        notes=notes,
    )


if __name__ == "__main__":
    patient_id = "P001"
    diagnosis = "Hypertension and Mild Anxiety"

    # 1. LLM Interaction Layer (Simulated/Mocked)
    natural_language_plan = generate_mock_llm_plan(patient_id, diagnosis)
    print("--- Natural Language Treatment Plan (Mocked LLM Output) ---")
    print(natural_language_plan)
    print("\n" + "="*70 + "\n")

    # 3. Post-processing / Parsing Layer
    structured_plan = parse_natural_language_plan(natural_language_plan, patient_id, diagnosis)

    # 4. Output Generation (and verification of structured data)
    json_output = structured_plan.json(indent=2)
    print("--- Structured Treatment Plan (JSON Output) ---")
    print(json_output)

    print("\n" + "="*70 + "\n")
    print("--- Verification: Accessing structured data ---")
    print(f"Patient ID: {structured_plan.patient_id}")
    print(f"Diagnosis: {structured_plan.diagnosis}")
    if structured_plan.medications:
        print(f"First medication: {structured_plan.medications[0].name}, {structured_plan.medications[0].dosage}, {structured_plan.medications[0].frequency}")
    if structured_plan.appointments:
        print(f"First appointment: {structured_plan.appointments[0].date} with {structured_plan.appointments[0].specialist} for {structured_plan.appointments[0].purpose}")
    if structured_plan.dietary_restrictions:
        print(f"Dietary restriction: {structured_plan.dietary_restrictions[0]}")
    if structured_plan.notes:
        print(f"Notes: {structured_plan.notes}")
