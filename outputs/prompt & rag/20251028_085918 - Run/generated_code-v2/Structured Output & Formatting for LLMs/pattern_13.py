import re
from typing import List, Optional
from pydantic import BaseModel, Field


class Medication(BaseModel):
    name: str
    dosage: str
    frequency: str
    duration: str
    route: Optional[str]


class Appointment(BaseModel):
    type: str
    specialist: str
    frequency: Optional[str]


class Therapy(BaseModel):
    type: str
    duration: str
    frequency: str
    details: Optional[str]


class DietaryRecommendation(BaseModel):
    description: str


class ExercisePlan(BaseModel):
    type: str
    intensity: str
    duration: str
    frequency: str


class FollowUpInstruction(BaseModel):
    description: str


class TreatmentPlan(BaseModel):
    patient_id: str
    diagnosis: str
    medications: List[Medication] = []
    appointments: List[Appointment] = []
    therapies: List[Therapy] = []
    dietary_recommendations: List[DietaryRecommendation] = []
    exercise_plans: List[ExercisePlan] = []
    follow_up_instructions: List[FollowUpInstruction] = []


def simulate_llm_plan_generation(patient_data: dict) -> str:
    patient_name = patient_data.get("name", "Patient")
    diagnosis = patient_data.get("diagnosis", "unknown condition")
    
    return f"""Based on {patient_name}'s {diagnosis} diagnosis, here is a personalized treatment plan:

Medications:
- Take Atorvastatin 20mg once daily for 6 months orally.
- Take Metformin 500mg twice a day for 3 months with meals.

Appointments:
- Follow-up with Cardiologist in 3 weeks.
- Specialist referral to a Nutritionist every month for 6 months.

Therapies:
- Physical Therapy sessions, 30 minutes, twice a week for 8 weeks, focusing on knee strengthening exercises.

Dietary Recommendations:
- Reduce saturated fat intake and increase fiber-rich foods.

Exercise Plans:
- Moderate intensity walking, 30 minutes, 5 times a week.

Follow-up Instructions:
- Monitor blood pressure daily and report any readings above 140/90.
- Contact emergency services if experiencing chest pain.
"""

def extract_structured_plan(patient_id: str, diagnosis: str, natural_language_plan: str) -> TreatmentPlan:
    medications = []
    appointments = []
    therapies = []
    dietary_recommendations = []
    exercise_plans = []
    follow_up_instructions = []

    # Regex for Medications
    medication_pattern = re.compile(r"- Take (\w+) (\d+mg) (\w+ \w+) for (\d+ \w+) (?:orally\.)?")
    for match in medication_pattern.finditer(natural_language_plan):
        medications.append(Medication(
            name=match.group(1),
            dosage=match.group(2),
            frequency=match.group(3),
            duration=match.group(4),
            route="orally" # Assuming orally if not specified or extracted simply
        ))

    # Regex for Appointments
    appointment_pattern = re.compile(r"- (\w+-\w+) with (\w+) in (\d+ \w+)\.")
    appointment_pattern_specialist = re.compile(r"- Specialist referral to a (\w+) (\w+ \w+) for (\d+ \w+)\.")

    for match in appointment_pattern.finditer(natural_language_plan):
        appointments.append(Appointment(
            type=match.group(1).replace("-", " "),
            specialist=match.group(2),
            frequency=f"in {match.group(3)}"
        ))
    for match in appointment_pattern_specialist.finditer(natural_language_plan):
        appointments.append(Appointment(
            type="Specialist referral",
            specialist=match.group(1),
            frequency=f"{match.group(2)} for {match.group(3)}"
        ))

    # Regex for Therapies
    therapy_pattern = re.compile(r"- (\w+ \w+) sessions, (\d+ minutes), (\w+ \w+) for (\d+ \w+), focusing on (.*?).")
    for match in therapy_pattern.finditer(natural_language_plan):
        therapies.append(Therapy(
            type=match.group(1),
            duration=match.group(2),
            frequency=match.group(3),
            details=match.group(5)
        ))

    # Regex for Dietary Recommendations
    dietary_pattern = re.compile(r"- (Reduce .*\.)")
    for match in dietary_pattern.finditer(natural_language_plan):
        dietary_recommendations.append(DietaryRecommendation(
            description=match.group(1)
        ))

    # Regex for Exercise Plans
    exercise_pattern = re.compile(r"- (\w+ intensity \w+), (\d+ minutes), (\d+ \w+ a week).")
    for match in exercise_pattern.finditer(natural_language_plan):
        exercise_plans.append(ExercisePlan(
            type=match.group(1).split(" ")[2],
            intensity=match.group(1).split(" ")[0],
            duration=match.group(2),
            frequency=match.group(3)
        ))

    # Regex for Follow-up Instructions
    follow_up_pattern = re.compile(r"- (Monitor .*\.)")
    follow_up_pattern_emergency = re.compile(r"- (Contact emergency .*\.)")

    for match in follow_up_pattern.finditer(natural_language_plan):
        follow_up_instructions.append(FollowUpInstruction(
            description=match.group(1)
        ))
    for match in follow_up_pattern_emergency.finditer(natural_language_plan):
        follow_up_instructions.append(FollowUpInstruction(
            description=match.group(1)
        ))

    return TreatmentPlan(
        patient_id=patient_id,
        diagnosis=diagnosis,
        medications=medications,
        appointments=appointments,
        therapies=therapies,
        dietary_recommendations=dietary_recommendations,
        exercise_plans=exercise_plans,
        follow_up_instructions=follow_up_instructions
    )


if __name__ == "__main__":
    # Sample Patient Data
    patient_data = {
        "name": "Jane Doe",
        "age": 55,
        "diagnosis": "Type 2 Diabetes and Hypercholesterolemia",
        "medical_history": "Hypertension, family history of heart disease",
        "symptoms": "Fatigue, increased thirst",
        "lab_results": {"glucose": "high", "cholesterol": "high"},
        "preferences": "prefers walking over strenuous exercise"
    }

    patient_id = "JD556789"
    diagnosis = patient_data["diagnosis"]

    # Stage 1: Simulate LLM Plan Generation
    print("\n--- Stage 1: LLM-based Plan Generation (Natural Language) ---")
    natural_language_plan = simulate_llm_plan_generation(patient_data)
    print(natural_language_plan)

    # Stage 2: Structured Extraction and Formatting
    print("\n--- Stage 2: Structured Extraction and Formatting ---")
    structured_plan = extract_structured_plan(patient_id, diagnosis, natural_language_plan)
    print("\nStructured Treatment Plan (JSON):\n")
    print(structured_plan.json(indent=2))

    print("\n--- Verification ---")
    print(f"Patient ID: {structured_plan.patient_id}")
    print(f"Diagnosis: {structured_plan.diagnosis}")
    print(f"Number of Medications: {len(structured_plan.medications)}")
    if structured_plan.medications:
        print(f"  First Medication: {structured_plan.medications[0].name}, {structured_plan.medications[0].dosage}")
    print(f"Number of Appointments: {len(structured_plan.appointments)}")
    if structured_plan.appointments:
        print(f"  First Appointment: {structured_plan.appointments[0].type} with {structured_plan.appointments[0].specialist}")