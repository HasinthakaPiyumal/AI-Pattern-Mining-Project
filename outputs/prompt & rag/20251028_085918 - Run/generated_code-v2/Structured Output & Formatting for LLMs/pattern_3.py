import json
from pydantic import BaseModel, Field
from typing import List, Optional

# 1. Pydantic Models for Structured Output
class Medication(BaseModel):
    name: str
    dosage: str
    frequency: str
    notes: Optional[str] = None

class DietaryRecommendation(BaseModel):
    meal_type: str
    description: str
    restrictions: List[str] = Field(default_factory=list)

class ExerciseRoutine(BaseModel):
    activity: str
    duration_minutes: int
    frequency: str
    intensity: Optional[str] = None

class Appointment(BaseModel):
    date: str
    specialist: str
    purpose: str

class VitalSignMonitoring(BaseModel):
    sign: str
    interval: str
    target_range: Optional[str] = None

class CarePlan(BaseModel):
    patient_id: str
    plan_start_date: str
    plan_end_date: str
    medications: List[Medication] = Field(default_factory=list)
    dietary_recommendations: List[DietaryRecommendation] = Field(default_factory=list)
    exercise_routines: List[ExerciseRoutine] = Field(default_factory=list)
    appointments: List[Appointment] = Field(default_factory=list)
    vital_sign_monitoring: List[VitalSignMonitoring] = Field(default_factory=list)
    general_recommendations: Optional[str] = None

# 2. Mock LLM Response (simulating natural language output from an LLM)
def mock_llm_response(patient_data: str) -> str:
    return f"""
    Here is a care plan for the patient based on their data: {patient_data}.

    Medications:
    - Take Metformin 500mg twice daily with meals for diabetes management.
    - Take Lisinopril 10mg once daily in the morning for blood pressure.

    Dietary Recommendations:
    - Breakfast: Oatmeal with berries. Restrictions: Avoid high sugar cereals.
    - Lunch: Grilled chicken salad. Restrictions: Limit processed foods, high sodium.
    - Dinner: Baked salmon with vegetables. Restrictions: Reduce red meat intake.

    Exercise Routines:
    - Walking: 30 minutes, 5 times a week, moderate intensity.
    - Strength Training: 20 minutes, 3 times a week, light intensity.

    Appointments:
    - 2024-03-15, Dr. Smith, Diabetes follow-up.
    - 2024-04-01, Nutritionist, Dietary review.

    Vital Sign Monitoring:
    - Blood Glucose: Daily, Target range: 80-120 mg/dL.
    - Blood Pressure: Weekly, Target range: <130/80 mmHg.

    General Recommendations: Stay hydrated and get adequate rest.
    """

# 3. Natural Language Parsing and Extraction (Simplified for demonstration)
def parse_llm_response(llm_output: str, patient_id: str) -> CarePlan:
    medications = []
    dietary_recommendations = []
    exercise_routines = []
    appointments = []
    vital_sign_monitoring = []
    general_recommendations = ""

    lines = llm_output.split('\n')
    current_section = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue

        if "Medications:" in line:
            current_section = "medications"
            continue
        elif "Dietary Recommendations:" in line:
            current_section = "dietary_recommendations"
            continue
        elif "Exercise Routines:" in line:
            current_section = "exercise_routines"
            continue
        elif "Appointments:" in line:
            current_section = "appointments"
            continue
        elif "Vital Sign Monitoring:" in line:
            current_section = "vital_sign_monitoring"
            continue
        elif "General Recommendations:" in line:
            current_section = "general_recommendations"
            continue
        
        if line.startswith('- '):
            content = line[2:].strip()
            if current_section == "medications":
                parts = content.split(' ', 3)
                if len(parts) >= 3:
                    name = parts[1]
                    dosage_freq_notes = parts[2:]
                    dosage_freq_notes_str = ' '.join(dosage_freq_notes)
                    
                    # Simplified extraction: find first numeric string as dosage, then the next string as frequency
                    dosage_match = next((p for p in dosage_freq_notes_str.split() if any(char.isdigit() for char in p)), "N/A")
                    
                    frequency_match = ""
                    for i, p in enumerate(dosage_freq_notes_str.split()):
                        if p == dosage_match and i + 1 < len(dosage_freq_notes_str.split()):
                            frequency_match = dosage_freq_notes_str.split()[i+1]
                            break

                    medications.append(Medication(name=name, dosage=dosage_match, frequency=frequency_match, notes=content))

            elif current_section == "dietary_recommendations":
                if ":" in content:
                    meal_type, rest = content.split(':', 1)
                    description = rest.split("Restrictions:", 1)[0].strip()
                    restrictions_str = rest.split("Restrictions:", 1)[-1].strip() if "Restrictions:" in rest else ""
                    restrictions = [r.strip() for r in restrictions_str.split(',') if r.strip()]
                    dietary_recommendations.append(DietaryRecommendation(meal_type=meal_type.strip(), description=description, restrictions=restrictions))

            elif current_section == "exercise_routines":
                parts = content.split(', ')
                if len(parts) >= 3:
                    activity = parts[0]
                    duration_match = next((int(s) for s in parts[1].split() if s.isdigit()), 0)
                    frequency = parts[2]
                    intensity = parts[3] if len(parts) > 3 else None
                    exercise_routines.append(ExerciseRoutine(activity=activity, duration_minutes=duration_match, frequency=frequency, intensity=intensity))

            elif current_section == "appointments":
                parts = content.split(', ')
                if len(parts) == 3:
                    date, specialist, purpose = parts
                    appointments.append(Appointment(date=date, specialist=specialist, purpose=purpose))

            elif current_section == "vital_sign_monitoring":
                parts = content.split(', ')
                if len(parts) >= 2:
                    sign = parts[0]
                    interval = parts[1]
                    target_range = parts[2] if len(parts) > 2 else None
                    vital_sign_monitoring.append(VitalSignMonitoring(sign=sign, interval=interval, target_range=target_range))
        elif current_section == "general_recommendations":
            general_recommendations += line + "\n"

    # Using dummy dates for now
    care_plan = CarePlan(
        patient_id=patient_id,
        plan_start_date="2024-03-01",
        plan_end_date="2024-03-31",
        medications=medications,
        dietary_recommendations=dietary_recommendations,
        exercise_routines=exercise_routines,
        appointments=appointments,
        vital_sign_monitoring=vital_sign_monitoring,
        general_recommendations=general_recommendations.strip()
    )
    return care_plan

def generate_patient_care_plan(patient_data: str, patient_id: str) -> str:
    llm_output = mock_llm_response(patient_data)
    care_plan_object = parse_llm_response(llm_output, patient_id)
    
    # Convert Pydantic model to JSON
    json_output = care_plan_object.model_dump_json(indent=2)
    
    print(f"\n--- Generated Structured Care Plan (JSON) for Patient {patient_id} ---")
    print(json_output)
    print("\n--- Placeholder for EHR Integration ---")
    print(f"Initiating secure API call to update EHR for Patient {patient_id} with the generated care plan.")
    
    return json_output

if __name__ == "__main__":
    # Example Usage:
    patient_medical_history = """
    Patient is a 65-year-old male with Type 2 Diabetes and Hypertension. 
    Current blood glucose levels are elevated, and blood pressure is inconsistently controlled. 
    No known drug allergies. 
    """
    patient_current_status = "Currently experiencing fatigue and occasional dizziness."
    
    full_patient_data = f"Medical History: {patient_medical_history}\nCurrent Status: {patient_current_status}"
    
    generated_plan_json = generate_patient_care_plan(full_patient_data, "PAT001")