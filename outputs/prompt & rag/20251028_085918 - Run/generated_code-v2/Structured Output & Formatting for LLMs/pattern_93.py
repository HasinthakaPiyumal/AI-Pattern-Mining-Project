import re
from datetime import date, time, datetime
from typing import List, Optional
from data_models import CarePlan, Medication, DietaryRecommendation, ExerciseRoutine, Appointment, Test # Assuming data_models.py is in the same directory

def parse_natural_language_plan_to_json(patient_id: str, nl_plan: str) -> CarePlan:
    """
    Parses a natural language care plan and structures it into a Pydantic CarePlan object.
    """
    medications: List[Medication] = []
    dietary_recommendations: List[DietaryRecommendation] = []
    exercise_routines: List[ExerciseRoutine] = []
    appointments: List[Appointment] = []
    tests: List[Test] = []
    goals: List[str] = []

    # Extract Overall Goal
    goal_match = re.search(r"\*\*Overall Goal:\*\*\s*(.*?)\n", nl_plan)
    if goal_match:
        goals.append(goal_match.group(1).strip())

    # Extract Medications
    med_section_match = re.search(r"\*\*Medications:\*\*\n(.*?)(?=\n\n\*\*|\Z)", nl_plan, re.DOTALL)
    if med_section_match:
        med_items = re.findall(r"\*\s*Take (.*?)\s*(?:once daily|twice daily|three times daily|four times daily|daily|in the morning|in the evening|as needed)\s*(?:for (\d+ days?|until further notice)\s*)?(?:\. Notes: (.*))?", med_section_match.group(1), re.IGNORECASE)
        for item in med_items:
            name_dosage_freq = item[0].strip()
            dosage = ""
            name = ""
            # Attempt to split name and dosage, handling various formats
            dose_match = re.search(r"(.*?)\s*(\d+m?g(?:/\w+)?)\s*", name_dosage_freq)
            if dose_match:
                name = dose_match.group(1).strip()
                dosage = dose_match.group(2).strip()
            else:
                name = name_dosage_freq
                dosage = "N/A"

            frequency_match = re.search(r"(once daily|twice daily|three times daily|four times daily|daily|in the morning|in the evening|as needed)", name_dosage_freq, re.IGNORECASE)
            frequency = frequency_match.group(1).lower() if frequency_match else "Unknown"

            notes = item[2].strip() if item[2] else None
            medications.append(Medication(name=name, dosage=dosage, frequency=frequency, notes=notes))

    # Extract Dietary Recommendations
    diet_section_match = re.search(r"\*\*Dietary Recommendations:\*\*\n(.*?)(?=\n\n\*\*|\Z)", nl_plan, re.DOTALL)
    if diet_section_match:
        diet_text = diet_section_match.group(1)
        description_match = re.search(r"\*\s*(Follow a|Aim for)\s*(.*?)(?:\.)", diet_text, re.IGNORECASE)
        description = description_match.group(2).strip() if description_match else "N/A"
        type_match = re.search(r"(low-sodium|diabetic-friendly|mediterranean|DASH)", description, re.IGNORECASE)
        diet_type = type_match.group(1).capitalize() if type_match else "General"

        foods_to_include = re.findall(r"\*\s*Include plenty of\s*(.*?)(?:\.|,)", diet_text)
        foods_to_avoid = re.findall(r"\*\s*Avoid\s*(.*?)(?:\.|,)", diet_text)
        
        dietary_recommendations.append(DietaryRecommendation(
            type=diet_type,
            description=description,
            foods_to_include=[f.strip() for f in foods_to_include[0].split(',')] if foods_to_include else [],
            foods_to_avoid=[f.strip() for f in foods_to_avoid[0].split(',')] if foods_to_avoid else []
        ))

    # Extract Exercise Routines
    exercise_section_match = re.search(r"\*\*Exercise Routine:\*\*\n(.*?)(?=\n\n\*\*|\Z)", nl_plan, re.DOTALL)
    if exercise_section_match:
        exercise_items = re.findall(r"\*\s*(.*?)\s*for (\d+) minutes?, (\d+) times? a week(?:\. Notes: (.*))?", exercise_section_match.group(1), re.IGNORECASE)
        for item in exercise_items:
            activity = item[0].strip()
            duration = int(item[1])
            frequency = f"{item[2]} times a week"
            notes = item[3].strip() if item[3] else None
            exercise_routines.append(ExerciseRoutine(activity=activity, duration_minutes=duration, frequency=frequency, notes=notes))

    # Extract Appointments
    app_section_match = re.search(r"\*\*Appointments:\*\*\n(.*?)(?=\n\n\*\*|\Z)", nl_plan, re.DOTALL)
    if app_section_match:
        app_items = re.findall(r"\*\s*(?:Follow-up with|Schedule a consultation with) (.*?) on (\d{4}-\d{2}-\d{2})(?: at (\d{2}:\d{2}) (?:AM|PM))?(?: for (.*?))?(?: before (\d{4}-\d{2}-\d{2}))?", app_section_match.group(1), re.IGNORECASE)
        for item in app_items:
            specialty = item[0].strip()
            app_date = datetime.strptime(item[1], "%Y-%m-%d").date()
            app_time = datetime.strptime(item[2], "%H:%M").time() if item[2] else time(0, 0) # Default to midnight if no time
            reason = item[3].strip() if item[3] else "General follow-up"
            appointments.append(Appointment(date=app_date, time=app_time, specialty=specialty, reason=reason))

    # Extract Tests
    test_section_match = re.search(r"\*\*Tests:\*\*\n(.*?)(?=\n\n\*\*|\Z)", nl_plan, re.DOTALL)
    if test_section_match:
        test_items = re.findall(r"\*\s*(?:Complete a|Undergo a) (.*?) \((.*?)\) by (\d{4}-\d{2}-\d{2})\. Reason: (.*?)\.", test_section_match.group(1), re.IGNORECASE)
        for item in test_items:
            name = item[0].strip()
            test_type = item[1].strip()
            due_date = datetime.strptime(item[2], "%Y-%m-%d").date()
            reason = item[3].strip()
            tests.append(Test(name=name, type=test_type, reason=reason, due_date=due_date))

    return CarePlan(
        patient_id=patient_id,
        overview="Preliminary care plan generated by AI.", # This could be extracted more robustly
        medications=medications,
        dietary_recommendations=dietary_recommendations,
        exercise_routines=exercise_routines,
        appointments=appointments,
        tests=tests,
        goals=goals
    )

# Example Usage (for testing)
if __name__ == "__main__":
    from llm_service import generate_natural_language_plan

    sample_patient_input = "Patient has hypertension and wants to improve heart health. Needs medication for blood pressure, diet advice, and exercise. Also needs a cardiology follow-up and a blood test."
    nl_plan = generate_natural_language_plan(sample_patient_input)
    print("\n--- Generated Natural Language Plan ---")
    print(nl_plan)

    structured_plan = parse_natural_language_plan_to_json(patient_id="PAT001", nl_plan=nl_plan)
    print("\n--- Structured JSON Plan ---")
    print(structured_plan.model_dump_json(indent=2))
