import json
import re

def generate_natural_language_care_plan(patient_data: dict) -> str:
    """Simulates an LLM generating a natural language care plan."""
    patient_name = patient_data.get("name", "Patient")
    diagnosis = patient_data.get("diagnosis", "general check-up")
    history = patient_data.get("history", "no significant history")
    preferences = patient_data.get("preferences", "no specific preferences")

    # Simulate LLM output
    care_plan_text = f"""
Dear {patient_name},

Based on your recent {diagnosis} and medical history ({history}), we have developed a personalized care plan for you. Your preferences ({preferences}) have been taken into account.

Medication Schedule:
- Take Aspirin 81mg once daily in the morning.
- Take Vitamin D 2000IU daily with food.

Dietary Restrictions:
- Reduce sodium intake to less than 2300mg per day.
- Avoid processed foods and sugary drinks.
- Increase intake of fresh fruits and vegetables.

Exercise Routine:
- Walk for 30 minutes, 5 times a week.
- Perform light stretching exercises daily for 10 minutes.

Follow-up Appointments:
- Schedule a follow-up with Dr. Smith in 4 weeks.
- Blood test appointment on [Date] at [Time].

Please adhere to this plan for optimal health outcomes. If you have any questions, please contact your healthcare provider.
"""
    return care_plan_text

def post_process_care_plan(natural_language_plan: str) -> dict:
    """Extracts key components from a natural language care plan and structures them into JSON."""
    structured_plan = {
        "medications": [],
        "dietary_restrictions": [],
        "exercise_routine": [],
        "follow_up_appointments": []
    }

    # Extract Medication Schedule
    med_match = re.search(r"Medication Schedule:\n((?:- .*\n)+)", natural_language_plan)
    if med_match:
        meds = [m.strip() for m in med_match.group(1).split('\n') if m.strip()]
        structured_plan["medications"] = meds

    # Extract Dietary Restrictions
    diet_match = re.search(r"Dietary Restrictions:\n((?:- .*\n)+)", natural_language_plan)
    if diet_match:
        diet = [d.strip() for d in diet_match.group(1).split('\n') if d.strip()]
        structured_plan["dietary_restrictions"] = diet

    # Extract Exercise Routine
    exercise_match = re.search(r"Exercise Routine:\n((?:- .*\n)+)", natural_language_plan)
    if exercise_match:
        exercise = [e.strip() for e in exercise_match.group(1).split('\n') if e.strip()]
        structured_plan["exercise_routine"] = exercise

    # Extract Follow-up Appointments
    followup_match = re.search(r"Follow-up Appointments:\n((?:- .*\n)+)", natural_language_plan)
    if followup_match:
        followup = [f.strip() for f in followup_match.group(1).split('\n') if f.strip()]
        structured_plan["follow_up_appointments"] = followup

    return structured_plan

if __name__ == "__main__":
    # Sample Patient Data
    patient_info = {
        "name": "Alice Wonderland",
        "diagnosis": "Hypertension and Vitamin D Deficiency",
        "history": "Family history of heart disease, previous mild anemia",
        "preferences": "prefers walking over intense cardio, dislikes leafy greens"
    }

    print("--- Generating Natural Language Care Plan ---")
    nl_care_plan = generate_natural_language_care_plan(patient_info)
    print(nl_care_plan)
    print("\n" + "="*50 + "\n")

    print("--- Post-processing into Structured JSON ---")
    structured_json_plan = post_process_care_plan(nl_care_plan)
    print(json.dumps(structured_json_plan, indent=4))

    print("\n" + "="*50 + "\n")
    print("Structured plan generated successfully for automated integration.")