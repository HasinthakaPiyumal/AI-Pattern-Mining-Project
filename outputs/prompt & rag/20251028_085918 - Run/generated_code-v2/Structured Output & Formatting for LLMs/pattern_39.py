
import json
import re

def generate_llm_treatment_plan(patient_summary: str) -> str:
    """
    Simulates an LLM generating a natural language treatment plan based on patient summary.
    In a real application, this would involve an API call to a large language model.
    """
    # This is a mock LLM output for demonstration purposes
    mock_plan = f"""
    Based on the patient summary: "{patient_summary}", here is a personalized treatment plan:

    Medication:
    - Metformin 500mg, twice daily with meals.
    - Lisinopril 10mg, once daily in the morning.
    - Aspirin 81mg, once daily.

    Therapy:
    - Physical therapy sessions, 3 times a week for 8 weeks, focusing on lower back strengthening.
    - Nutritional counseling, bi-weekly for 1 month, to manage blood sugar and cholesterol.

    Dietary Restrictions:
    - Low-sodium diet, limit processed foods.
    - Low-sugar diet, avoid sugary drinks and desserts.
    - Increase fiber intake with fruits, vegetables, and whole grains.

    Follow-up Appointments:
    - Cardiology review: 2 weeks from now (Dr. Smith).
    - Endocrinology review: 1 month from now (Dr. Jones).
    - Physical therapy re-evaluation: After 8 weeks of therapy.

    Important Notes:
    - Monitor blood pressure and blood glucose daily.
    - Report any unusual symptoms immediately.
    """
    return mock_plan

def parse_natural_language_plan(nl_plan: str) -> dict:
    """
    Parses the natural language treatment plan and extracts key components
    into a structured dictionary.
    """
    parsed_data = {
        "medication": [],
        "therapy": [],
        "dietary_restrictions": [],
        "follow_up_appointments": [],
        "important_notes": ""
    }

    # Use regular expressions to extract sections
    medication_match = re.search(r"Medication:\n(.*?)Therapy:", nl_plan, re.DOTALL)
    if medication_match:
        meds = [m.strip() for m in medication_match.group(1).split('-') if m.strip()]
        parsed_data["medication"] = meds

    therapy_match = re.search(r"Therapy:\n(.*?)Dietary Restrictions:", nl_plan, re.DOTALL)
    if therapy_match:
        therapies = [t.strip() for t in therapy_match.group(1).split('-') if t.strip()]
        parsed_data["therapy"] = therapies

    diet_match = re.search(r"Dietary Restrictions:\n(.*?)Follow-up Appointments:", nl_plan, re.DOTALL)
    if diet_match:
        diets = [d.strip() for d in diet_match.group(1).split('-') if d.strip()]
        parsed_data["dietary_restrictions"] = diets

    followup_match = re.search(r"Follow-up Appointments:\n(.*?)Important Notes:", nl_plan, re.DOTALL)
    if followup_match:
        followups = [f.strip() for f in followup_match.group(1).split('-') if f.strip()]
        parsed_data["follow_up_appointments"] = followups

    notes_match = re.search(r"Important Notes:\n(.*)", nl_plan, re.DOTALL)
    if notes_match:
        parsed_data["important_notes"] = notes_match.group(1).strip()

    return parsed_data

def create_json_plan(structured_data: dict) -> str:
    """
    Converts the structured treatment plan data into a JSON string.
    """
    return json.dumps(structured_data, indent=4)

if __name__ == "__main__":
    # Example usage
    patient_info = "45-year-old male with Type 2 Diabetes, hypertension, and chronic lower back pain."

    print("\n--- Generating Natural Language Treatment Plan ---")
    nl_plan_output = generate_llm_treatment_plan(patient_info)
    print(nl_plan_output)

    print("\n--- Parsing Natural Language Plan into Structured Data ---")
    structured_plan = parse_natural_language_plan(nl_plan_output)
    print(json.dumps(structured_plan, indent=4))

    print("\n--- Converting Structured Data to JSON for EHR Integration ---")
    json_plan = create_json_plan(structured_plan)
    print(json_plan)

    print("\n--- Simulation Complete ---")
    print("The JSON output can now be easily integrated into EHR systems for automated tracking and evaluation.")
