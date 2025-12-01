import json
import re

def llm_generate_care_plan(patient_data: dict) -> str:
    patient_name = patient_data.get("name", "Patient")
    medical_history = patient_data.get("history", "general check-up")
    symptoms = patient_data.get("symptoms", "mild cough")
    recommendations = patient_data.get("recommendations", "rest and fluids")

    plan_text = f"""
    --- Patient Care Plan for {patient_name} ---

    Based on the patient's medical history ({medical_history}), current symptoms ({symptoms}),
    and doctor's recommendations ({recommendations}), the following care plan is proposed:

    Medication Schedule:
    - Amoxicillin 500mg, twice a day for 7 days, with food.
    - Paracetamol 500mg, as needed for fever, every 6 hours.
    - Ibuprofen 200mg, for pain relief, up to 3 times a day.

    Follow-up Appointments:
    - Follow-up with GP in 7 days to review progress (scheduled for 2024-08-10, 10:00 AM).
    - Specialist consultation with Pulmonologist in 3 weeks (appointment details to be confirmed, likely 2024-08-25).

    Dietary Restrictions:
    - Avoid spicy and fried foods.
    - Increase intake of fluids (water, herbal teas, broths).
    - Consume soft, easily digestible foods like soups and porridges.

    Exercise Routine:
    - Light walking for 15-20 minutes daily, if symptoms allow.
    - Avoid strenuous activities for the next 2 weeks.
    - Gentle stretching exercises can be performed.

    Symptom Monitoring Instructions:
    - Monitor body temperature daily and record any fever spikes (morning and evening).
    - Note frequency and severity of cough.
    - Report any difficulty breathing, chest pain, or worsening symptoms immediately to your doctor.

    --- End of Plan ---
    """
    return plan_text

def parse_care_plan_to_json(natural_language_plan: str) -> dict:
    structured_plan = {
        "medication_schedule": [],
        "follow_up_appointments": [],
        "dietary_restrictions": [],
        "exercise_routine": [],
        "symptom_monitoring_instructions": []
    }

    section_patterns = {
        "Medication Schedule": "medication_schedule",
        "Follow-up Appointments": "follow_up_appointments",
        "Dietary Restrictions": "dietary_restrictions",
        "Exercise Routine": "exercise_routine",
        "Symptom Monitoring Instructions": "symptom_monitoring_instructions",
    }
    
    split_pattern = '|'.join([f'({re.escape(s)}:)' for s in section_patterns.keys()])
    parts = re.split(f'({split_pattern})', natural_language_plan, flags=re.IGNORECASE)

    current_section_key = None
    for part in parts:
        if not part.strip():
            continue

        found_header = False
        for header_text, key_name in section_patterns.items():
            if re.match(re.escape(header_text) + r':', part.strip(), re.IGNORECASE):
                current_section_key = key_name
                found_header = True
                break
        
        if found_header:
            continue

        if current_section_key:
            items = [item.strip() for item in part.split('\n-') if item.strip()]
            structured_plan[current_section_key].extend(items)
            
    for key in structured_plan:
        structured_plan[key] = [item for item in structured_plan[key] if item.strip()]

    return structured_plan

if __name__ == "__main__":
    patient_info = {
        "name": "Jane Doe",
        "history": "Chronic bronchitis, type 2 diabetes",
        "symptoms": "Persistent dry cough, occasional shortness of breath, elevated blood sugar",
        "recommendations": "Bronchodilator, dietary management for diabetes, regular exercise"
    }

    print("--- Generating Natural Language Care Plan ---")
    natural_plan = llm_generate_care_plan(patient_info)
    print(natural_plan)

    print("\n--- Parsing Natural Language Plan to Structured JSON ---")
    structured_plan_dict = parse_care_plan_to_json(natural_plan)
    structured_plan_json = json.dumps(structured_plan_dict, indent=4)
    print(structured_plan_json)

    print("\n--- Automated Evaluation/Integration Placeholder ---")
    print("Structured plan ready for EHR integration or automated evaluation.")