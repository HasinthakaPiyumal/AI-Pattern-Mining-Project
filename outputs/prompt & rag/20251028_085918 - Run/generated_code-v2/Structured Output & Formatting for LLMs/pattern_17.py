import json
import re

def get_patient_data():
    return {
        "patient_id": "P001",
        "name": "Alice Smith",
        "age": 65,
        "diagnosis": "Type 2 Diabetes, Hypertension",
        "treatment_goals": "Blood sugar control, blood pressure management, weight reduction"
    }

def simulate_llm_care_plan_generation(patient_data):
    name = patient_data["name"]
    diagnosis = patient_data["diagnosis"]
    goals = patient_data["treatment_goals"]
    
    # Simulate LLM generating a natural language care plan
    care_plan = f"""Patient Care Plan for {name} (ID: {patient_data["patient_id"]})

Based on a diagnosis of {diagnosis} and treatment goals of {goals}, here is a personalized care plan:

Medications:
- Metformin 500mg twice daily
- Lisinopril 10mg once daily

Appointments:
- Follow-up with endocrinologist in 2 weeks (virtual)
- Nutrition counseling session next Tuesday at 10:00 AM

Dietary Recommendations:
- Low-carb, high-fiber diet
- Limit processed sugars and unhealthy fats
- Increase intake of fresh vegetables and lean proteins

Exercise Routines:
- Daily 30-minute brisk walk
- Light strength training 3 times a week

Additional Notes:
- Monitor blood sugar levels daily and record readings.
- Keep blood pressure logs.
- Stay hydrated.
"""
    return care_plan

def extract_structured_components(natural_language_plan):
    structured_data = {}
    
    medications_match = re.search(r"Medications:\n([\s\S]*?)(?:\nAppointments:|\nDietary Recommendations:|\nExercise Routines:|\nAdditional Notes:|$)", natural_language_plan)
    if medications_match: 
        structured_data["medications"] = [m.strip() for m in medications_match.group(1).strip().split('\n- ') if m.strip()]

    appointments_match = re.search(r"Appointments:\n([\s\S]*?)(?:\nDietary Recommendations:|\nExercise Routines:|\nAdditional Notes:|$)", natural_language_plan)
    if appointments_match: 
        structured_data["appointments"] = [a.strip() for a in appointments_match.group(1).strip().split('\n- ') if a.strip()]

    dietary_match = re.search(r"Dietary Recommendations:\n([\s\S]*?)(?:\nExercise Routines:|\nAdditional Notes:|$)", natural_language_plan)
    if dietary_match: 
        structured_data["dietary_recommendations"] = [d.strip() for d in dietary_match.group(1).strip().split('\n- ') if d.strip()]

    exercise_match = re.search(r"Exercise Routines:\n([\s\S]*?)(?:\nAdditional Notes:|$)", natural_language_plan)
    if exercise_match: 
        structured_data["exercise_routines"] = [e.strip() for e in exercise_match.group(1).strip().split('\n- ') if e.strip()]

    return structured_data

def format_to_json(patient_id, structured_components):
    final_plan = {
        "patient_id": patient_id,
        "care_plan": structured_components
    }
    return json.dumps(final_plan, indent=4)


if __name__ == "__main__":
    # 1. Patient Data Input Module
    patient_info = get_patient_data()
    print("--- Patient Info ---")
    print(json.dumps(patient_info, indent=4))
    print("\n")

    # 2. LLM Integration Module
    natural_plan = simulate_llm_care_plan_generation(patient_info)
    print("--- Natural Language Care Plan (from LLM) ---")
    print(natural_plan)
    print("\n")

    # 3. Structured Output Extraction Module
    structured_parts = extract_structured_components(natural_plan)
    print("--- Extracted Structured Components ---")
    print(json.dumps(structured_parts, indent=4))
    print("\n")

    # 4. JSON Formatting Module
    json_output = format_to_json(patient_info["patient_id"], structured_parts)
    print("--- Final JSON Care Plan ---")
    print(json_output)
    print("\n")

    # 5. Output & Integration Simulation Module
    output_filename = f"care_plan_{patient_info['patient_id']}.json"
    with open(output_filename, "w") as f:
        f.write(json_output)
    print(f"Simulating integration: Care plan saved to {output_filename}")