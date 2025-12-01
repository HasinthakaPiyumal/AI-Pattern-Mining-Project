import json
import re

def simulate_llm_output(doctor_notes: str) -> str:
    # In a real application, this would be an actual LLM API call.
    # For this simulation, we'll return a string that's designed to be parsed.
    # The LLM is instructed to output in a semi-structured natural language format.
    
    # Example of what an LLM might generate based on doctor's notes:
    if "headache" in doctor_notes.lower() and "painkiller" in doctor_notes.lower():
        return (
            "Patient requires medication: Ibuprofen, Dosage: 400mg, Frequency: every 6 hours as needed. "
            "Dietary: Avoid spicy foods. Follow-up: Dr. Smith in 2 weeks. "
            "Exercise: Light walking daily for 30 minutes."
        )
    elif "diabetes" in doctor_notes.lower() and "insulin" in doctor_notes.lower():
        return (
            "Medication: Insulin Glargine, Dosage: 10 units, Frequency: once daily. "
            "Dietary: Low sugar, balanced meals. Follow-up: Endocrinology clinic in 1 month. "
            "Exercise: Moderate cardio 3 times a week, 45 minutes each."
        )
    else:
        return (
            "Medication: No specific new medication. Dietary: Maintain healthy diet. "
            "Follow-up: PCP in 1 month for routine check-up. Exercise: Continue regular activity."
        )

def parse_llm_output_to_json(llm_output: str) -> dict:
    care_plan = {
        "medication": {},
        "dietary_restrictions": "",
        "follow_up": "",
        "exercise_recommendations": ""
    }

    # Regex for medication
    medication_match = re.search(
        r"Medication: (.*?)(?:, Dosage: (.*?))?(?:, Frequency: (.*?))?(?:\.|\n|$)",
        llm_output
    )
    if medication_match:
        med_name = medication_match.group(1).strip()
        dosage = medication_match.group(2).strip() if medication_match.group(2) else "N/A"
        frequency = medication_match.group(3).strip() if medication_match.group(3) else "N/A"
        care_plan["medication"] = {
            "name": med_name,
            "dosage": dosage,
            "frequency": frequency
        }

    # Regex for dietary restrictions
    dietary_match = re.search(r"Dietary: (.*?)(?:\.|\n|$)", llm_output)
    if dietary_match:
        care_plan["dietary_restrictions"] = dietary_match.group(1).strip()

    # Regex for follow-up
    follow_up_match = re.search(r"Follow-up: (.*?)(?:\.|\n|$)", llm_output)
    if follow_up_match:
        care_plan["follow_up"] = follow_up_match.group(1).strip()

    # Regex for exercise recommendations
    exercise_match = re.search(r"Exercise: (.*?)(?:\.|\n|$)", llm_output)
    if exercise_match:
        care_plan["exercise_recommendations"] = exercise_match.group(1).strip()

    return care_plan

if __name__ == "__main__":
    # Example 1: Doctor's notes for headache
    doctor_notes_1 = "Patient has a severe headache, prescribe a painkiller. Needs to avoid spicy food. See Dr. Smith in two weeks. Suggest light daily walks."
    print(f"--- Doctor's Notes 1 ---")
    print(doctor_notes_1)
    simulated_output_1 = simulate_llm_output(doctor_notes_1)
    print(f"\n--- Simulated LLM Output 1 ---")
    print(simulated_output_1)
    structured_plan_1 = parse_llm_output_to_json(simulated_output_1)
    print(f"\n--- Structured Care Plan 1 (JSON) ---")
    print(json.dumps(structured_plan_1, indent=4))
    print("\n" + "="*50 + "\n")

    # Example 2: Doctor's notes for diabetes management
    doctor_notes_2 = "Patient with diabetes, requiring insulin. Advise low sugar diet. Follow up with endocrinology in one month. Recommend moderate cardio."
    print(f"--- Doctor's Notes 2 ---")
    print(doctor_notes_2)
    simulated_output_2 = simulate_llm_output(doctor_notes_2)
    print(f"\n--- Simulated LLM Output 2 ---")
    print(simulated_output_2)
    structured_plan_2 = parse_llm_output_to_json(simulated_output_2)
    print(f"\n--- Structured Care Plan 2 (JSON) ---")
    print(json.dumps(structured_plan_2, indent=4))
    print("\n" + "="*50 + "\n")

    # Example 3: General check-up notes
    doctor_notes_3 = "Patient in for a routine check-up, no major issues. Just advise healthy eating and regular activity. See PCP in a month."
    print(f"--- Doctor's Notes 3 ---")
    print(doctor_notes_3)
    simulated_output_3 = simulate_llm_output(doctor_notes_3)
    print(f"\n--- Simulated LLM Output 3 ---")
    print(simulated_output_3)
    structured_plan_3 = parse_llm_output_to_json(simulated_output_3)
    print(f"\n--- Structured Care Plan 3 (JSON) ---")
    print(json.dumps(structured_plan_3, indent=4))
    print("\n" + "="*50 + "\n")