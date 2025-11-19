def search_medical_database(query: str) -> str:
    query = query.lower()
    if "fever" in query and "cough" in query:
        return "Common causes of fever and cough include viral infections (e.g., flu, common cold), bronchitis, and pneumonia."
    elif "headache" in query:
        return "Headaches can be caused by various factors including tension, migraines, sinus issues, or rarely, more serious conditions."
    elif "chest pain" in query:
        return "Chest pain requires urgent evaluation, potential causes include cardiac issues (angina, heart attack), lung problems (pneumonia, pleurisy), or musculoskeletal pain."
    else:
        return "No specific conditions found for the given query, general health advice is recommended."

def analyze_lab_results(lab_data: dict) -> str:
    interpretations = []
    if "white_blood_cells" in lab_data:
        wbc = lab_data["white_blood_cells"]
        if wbc > 10.0:
            interpretations.append(f"Elevated white blood cells ({wbc}): Suggests an infection or inflammation.")
        elif wbc < 4.0:
            interpretations.append(f"Low white blood cells ({wbc}): Could indicate a weakened immune system, certain medications, or bone marrow issues.")
        else:
            interpretations.append(f"White blood cells ({wbc}): Within normal range.")
    
    if "c_reactive_protein" in lab_data:
        crp = lab_data["c_reactive_protein"]
        if crp > 5.0:
            interpretations.append(f"Elevated C-reactive protein ({crp}): Indicates significant inflammation or infection.")
        else:
            interpretations.append(f"C-reactive protein ({crp}): Within normal range.")

    if not interpretations:
        return "No specific interpretations available for the provided lab results."
    return "; ".join(interpretations)

def simulate_llm_response(patient_data: dict, available_tools: dict) -> str:
    symptoms = patient_data.get("symptoms", "")
    lab_results = patient_data.get("lab_results", {})

    reasoning_steps = []
    diagnosis_options = []
    overall_confidence = 0

    db_search_query = f"Information about symptoms: {symptoms}"
    reasoning_steps.append(f"Consulting medical database for symptoms: '{symptoms}'.")
    db_response = available_tools["search_medical_database"](db_search_query)
    reasoning_steps.append(f"Database response for symptoms: {db_response}")

    lab_analysis_summary = "No lab results provided or analyzed."
    if lab_results:
        reasoning_steps.append(f"Analyzing lab results: {lab_results}.")
        lab_analysis_summary = available_tools["analyze_lab_results"](lab_results)
        reasoning_steps.append(f"Lab analysis summary: {lab_analysis_summary}")

    if "fever" in symptoms.lower() and "cough" in symptoms.lower() and "elevated white blood cells" in lab_analysis_summary.lower():
        diagnosis_options.append({
            "condition": "Bacterial Infection",
            "confidence": 85,
            "reasoning": "Consistent with fever, cough, and signs of infection from lab results.",
            "details": "Further tests like culture may be needed to confirm specific bacteria."
        })
        overall_confidence += 85

    if "headache" in symptoms.lower() and "fatigue" in symptoms.lower() and "normal lab values" in lab_analysis_summary.lower():
        diagnosis_options.append({
            "condition": "Viral Syndrome",
            "confidence": 70,
            "reasoning": "Common viral symptoms with no specific bacterial indicators in labs.",
            "details": "Rest and symptomatic treatment are typically recommended."
        })
        overall_confidence += 70

    if "chest pain" in symptoms.lower() and "shortness of breath" in symptoms.lower():
        diagnosis_options.append({
            "condition": "Cardiac Concern (e.g., Angina)",
            "confidence": 90,
            "reasoning": "Classic symptoms requiring urgent medical evaluation.",
            "details": "Immediate ECG and cardiac enzyme tests are crucial."
        })
        overall_confidence += 90
    
    if not diagnosis_options:
        diagnosis_options.append({
            "condition": "Undetermined/Non-specific",
            "confidence": 40,
            "reasoning": "Insufficient specific symptoms or lab findings for a clear diagnosis.",
            "details": "Further observation or tests may be required."
        })
        overall_confidence = 40

    overall_confidence = int(overall_confidence / len(diagnosis_options)) if diagnosis_options else 0

    output = ["Diagnosis:"]
    for i, diag in enumerate(diagnosis_options):
        output.append(f" {i+1}. {diag['condition']} (Confidence: {diag['confidence']}%) - {diag['reasoning']}")
        output.append(f"    - Details: {diag['details']}")
    
    output.append("\nReasoning Process:")
    output.extend([f"- {step}" for step in reasoning_steps])

    output.append(f"\nOverall Confidence Score: {overall_confidence}%")

    return "\n".join(output)

def evaluate_diagnosis(actual_diagnosis: dict, predicted_output: str) -> dict:
    evaluation_results = {
        "match": False,
        "confidence_alignment": "N/A",
        "reasoning_quality_score": 0.0
    }

    predicted_condition = ""
    for line in predicted_output.split('\n'):
        if line.strip().startswith('1.') and '(Confidence:' in line:
            predicted_condition = line.split('(Confidence:')[0].replace('1.', '').strip()
            break

    if actual_diagnosis and predicted_condition and actual_diagnosis['condition'].lower() in predicted_condition.lower():
        evaluation_results['match'] = True
        predicted_confidence_str = "0"
        try:
            start_idx = predicted_output.find("Overall Confidence Score:")
            if start_idx != -1:
                end_idx = predicted_output.find("%", start_idx)
                if end_idx != -1:
                    predicted_confidence_str = predicted_output[start_idx + len("Overall Confidence Score:"):end_idx].strip()
            predicted_confidence = int(predicted_confidence_str)
            
            if predicted_confidence >= 70:
                evaluation_results['confidence_alignment'] = "High (matched)"
            else:
                evaluation_results['confidence_alignment'] = "Moderate (matched)"
        except ValueError:
            pass
        
        evaluation_results['reasoning_quality_score'] = 0.8
    else:
        evaluation_results['match'] = False
        evaluation_results['confidence_alignment'] = "Low (no match)"
        evaluation_results['reasoning_quality_score'] = 0.3

    print("\n--- Evaluation Results (Simulated) ---")
    print(f"Actual Diagnosis: {actual_diagnosis.get('condition', 'N/A')}")
    print(f"Predicted Match: {evaluation_results['match']}")
    print(f"Confidence Alignment: {evaluation_results['confidence_alignment']}")
    print(f"Reasoning Quality Score: {evaluation_results['reasoning_quality_score']:.2f}/1.0")
    
    return evaluation_results

def run_diagnostic_assistant():
    print("\n--- AI-powered Medical Diagnostic Assistant ---")
    print("Please provide patient information.")

    symptoms_input = input("Enter patient symptoms (e.g., 'fever, cough, fatigue'): ")
    
    lab_results_input = input("Enter lab results as comma-separated key-value pairs (e.g., 'white_blood_cells:12.5,c_reactive_protein:8.2') or leave empty: ")
    lab_results_dict = {}
    if lab_results_input:
        try:
            pairs = lab_results_input.split(',')
            for pair in pairs:
                key_val = pair.split(':')
                if len(key_val) == 2:
                    lab_results_dict[key_val[0].strip()] = float(key_val[1].strip())
        except ValueError:
            print("Warning: Could not parse lab results. Please use 'key:value' format.")
            lab_results_dict = {}

    patient_data = {
        "symptoms": symptoms_input,
        "lab_results": lab_results_dict
    }

    available_tools = {
        "search_medical_database": search_medical_database,
        "analyze_lab_results": analyze_lab_results
    }

    print("\nConsulting the AI assistant...")
    diagnostic_output = simulate_llm_response(patient_data, available_tools)
    
    print("\n--- Diagnostic Report ---")
    print(diagnostic_output)

    print("\nDo you want to elaborate on a specific diagnosis or aspect? (yes/no)")
    if input().lower() == 'yes':
        print("Please specify what you'd like to know more about (e.g., 'Bacterial Infection details' or 'Reasoning process'):")
        user_query = input()
        if "bacterial infection" in user_query.lower() and "details" in diagnostic_output.lower():
            start = diagnostic_output.lower().find("bacterial infection")
            end = diagnostic_output.lower().find("\nreasoning process")
            if start != -1 and end != -1 and start < end:
                print("\n--- Elaboration ---")
                print(diagnostic_output[start:end].strip())
            else:
                print("Could not find specific details for 'Bacterial Infection'.")
        elif "reasoning process" in user_query.lower():
            start = diagnostic_output.lower().find("reasoning process")
            if start != -1:
                print("\n--- Elaboration ---")
                print(diagnostic_output[start:].strip())
            else:
                print("Could not find the reasoning process.")
        else:
            print("No further specific elaboration available for your query at this time. Please refer to the full report.")

if __name__ == "__main__":
    run_diagnostic_assistant()