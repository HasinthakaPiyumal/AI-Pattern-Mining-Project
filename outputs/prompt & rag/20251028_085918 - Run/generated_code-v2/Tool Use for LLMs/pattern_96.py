def get_patient_data():
    # Simulate patient input
    symptoms = ["fever", "cough", "fatigue"]
    lab_results = {"WBC_count": 13.5} # Elevated WBC count
    # symptoms = ["fever", "sore throat", "body aches"]
    # lab_results = {"WBC_count": 7.2} # Normal WBC count
    return symptoms, lab_results

def generate_nl_explanation(symptoms, lab_results):
    explanation = "Based on the provided symptoms and lab results, an initial assessment suggests exploring the possibility of an infection. "
    if "fever" in symptoms and lab_results.get("WBC_count", 0) > 11.0:
        explanation += "The presence of fever combined with an elevated White Blood Cell count often points towards a bacterial infection, which typically requires further investigation and potentially antibiotic treatment. However, other factors also need to be considered."
    elif "fever" in symptoms and lab_results.get("WBC_count", 0) <= 11.0 and lab_results.get("WBC_count", 0) >= 4.5:
        explanation += "Fever with a normal or slightly low White Blood Cell count could indicate a viral infection, such as the flu, which usually resolves with supportive care."
    else:
        explanation += "Further specific symptoms and detailed lab markers would be needed for a more focused natural language reasoning."
    return explanation

def perform_symbolic_diagnosis(symptoms, lab_results):
    symbolic_reasoning = "\n--- Symbolic Reasoning (Rule-Based System) ---\n"
    diagnosis = "Undetermined"
    wbc_count = lab_results.get("WBC_count", None)

    symbolic_reasoning += f"Patient Symptoms: {', '.join(symptoms)}\n"
    symbolic_reasoning += f"Lab Result - WBC Count: {wbc_count} x 10^9/L\n"

    # Define normal WBC range
    NORMAL_WBC_MIN = 4.5
    NORMAL_WBC_MAX = 11.0

    symbolic_reasoning += f"Diagnostic Rule: If WBC_count > {NORMAL_WBC_MAX} AND 'fever' in symptoms, then consider Bacterial Infection.\n"
    symbolic_reasoning += f"Diagnostic Rule: If WBC_count >= {NORMAL_WBC_MIN} AND WBC_count <= {NORMAL_WBC_MAX} AND 'fever' in symptoms, then consider Viral Infection.\n"

    if wbc_count is not None:
        if wbc_count > NORMAL_WBC_MAX and "fever" in symptoms:
            diagnosis = "Bacterial Infection (Likely)"
            symbolic_reasoning += f"Decision: WBC count ({wbc_count}) is elevated ({wbc_count} > {NORMAL_WBC_MAX}) and 'fever' is present. Conclusion: {diagnosis}\n"
        elif wbc_count >= NORMAL_WBC_MIN and wbc_count <= NORMAL_WBC_MAX and "fever" in symptoms:
            diagnosis = "Viral Infection (Likely)"
            symbolic_reasoning += f"Decision: WBC count ({wbc_count}) is within normal range ({NORMAL_WBC_MIN}-{NORMAL_WBC_MAX}) and 'fever' is present. Conclusion: {diagnosis}\n"
        else:
            diagnosis = "Inconclusive based on provided rules"
            symbolic_reasoning += f"Decision: No specific rule matched for provided WBC count and symptoms. Conclusion: {diagnosis}\n"
    else:
        symbolic_reasoning += "Decision: WBC count not available for symbolic diagnosis.\n"
        diagnosis = "Inconclusive (Missing WBC data)"

    return diagnosis, symbolic_reasoning

def diagnose_patient():
    print("\n--- Medical Diagnosis Assistant ---")
    symptoms, lab_results = get_patient_data()

    print("\n--- Patient Data ---")
    print(f"Symptoms: {', '.join(symptoms)}")
    print(f"Lab Results: {lab_results}")

    # Natural Language Reasoning
    nl_explanation = generate_nl_explanation(symptoms, lab_results)
    print("\n--- Natural Language Reasoning ---")
    print(nl_explanation)

    # Symbolic Language Reasoning
    diagnosis, symbolic_reasoning = perform_symbolic_diagnosis(symptoms, lab_results)
    print(symbolic_reasoning)

    print("\n--- Final Diagnosis ---")
    print(f"Overall Diagnosis: {diagnosis}")

if __name__ == "__main__":
    diagnose_patient()