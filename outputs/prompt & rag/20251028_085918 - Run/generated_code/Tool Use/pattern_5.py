
import random

# --- Placeholder Specialized Tools ---

def medical_knowledge_database_search(query: str) -> str:
    """Simulates searching a medical knowledge database."""
    print(f"[Tool Call] Medical Knowledge Search for: '{query}'")
    if "fever" in query.lower() and "rash" in query.lower():
        return "Found information on viral exanthems and allergic reactions. Consider measles, rubella, drug eruption."
    elif "chest pain" in query.lower() and "shortness of breath" in query.lower():
        return "Found information on cardiac conditions (MI, angina) and pulmonary conditions (PE, pneumonia)."
    else:
        return "Found general medical information related to the query."

def image_analysis(image_description: str) -> str:
    """Simulates analyzing a medical image."""
    print(f"[Tool Call] Image Analysis for: '{image_description}'")
    if "skin rash" in image_description.lower():
        return "Image analysis suggests maculopapular rash, consistent with viral or allergic etiology."
    elif "x-ray lung" in image_description.lower():
        return "X-ray analysis indicates possible consolidation in lower left lobe, suggestive of pneumonia."
    else:
        return "Image analysis found no specific abnormalities related to the description."

def lab_result_interpretation(lab_results: dict) -> str:
    """Simulates interpreting lab results."""
    print(f"[Tool Call] Lab Result Interpretation for: {lab_results}")
    interpretations = []
    if lab_results.get("WBC") and lab_results["WBC"] > 12.0:
        interpretations.append("Elevated WBC: Suggests infection or inflammation.")
    if lab_results.get("CRP") and lab_results["CRP"] > 5.0:
        interpretations.append("Elevated CRP: Indicates inflammation.")
    if lab_results.get("Hb") and lab_results["Hb"] < 12.0:
        interpretations.append("Low Hemoglobin: Suggests anemia.")

    if not interpretations:
        return "Lab results appear within normal limits or no specific abnormalities detected."
    return " ".join(interpretations)

def drug_interaction_checker(medications: list) -> str:
    """Simulates checking for drug interactions."""
    print(f"[Tool Call] Drug Interaction Check for: {medications}")
    if "warfarin" in [m.lower() for m in medications] and "ibuprofen" in [m.lower() for m in medications]:
        return "Warning: Potential for increased bleeding risk with Warfarin and NSAIDs like Ibuprofen."
    elif len(medications) > 2:
        return "Consider reviewing polypharmacy; no critical interactions found in this simplified check."
    else:
        return "No significant drug interactions detected in this simplified check."

def differential_diagnosis_tool(symptoms: list, findings: list) -> list:
    """Simulates generating a differential diagnosis."""
    print(f"[Tool Call] Differential Diagnosis for Symptoms: {symptoms}, Findings: {findings}")
    possible_diagnoses = []
    if "fever" in [s.lower() for s in symptoms] and "rash" in [s.lower() for s in symptoms]:
        possible_diagnoses.extend(["Measles", "Rubella", "Drug Eruption", "Scarlet Fever"])
    if "chest pain" in [s.lower() for s in symptoms] and "dyspnea" in [s.lower() for s in symptoms]:
        possible_diagnoses.extend(["Myocardial Infarction", "Pulmonary Embolism", "Pneumonia"])
    if "elevated wbc" in [f.lower() for f in findings]:
        possible_diagnoses.append("Bacterial Infection")
    if "consolidation lung" in [f.lower() for f in findings]:
        possible_diagnoses.append("Pneumonia")

    # Remove duplicates and return
    return list(set(possible_diagnoses))

def treatment_recommendation_tool(diagnosis: str, patient_info: dict) -> str:
    """Simulates providing treatment recommendations."""
    print(f"[Tool Call] Treatment Recommendation for: '{diagnosis}' (Patient: {patient_info.get('age', 'N/A')}yo {patient_info.get('gender', 'N/A')})")
    if "pneumonia" in diagnosis.lower():
        return "Recommend antibiotics (e.g., Azithromycin or Amoxicillin), rest, hydration. Consider hospitalization for severe cases."
    elif "measles" in diagnosis.lower():
        return "Recommend supportive care, Vitamin A supplementation. Isolation to prevent spread."
    elif "myocardial infarction" in diagnosis.lower():
        return "Recommend immediate medical attention, aspirin, nitrates, oxygen. Consider reperfusion therapy."
    else:
        return f"General supportive care for {diagnosis}. Consult relevant specialist for detailed plan."


# --- Simulated LLM Orchestrator ---

class MedicalDiagnosisOrchestrator:
    def __init__(self):
        self.context = {}

    def orchestrate_diagnosis(self, patient_case: dict) -> dict:
        """Orchestrates the use of specialized tools for medical diagnosis."""
        print("\n--- Starting Medical Diagnosis Orchestration ---")
        self.context = {"patient_case": patient_case, "findings": [], "diagnoses": []}

        chief_complaint = patient_case.get("chief_complaint", "").lower()
        symptoms = patient_case.get("symptoms", [])
        image_data = patient_case.get("image_data", None)
        lab_data = patient_case.get("lab_data", None)
        medications = patient_case.get("medications", [])

        # Step 1: Initial Medical Knowledge Search based on chief complaint and primary symptoms
        print("\n[Orchestrator] Step 1: Initial Knowledge Search")
        initial_knowledge = medical_knowledge_database_search(f"{chief_complaint} {' '.join(symptoms)}")
        self.context["findings"].append(f"Initial knowledge search: {initial_knowledge}")
        print(f"[Orchestrator] -> {initial_knowledge}")

        # Step 2: Image Analysis if image data is present
        if image_data:
            print("\n[Orchestrator] Step 2: Analyzing Medical Image")
            image_report = image_analysis(image_data)
            self.context["findings"].append(f"Image analysis report: {image_report}")
            print(f"[Orchestrator] -> {image_report}")

        # Step 3: Lab Result Interpretation if lab data is present
        if lab_data:
            print("\n[Orchestrator] Step 3: Interpreting Lab Results")
            lab_interpretation = lab_result_interpretation(lab_data)
            self.context["findings"].append(f"Lab interpretation: {lab_interpretation}")
            print(f"[Orchestrator] -> {lab_interpretation}")

        # Step 4: Drug Interaction Check if medications are listed
        if medications:
            print("\n[Orchestrator] Step 4: Checking Drug Interactions")
            drug_interactions = drug_interaction_checker(medications)
            self.context["findings"].append(f"Drug interaction report: {drug_interactions}")
            print(f"[Orchestrator] -> {drug_interactions}")

        # Step 5: Differential Diagnosis based on all collected information
        print("\n[Orchestrator] Step 5: Generating Differential Diagnosis")
        all_symptoms = symptoms + [chief_complaint] # Add chief complaint to symptoms list for DDx
        # Extract keywords from findings for better DDx
        all_findings_keywords = []
        for f in self.context["findings"]:
            if "elevated wbc" in f.lower(): all_findings_keywords.append("elevated wbc")
            if "consolidation" in f.lower(): all_findings_keywords.append("consolidation lung")
            if "rash" in f.lower(): all_findings_keywords.append("maculopapular rash")

        self.context["diagnoses"] = differential_diagnosis_tool(all_symptoms, all_findings_keywords)
        print(f"[Orchestrator] -> Possible Diagnoses: {self.context['diagnoses']}")

        # Step 6: Treatment Recommendation for the primary or most likely diagnosis
        final_diagnosis = "Unknown Condition" # Default
        if self.context["diagnoses"]:
            final_diagnosis = self.context["diagnoses"][0] # Pick the first one for simplicity
            print("\n[Orchestrator] Step 6: Recommending Treatment")
            treatment_plan = treatment_recommendation_tool(final_diagnosis, patient_case)
            self.context["treatment_plan"] = treatment_plan
            print(f"[Orchestrator] -> Treatment Plan for {final_diagnosis}: {treatment_plan}")
        else:
            self.context["treatment_plan"] = "No specific treatment recommendations generated due to unclear diagnosis."

        print("\n--- Medical Diagnosis Orchestration Complete ---")
        return {
            "patient_summary": patient_case,
            "collected_findings": self.context["findings"],
            "differential_diagnosis": self.context["diagnoses"],
            "recommended_treatment": self.context["treatment_plan"]
        }

# --- Example Usage ---
if __name__ == "__main__":
    orchestrator = MedicalDiagnosisOrchestrator()

    # Case 1: Fever and Rash
    patient_case_1 = {
        "patient_id": "P001",
        "age": 5,
        "gender": "Female",
        "chief_complaint": "Fever and Rash",
        "symptoms": ["High fever", "Red spots all over body", "Cough"],
        "image_data": "Description of a patient's skin with a diffuse maculopapular rash",
        "lab_data": {"WBC": 10.5, "CRP": 3.2, "Hb": 13.0},
        "medications": []
    }
    result_1 = orchestrator.orchestrate_diagnosis(patient_case_1)
    print("\n--- Result for Patient P001 ---")
    print(f"Differential Diagnosis: {result_1['differential_diagnosis']}")
    print(f"Recommended Treatment: {result_1['recommended_treatment']}")

    print("\n" + "="*80 + "\n")

    # Case 2: Chest Pain and Shortness of Breath
    patient_case_2 = {
        "patient_id": "P002",
        "age": 60,
        "gender": "Male",
        "chief_complaint": "Severe Chest Pain",
        "symptoms": ["Shortness of breath", "Sweating", "Left arm pain"],
        "image_data": "Description of a chest X-ray showing no immediate acute findings",
        "lab_data": {"WBC": 9.8, "CRP": 7.5, "Hb": 14.2, "Troponin": 0.8},
        "medications": ["Aspirin", "Lisinopril"]
    }
    result_2 = orchestrator.orchestrate_diagnosis(patient_case_2)
    print("\n--- Result for Patient P002 ---")
    print(f"Differential Diagnosis: {result_2['differential_diagnosis']}")
    print(f"Recommended Treatment: {result_2['recommended_treatment']}")

    print("\n" + "="*80 + "\n")

    # Case 3: General Check-up with high WBC
    patient_case_3 = {
        "patient_id": "P003",
        "age": 35,
        "gender": "Female",
        "chief_complaint": "Follow-up for fatigue",
        "symptoms": ["Tiredness"],
        "image_data": None,
        "lab_data": {"WBC": 15.0, "CRP": 1.0, "Hb": 12.5},
        "medications": ["Multivitamin"]
    }
    result_3 = orchestrator.orchestrate_diagnosis(patient_case_3)
    print("\n--- Result for Patient P003 ---")
    print(f"Differential Diagnosis: {result_3['differential_diagnosis']}")
    print(f"Recommended Treatment: {result_3['recommended_treatment']}")
