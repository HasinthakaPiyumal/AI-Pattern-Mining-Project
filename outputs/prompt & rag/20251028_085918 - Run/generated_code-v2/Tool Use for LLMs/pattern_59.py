class MedicalDiagnosisAssistant:
    def __init__(self):
        self.knowledge_base = {
            "flu": {
                "symptoms": ["fever", "cough", "fatigue", "body aches"],
                "treatment": "Antivirals, rest, fluids",
                "typical_dosage_mg_per_kg": 10 # Example for a hypothetical antiviral
            },
            "common_cold": {
                "symptoms": ["sore throat", "runny nose", "sneezing"],
                "treatment": "Rest, fluids, symptom relief",
                "typical_dosage_mg_per_kg": 0 # No specific drug for dosage example
            },
            "diabetes": {
                "symptoms": ["frequent urination", "increased thirst", "fatigue", "blurred vision"],
                "lab_markers": {"glucose": {"high": "indicative of diabetes"}},
                "treatment": "Insulin, diet, exercise",
                "typical_dosage_units_per_day": 50 # Example for a hypothetical insulin regimen
            }
        }

    def _interpret_symptoms_nl(self, symptoms):
        interpretation = f"Based on the reported symptoms: {', '.join(symptoms)}. "
        possible_conditions = []
        for condition, data in self.knowledge_base.items():
            if all(s in symptoms for s in data["symptoms"]):
                possible_conditions.append(condition.replace('_', ' ').title())
        
        if possible_conditions:
            interpretation += f"This points towards possible conditions such as: {', '.join(possible_conditions)}."
        else:
            interpretation += "More information is needed to identify specific conditions."
        return interpretation

    def _analyze_lab_results_symbolic(self, lab_results):
        analysis = []
        for marker, value in lab_results.items():
            if marker == "glucose":
                if value > 120: # mg/dL, example threshold for high glucose
                    analysis.append(f"Glucose level ({value} mg/dL) is high. This is a significant marker.")
                    return {"glucose": "high"}
                elif value < 70:
                    analysis.append(f"Glucose level ({value} mg/dL) is low. Requires immediate attention.")
                    return {"glucose": "low"}
                else:
                    analysis.append(f"Glucose level ({value} mg/dL) is within normal limits.")
            # Add more lab markers and their symbolic analysis here
        return {"glucose": "normal"} # Default if no specific high/low found or other markers

    def _run_diagnostic_algorithm_symbolic(self, symptoms, lab_analysis):
        diagnosis = "Undetermined"
        reasoning_steps = []

        # Rule 1: Flu based on symptoms
        flu_symptoms = self.knowledge_base["flu"]["symptoms"]
        if all(s in symptoms for s in flu_symptoms):
            diagnosis = "Flu"
            reasoning_steps.append("All key flu symptoms are present.")

        # Rule 2: Diabetes based on symptoms and high glucose
        diabetes_symptoms = self.knowledge_base["diabetes"]["symptoms"]
        if all(s in symptoms for s in diabetes_symptoms) and lab_analysis.get("glucose") == "high":
            if diagnosis == "Undetermined": # Prioritize more specific diagnosis if applicable
                diagnosis = "Diabetes (Type 2 suspected)"
                reasoning_steps.append("Key diabetes symptoms combined with high glucose levels strongly suggest diabetes.")
            else:
                reasoning_steps.append("Note: Diabetes symptoms and high glucose also present, but another primary diagnosis identified.")

        if diagnosis == "Undetermined" and any(s in symptoms for s in self.knowledge_base["common_cold"]["symptoms"]):
            diagnosis = "Common Cold"
            reasoning_steps.append("Symptoms align with common cold, no other specific conditions identified.")

        return diagnosis, reasoning_steps

    def _calculate_treatment_symbolic(self, diagnosis, patient_weight_kg=None):
        treatment_info = {"recommendation": "No specific treatment identified.", "dosage": "N/A"}
        if diagnosis in self.knowledge_base:
            disease_data = self.knowledge_base[diagnosis]
            treatment_info["recommendation"] = disease_data["treatment"]

            if diagnosis == "flu" and patient_weight_kg is not None:
                dosage_per_kg = disease_data.get("typical_dosage_mg_per_kg", 0)
                if dosage_per_kg > 0:
                    total_dosage = dosage_per_kg * patient_weight_kg
                    treatment_info["dosage"] = f"{total_dosage:.2f} mg daily (based on {patient_weight_kg} kg weight)"
            elif diagnosis == "diabetes":
                 # For diabetes, could be insulin units, often not simply weight-based for initial demo
                treatment_info["dosage"] = f"{disease_data.get('typical_dosage_units_per_day', 'Varies')} units daily (requires physician's adjustment)"
        return treatment_info

    def generate_faithful_cot(self, patient_data):
        symptoms = patient_data.get("symptoms", [])
        lab_results = patient_data.get("lab_results", {})
        patient_weight_kg = patient_data.get("weight_kg")

        cot_steps = []
        
        # Step 1: Natural Language Interpretation of Symptoms
        cot_steps.append("**Natural Language Reasoning - Symptom Interpretation:**")
        cot_steps.append(self._interpret_symptoms_nl(symptoms))

        # Step 2: Symbolic Language Analysis of Lab Results
        cot_steps.append("\n**Symbolic Reasoning - Lab Results Analysis:**")
        symbolic_lab_analysis = self._analyze_lab_results_symbolic(lab_results)
        cot_steps.append(f"Processed Lab Results: {symbolic_lab_analysis}")
        for marker, status in symbolic_lab_analysis.items():
            if status == "high":
                cot_steps.append(f"    - The {marker} level is flagged as HIGH.")
            elif status == "low":
                cot_steps.append(f"    - The {marker} level is flagged as LOW.")
            else:
                cot_steps.append(f"    - The {marker} level is within normal range.")


        # Step 3: Combined Diagnostic Algorithm (Symbolic & NL Explanation)
        cot_steps.append("\n**Combined Reasoning - Diagnostic Algorithm Application:**")
        diagnosis, diagnostic_reasoning = self._run_diagnostic_algorithm_symbolic(symptoms, symbolic_lab_analysis)
        cot_steps.append(f"Applying a rule-based diagnostic algorithm...")
        for step in diagnostic_reasoning:
            cot_steps.append(f"    - Rule applied: {step}")
        cot_steps.append(f"**Final Proposed Diagnosis:** {diagnosis}")

        # Step 4: Treatment Recommendation & Symbolic Dosage Calculation
        cot_steps.append("\n**Symbolic Reasoning - Treatment & Dosage Calculation:**")
        treatment_details = self._calculate_treatment_symbolic(diagnosis, patient_weight_kg)
        cot_steps.append(f"Recommended Treatment for {diagnosis}: {treatment_details['recommendation']}")
        if treatment_details["dosage"] != "N/A":
            cot_steps.append(f"Calculated Dosage: {treatment_details['dosage']} (Symbolic Computation)")
        else:
            cot_steps.append(f"Dosage: {treatment_details['dosage']} (No specific calculation for this diagnosis/drug in current model)")

        return "\n".join(cot_steps)

# --- Example Usage ---
if __name__ == "__main__":
    assistant = MedicalDiagnosisAssistant()

    print("--- Patient Case 1: Flu-like Symptoms ---")
    patient_data_1 = {
        "symptoms": ["fever", "cough", "fatigue", "body aches", "sore throat"],
        "lab_results": {"glucose": 95},
        "weight_kg": 70
    }
    cot_1 = assistant.generate_faithful_cot(patient_data_1)
    print(cot_1)
    print("\n" + "="*80 + "\n")

    print("--- Patient Case 2: Potential Diabetes ---")
    patient_data_2 = {
        "symptoms": ["frequent urination", "increased thirst", "fatigue", "blurred vision"],
        "lab_results": {"glucose": 180},
        "weight_kg": 85
    }
    cot_2 = assistant.generate_faithful_cot(patient_data_2)
    print(cot_2)
    print("\n" + "="*80 + "\n")

    print("--- Patient Case 3: Common Cold ---")
    patient_data_3 = {
        "symptoms": ["runny nose", "sneezing", "sore throat"],
        "lab_results": {"glucose": 100},
        "weight_kg": 60
    }
    cot_3 = assistant.generate_faithful_cot(patient_data_3)
    print(cot_3)
    print("\n" + "="*80 + "\n")

    print("--- Patient Case 4: Undetermined (More info needed) ---")
    patient_data_4 = {
        "symptoms": ["headache", "nausea"],
        "lab_results": {"glucose": 88},
        "weight_kg": 65
    }
    cot_4 = assistant.generate_faithful_cot(patient_data_4)
    print(cot_4)
    print("\n" + "="*80 + "\n")