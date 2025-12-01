import time

class LLM_Simulator:
    def __init__(self):
        self.knowledge_base = {
            "fever, cough, fatigue": {
                "influenza": "Common viral infection. Look for muscle aches, sore throat.",
                "common cold": "Milder than flu. Usually no fever or mild fever.",
                "pneumonia": "Bacterial or viral lung infection. May have shortness of breath, chest pain."
            },
            "abdominal pain, nausea, vomiting": {
                "appendicitis": "Acute inflammation of appendix. Pain often starts around navel and moves to lower right abdomen.",
                "gastroenteritis": "Stomach flu. Often accompanied by diarrhea.",
                "kidney stones": "Severe pain, often in back or side, radiating to lower abdomen/groin."
            },
            "shortness of breath, chest pain, palpitations": {
                "anxiety attack": "Often sudden onset, can mimic heart attack symptoms, hyperventilation.",
                "heart attack": "Crushing chest pain, often radiating to arm/jaw, sweating, nausea.",
                "asthma exacerbation": "Wheezing, tightness in chest, difficulty breathing, history of asthma."
            }
        }

    def _simulate_llm_response(self, prompt, reasoning_steps, final_output):
        print(f"\nLLM Thinking Process (Simulated CoT for: {prompt[:50]}...)")
        for step in reasoning_steps:
            print(f"  - {step}")
            time.sleep(0.5) # Simulate processing time
        print(f"LLM Output: {final_output}")
        return final_output

    def generate_diagnosis(self, symptoms):
        prompt = f"Given the symptoms: {symptoms}, propose differential diagnoses."
        reasoning_steps = [
            "Analyzing reported symptoms and cross-referencing with known medical conditions.",
            "Identifying common and less common conditions associated with the symptom set.",
            "Prioritizing potential diagnoses based on typical presentations."
        ]
        
        potential_diagnoses = {}
        for symptom_set, diagnoses in self.knowledge_base.items():
            # Check if all symptoms in the knowledge base set are present in the input symptoms
            if all(s.strip() in symptoms.lower() for s in symptom_set.split(",")):
                potential_diagnoses.update(diagnoses)
        
        if not potential_diagnoses:
            final_output = "Could not find immediate differential diagnoses based on the provided symptoms in the simulated knowledge base."
        else:
            final_output = "Differential Diagnoses:\n" + "\n".join([f"  - {d}: {desc}" for d, desc in potential_diagnoses.items()])

        return self._simulate_llm_response(prompt, reasoning_steps, final_output)

    def recommend_investigations(self, current_diagnoses):
        prompt = f"Based on the current differential diagnoses: {current_diagnoses}, recommend further investigations."
        reasoning_steps = [
            "Considering the most critical or ambiguous diagnoses from the current list.",
            "Selecting investigations that can differentiate between closely related conditions.",
            "Prioritizing non-invasive and cost-effective tests where possible."
        ]
        
        recommendations = []
        for diagnosis in current_diagnoses.keys():
            if "influenza" in diagnosis.lower() or "pneumonia" in diagnosis.lower():
                if "Chest X-ray" not in recommendations: recommendations.append("Chest X-ray")
                if "Influenza PCR test" not in recommendations: recommendations.append("Influenza PCR test")
            if "appendicitis" in diagnosis.lower():
                if "Abdominal Ultrasound or CT scan" not in recommendations: recommendations.append("Abdominal Ultrasound or CT scan")
                if "Complete Blood Count (CBC)" not in recommendations: recommendations.append("Complete Blood Count (CBC)")
            if "kidney stones" in diagnosis.lower():
                if "Urinalysis" not in recommendations: recommendations.append("Urinalysis")
                if "CT KUB (Kidneys, Ureters, Bladder)" not in recommendations: recommendations.append("CT KUB (Kidneys, Ureters, Bladder)")
            if "heart attack" in diagnosis.lower():
                if "ECG (Electrocardiogram)" not in recommendations: recommendations.append("ECG (Electrocardiogram)")
                if "Cardiac Enzyme test (e.g., Troponin)" not in recommendations: recommendations.append("Cardiac Enzyme test (e.g., Troponin)")

        if not recommendations:
            final_output = "No specific investigations recommended based on the current simulated diagnoses."
        else:
            final_output = "Recommended Investigations:\n" + "\n".join([f"  - {rec}" for rec in recommendations])

        return self._simulate_llm_response(prompt, reasoning_steps, final_output)

    def evaluate_results(self, initial_diagnoses, test_results):
        prompt = f"Evaluate the following test results: {test_results}, in light of initial diagnoses: {initial_diagnoses}."
        reasoning_steps = [
            "Correlating each test result with the diagnostic criteria of the differential diagnoses.",
            "Eliminating diagnoses that are inconsistent with new evidence.",
            "Strengthening the likelihood of diagnoses supported by the results.",
            "Considering new potential diagnoses if results point to unexpected findings."
        ]

        refined_diagnoses = initial_diagnoses.copy()
        evaluation_output = []
        test_results_lower = test_results.lower()

        # Evaluate for Influenza/Common Cold/Pneumonia
        if "positive influenza pcr" in test_results_lower:
            if "influenza" in refined_diagnoses:
                refined_diagnoses["influenza"] = "Highly likely due to positive PCR."
                evaluation_output.append("Influenza likelihood significantly increased.")
            if "common cold" in refined_diagnoses: del refined_diagnoses["common cold"]
        elif "normal chest x-ray" in test_results_lower:
            if "pneumonia" in refined_diagnoses: 
                del refined_diagnoses["pneumonia"]
                evaluation_output.append("Pneumonia less likely given normal chest X-ray.")

        # Evaluate for Appendicitis
        if "inflamed appendix" in test_results_lower and "elevated white blood cell count" in test_results_lower:
             if "appendicitis" in refined_diagnoses:
                 refined_diagnoses["appendicitis"] = "Confirmed by imaging and lab results."
                 evaluation_output.append("Appendicitis confirmed.")

        # Evaluate for Kidney Stones
        if "kidney stones seen on ct" in test_results_lower:
            if "kidney stones" in refined_diagnoses:
                refined_diagnoses["kidney stones"] = "Confirmed by imaging."
                evaluation_output.append("Kidney stones confirmed.")
        elif "normal urinalysis" in test_results_lower and "kidney stones" in refined_diagnoses:
             evaluation_output.append("Normal urinalysis may reduce suspicion of kidney stones, but imaging is more definitive.")

        # Evaluate for Heart Attack
        if "ecg shows st elevation" in test_results_lower and "troponin levels are significantly elevated" in test_results_lower:
            if "heart attack" in refined_diagnoses:
                refined_diagnoses["heart attack"] = "Strong evidence for myocardial infarction."
                evaluation_output.append("Myocardial infarction strongly indicated.")
        elif "normal ecg" in test_results_lower and "normal troponin" in test_results_lower:
            if "heart attack" in refined_diagnoses: del refined_diagnoses["heart attack"]
            if "anxiety attack" in refined_diagnoses: refined_diagnoses["anxiety attack"] = "More likely given exclusion of cardiac event."
            evaluation_output.append("Cardiac event unlikely given normal ECG and troponin.")

        if not evaluation_output:
            evaluation_output.append("Test results did not significantly alter the differential diagnoses in the simulated context.")

        final_output = "Evaluated Results:\n" + "\n".join(evaluation_output)
        final_output += "\nRefined Differential Diagnoses:\n" + "\n".join([f"  - {d}: {desc}" for d, desc in refined_diagnoses.items()])
        
        return self._simulate_llm_response(prompt, reasoning_steps, final_output), refined_diagnoses

    def suggest_treatment(self, final_diagnosis_summary):
        prompt = f"Given the final diagnosis summary: {final_diagnosis_summary}, suggest a treatment plan."
        reasoning_steps = [
            "Consulting treatment guidelines for the confirmed or most likely diagnosis.",
            "Considering patient-specific factors (e.g., allergies, comorbidities) if provided (simulated here).",
            "Proposing evidence-based interventions and follow-up."
        ]

        treatment_plan = []
        final_diagnosis_lower = final_diagnosis_summary.lower()

        if "influenza" in final_diagnosis_lower:
            treatment_plan.append("Antiviral medication (e.g., Oseltamivir)")
            treatment_plan.append("Rest and hydration")
            treatment_plan.append("Symptomatic relief (e.g., antipyretics for fever)")
        elif "appendicitis" in final_diagnosis_lower:
            treatment_plan.append("Surgical appendectomy (urgent)")
            treatment_plan.append("Intravenous antibiotics")
        elif "kidney stones" in final_diagnosis_lower:
            treatment_plan.append("Pain management (NSAIDs or opioids)")
            treatment_plan.append("Hydration")
            treatment_plan.append("Medical expulsive therapy (e.g., alpha-blockers) for smaller stones")
            treatment_plan.append("Lithotripsy or surgery for larger stones if needed")
        elif "heart attack" in final_diagnosis_lower or "myocardial infarction" in final_diagnosis_lower:
            treatment_plan.append("Immediate hospitalization and emergency medical treatment (e.g., aspirin, nitroglycerin, oxygen)")
            treatment_plan.append("Coronary angioplasty and stenting or bypass surgery")
            treatment_plan.append("Long-term medication (e.g., beta-blockers, statins, ACE inhibitors)")
        else:
            treatment_plan.append("Further specialist consultation may be required for complex or undiagnosed cases.")

        final_output = "Treatment Plan:\n" + "\n".join([f"  - {plan}" for plan in treatment_plan])
        return self._simulate_llm_response(prompt, reasoning_steps, final_output)

class MedicalDiagnosisAssistant:
    def __init__(self, knowledge_base=None):
        self.llm_simulator = LLM_Simulator()
        self.current_symptoms = ""
        self.current_differential_diagnoses = {}
        print("Medical Diagnosis Assistant Initialized. Ready to assist.")

    def analyze_symptoms(self, symptoms):
        self.current_symptoms = symptoms
        print(f"\n--- Step 1: Analyzing Initial Symptoms ---")
        diagnosis_output = self.llm_simulator.generate_diagnosis(symptoms)
        
        # Parse differential diagnoses from LLM output (simplified for simulation)
        self.current_differential_diagnoses = {}
        if "Differential Diagnoses:" in diagnosis_output:
            lines = diagnosis_output.split("\n")
            for line in lines:
                if line.strip().startswith("-") and ":" in line:
                    try:
                        diag_name = line.split("-", 1)[1].strip().split(":")[0].strip()
                        diag_desc = line.split(":", 1)[1].strip()
                        self.current_differential_diagnoses[diag_name] = diag_desc
                    except IndexError: # Handle cases where parsing might fail
                        pass
        return diagnosis_output

    def recommend_further_investigations(self):
        print(f"\n--- Step 2: Recommending Further Investigations ---")
        if not self.current_differential_diagnoses:
            print("No differential diagnoses to base investigations on. Please analyze symptoms first.")
            return ""

        return self.llm_simulator.recommend_investigations(self.current_differential_diagnoses)

    def evaluate_test_results(self, test_results):
        print(f"\n--- Step 3: Evaluating Test Results ---")
        if not self.current_differential_diagnoses:
            print("No initial diagnoses to evaluate results against. Please analyze symptoms first.")
            return ""

        evaluation_output, refined_diagnoses = self.llm_simulator.evaluate_results(self.current_differential_diagnoses, test_results)
        self.current_differential_diagnoses = refined_diagnoses
        return evaluation_output

    def get_final_diagnosis_and_treatment(self):
        print(f"\n--- Step 4: Suggesting Final Diagnosis and Treatment ---")
        if not self.current_differential_diagnoses:
            final_diagnosis_str = "No conclusive diagnosis reached yet. More information needed."
        elif len(self.current_differential_diagnoses) == 1:
            diag_name = list(self.current_differential_diagnoses.keys())[0]
            diag_desc = list(self.current_differential_diagnoses.values())[0]
            final_diagnosis_str = f"Confirmed Diagnosis: {diag_name} ({diag_desc})"
        else:
            final_diagnosis_str = "Multiple potential diagnoses still exist. Further refinement needed.\nPossible Diagnoses:\n" + "\n".join([f"  - {d}: {desc}" for d, desc in self.current_differential_diagnoses.items()])
            
        treatment_output = self.llm_simulator.suggest_treatment(final_diagnosis_str)
        return final_diagnosis_str, treatment_output

if __name__ == "__main__":
    # Scenario 1: Flu-like symptoms leading to a confirmed influenza diagnosis
    print("\n====== Scenario 1: Flu-like symptoms ======")
    assistant1 = MedicalDiagnosisAssistant()
    assistant1.analyze_symptoms("patient reports fever, cough, and fatigue for 2 days")
    assistant1.recommend_further_investigations()
    assistant1.evaluate_test_results("patient's influenza PCR test is positive, chest X-ray is normal")
    final_diag1, treatment1 = assistant1.get_final_diagnosis_and_treatment()
    print(f"\nSummary of Scenario 1:\nFinal Diagnosis: {final_diag1}\nTreatment: {treatment1}")

    # Scenario 2: Abdominal pain - appendicitis suspicion and confirmation
    print("\n====== Scenario 2: Abdominal pain - appendicitis suspicion ======")
    assistant2 = MedicalDiagnosisAssistant() # New instance for a new case
    assistant2.analyze_symptoms("severe abdominal pain starting around navel and moving to lower right, with nausea and vomiting")
    assistant2.recommend_further_investigations()
    assistant2.evaluate_test_results("abdominal ultrasound shows inflamed appendix, elevated white blood cell count")
    final_diag2, treatment2 = assistant2.get_final_diagnosis_and_treatment()
    print(f"\nSummary of Scenario 2:\nFinal Diagnosis: {final_diag2}\nTreatment: {treatment2}")

    # Scenario 3: Chest pain - cardiac concern leading to heart attack diagnosis
    print("\n====== Scenario 3: Chest pain - cardiac concern ======")
    assistant3 = MedicalDiagnosisAssistant()
    assistant3.analyze_symptoms("sudden onset of crushing chest pain radiating to left arm, sweating, and shortness of breath")
    assistant3.recommend_further_investigations()
    assistant3.evaluate_test_results("ECG shows ST elevation, cardiac troponin levels are significantly elevated")
    final_diag3, treatment3 = assistant3.get_final_diagnosis_and_treatment()
    print(f"\nSummary of Scenario 3:\nFinal Diagnosis: {final_diag3}\nTreatment: {treatment3}")

    # Scenario 4: Ambiguous symptoms - requiring more information
    print("\n====== Scenario 4: Ambiguous symptoms ======")
    assistant4 = MedicalDiagnosisAssistant()
    assistant4.analyze_symptoms("general malaise, mild headache, and occasional dizziness")
    assistant4.recommend_further_investigations()
    assistant4.evaluate_test_results("basic blood tests are normal, neurological exam is unremarkable")
    final_diag4, treatment4 = assistant4.get_final_diagnosis_and_treatment()
    print(f"\nSummary of Scenario 4:\nFinal Diagnosis: {final_diag4}\nTreatment: {treatment4}")

    # Scenario 5: Kidney stone suspicion and confirmation
    print("\n====== Scenario 5: Kidney Stone Suspicion ======")
    assistant5 = MedicalDiagnosisAssistant()
    assistant5.analyze_symptoms("severe pain in the lower back radiating to the groin, nausea, and frequent urination")
    assistant5.recommend_further_investigations()
    assistant5.evaluate_test_results("urinalysis shows microscopic hematuria, CT KUB reveals a 4mm kidney stone in the right ureter")
    final_diag5, treatment5 = assistant5.get_final_diagnosis_and_treatment()
    print(f"\nSummary of Scenario 5:\nFinal Diagnosis: {final_diag5}\nTreatment: {treatment5}")