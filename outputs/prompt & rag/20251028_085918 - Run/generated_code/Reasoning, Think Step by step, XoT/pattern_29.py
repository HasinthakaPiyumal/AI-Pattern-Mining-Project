import random

class MockLLM:
    def __init__(self, name="MockLLM", sometimes_incorrect=False):
        self.name = name
        self.sometimes_incorrect = sometimes_incorrect

    def generate(self, prompt):
        if self.sometimes_incorrect and random.random() < 0.2: # 20% chance of being incorrect
            if "differential diagnosis" in prompt.lower():
                return "Incorrect differential diagnosis: Common Cold, Flu, Heart Attack"
            elif "most likely diagnosis" in prompt.lower():
                return "Incorrect most likely diagnosis: Malaria"
            elif "treatment" in prompt.lower():
                return "Incorrect treatment: Recommend surgery for a headache"
            else:
                return f"[INCORRECT SIMULATION] {prompt.replace('patient data', 'invalid data').replace('symptoms', 'wrong symptoms')}"
        else:
            if "differential diagnosis" in prompt.lower() and "fever" in prompt.lower() and "cough" in prompt.lower():
                return "Differential diagnosis: Viral Infection (Common Cold, Flu), Bacterial Pneumonia, Bronchitis"
            elif "most likely diagnosis" in prompt.lower() and "viral infection" in prompt.lower():
                return "Most likely diagnosis: Viral Infection (Common Cold)"
            elif "treatment for Viral Infection (Common Cold)" in prompt.lower():
                return "Recommended treatment: Rest, Hydration, Symptomatic relief (e.g., pain relievers, cough suppressants)."
            elif "differential diagnosis" in prompt.lower():
                return "Differential diagnosis: Placeholder condition 1, Placeholder condition 2"
            elif "most likely diagnosis" in prompt.lower():
                return "Most likely diagnosis: Placeholder condition 1"
            elif "treatment" in prompt.lower():
                return "Recommended treatment: Placeholder treatment for Placeholder condition 1."
            else:
                return f"[SIMULATION] Processing: {prompt}"

class ReasoningVerifier:
    def verify(self, step_name, reasoning_output, expected_keywords=None):
        if expected_keywords is None:
            expected_keywords = []

        is_verified = True
        feedback = []

        for keyword in expected_keywords:
            if keyword.lower() not in reasoning_output.lower():
                is_verified = False
                feedback.append(f"Missing expected keyword: '{keyword}'")
        
        if "incorrect" in reasoning_output.lower() or "invalid" in reasoning_output.lower() or "wrong" in reasoning_output.lower():
            is_verified = False
            feedback.append("Output contains negative keywords indicating potential error.")

        if is_verified:
            return True, "Verification successful: Output appears consistent."
        else:
            return False, f"Verification failed for {step_name}: {'; '.join(feedback)}"

class MedicalDiagnosisSystem:
    def __init__(self):
        self.reasoning_llm = MockLLM(name="Reasoning LLM")
        self.verifier = ReasoningVerifier()
        self.reasoning_steps = []

    def _log_step(self, step_name, prompt, llm_output, verification_status, verification_feedback):
        self.reasoning_steps.append({
            "step_name": step_name,
            "prompt": prompt,
            "llm_output": llm_output,
            "verification_status": verification_status,
            "verification_feedback": verification_feedback
        })

    def diagnose_and_recommend(self, patient_data):
        self.reasoning_steps = [] # Reset for new diagnosis

        symptoms = patient_data.get("symptoms", "")
        lab_results = patient_data.get("lab_results", "")
        history = patient_data.get("history", "")

        # Step 1: Initial Symptom Analysis and Differential Diagnosis (Chain-of-Thought)
        prompt_dd = f"Based on patient's symptoms: '{symptoms}', history: '{history}', provide a differential diagnosis."
        dd_output = self.reasoning_llm.generate(prompt_dd)
        dd_verified, dd_feedback = self.verifier.verify("Differential Diagnosis", dd_output, ["diagnosis", "infection"])
        self._log_step("Differential Diagnosis", prompt_dd, dd_output, dd_verified, dd_feedback)
        differential_diagnosis = dd_output

        # Step 2: Integrate Lab Results and Refine Diagnosis
        prompt_refine = f"Given the differential diagnosis: '{differential_diagnosis}', and lab results: '{lab_results}', refine the diagnosis to identify the most likely condition."
        refine_output = self.reasoning_llm.generate(prompt_refine)
        refine_verified, refine_feedback = self.verifier.verify("Refined Diagnosis", refine_output, ["most likely diagnosis"])
        self._log_step("Refined Diagnosis", prompt_refine, refine_output, refine_verified, refine_feedback)
        most_likely_diagnosis = refine_output

        # Step 3: Treatment Recommendation
        prompt_treatment = f"For the most likely diagnosis: '{most_likely_diagnosis}', recommend appropriate treatment plans."
        treatment_output = self.reasoning_llm.generate(prompt_treatment)
        treatment_verified, treatment_feedback = self.verifier.verify("Treatment Recommendation", treatment_output, ["treatment", "recommend"])
        self._log_step("Treatment Recommendation", prompt_treatment, treatment_output, treatment_verified, treatment_feedback)
        recommended_treatment = treatment_output

        return {
            "final_diagnosis_summary": most_likely_diagnosis,
            "final_treatment_summary": recommended_treatment,
            "reasoning_process": self.reasoning_steps
        }

if __name__ == "__main__":
    system = MedicalDiagnosisSystem()

    # Patient Data Example 1 (Normal Scenario)
    patient_data_1 = {
        "symptoms": "fever, cough, fatigue",
        "lab_results": "normal white blood cell count",
        "history": "no recent travel"
    }
    print("\n--- Patient 1: Normal Scenario ---")
    result_1 = system.diagnose_and_recommend(patient_data_1)
    for step in result_1["reasoning_process"]:
        print(f"\nStep: {step['step_name']}")
        print(f"  Prompt: {step['prompt']}")
        print(f"  LLM Output: {step['llm_output']}")
        print(f"  Verification: {step['verification_status']} - {step['verification_feedback']}")
    print(f"\nFinal Diagnosis: {result_1['final_diagnosis_summary']}")
    print(f"Final Treatment: {result_1['final_treatment_summary']}")

    # Patient Data Example 2 (Simulating an incorrect LLM response for verification failure)
    print("\n--- Patient 2: Incorrect LLM Response Scenario ---")
    system.reasoning_llm.sometimes_incorrect = True # Enable incorrect responses
    patient_data_2 = {
        "symptoms": "severe headache, nausea",
        "lab_results": "high blood pressure",
        "history": "no prior migraines"
    }
    result_2 = system.diagnose_and_recommend(patient_data_2)
    for step in result_2["reasoning_process"]:
        print(f"\nStep: {step['step_name']}")
        print(f"  Prompt: {step['prompt']}")
        print(f"  LLM Output: {step['llm_output']}")
        print(f"  Verification: {step['verification_status']} - {step['verification_feedback']}")
    print(f"\nFinal Diagnosis: {result_2['final_diagnosis_summary']}")
    print(f"Final Treatment: {result_2['final_treatment_summary']}")
    system.reasoning_llm.sometimes_incorrect = False # Disable incorrect responses

    # Patient Data Example 3 (Another normal scenario to show reset)
    print("\n--- Patient 3: Another Normal Scenario ---")
    patient_data_3 = {
        "symptoms": "sore throat, runny nose",
        "lab_results": "",
        "history": "exposed to sick colleague"
    }
    result_3 = system.diagnose_and_recommend(patient_data_3)
    for step in result_3["reasoning_process"]:
        print(f"\nStep: {step['step_name']}")
        print(f"  Prompt: {step['prompt']}")
        print(f"  LLM Output: {step['llm_output']}")
        print(f"  Verification: {step['verification_status']} - {step['verification_feedback']}")
    print(f"\nFinal Diagnosis: {result_3['final_diagnosis_summary']}")
    print(f"Final Treatment: {result_3['final_treatment_summary']}")
