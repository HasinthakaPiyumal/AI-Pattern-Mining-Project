from typing import List, Dict

# --- 1. Data Models (using a simplified approach as pydantic is not directly available without import) ---
class MedicalCase:
    """Represents a medical case with symptoms and potentially a known diagnosis."""
    def __init__(self, case_id: str, symptoms: str, relevant_history: str = "", known_diagnosis: str = "Unknown"):
        self.case_id = case_id
        self.symptoms = symptoms
        self.relevant_history = relevant_history
        self.known_diagnosis = known_diagnosis

    def __str__(self):
        return f"Case ID: {self.case_id}\nSymptoms: {self.symptoms}\nHistory: {self.relevant_history}"


# --- 2. Zero-Shot CoT Generator (Simulated LLM Interaction) ---
class ZeroShotCoTGenerator:
    """Simulates generating a Chain of Thought for a medical case using a Zero-Shot prompt."""

    def __init__(self, llm_model_name: str = "Simulated-Medical-LLM"):
        self.llm_model_name = llm_model_name

    def generate_cot(self, medical_case: MedicalCase, zero_shot_prompt_template: str) -> str:
        """
        Generates a diagnostic chain of thought for a given medical case.
        In a real application, this would involve an actual LLM call.
        """
        print(f"\n--- Simulating Zero-Shot CoT Generation for Case {medical_case.case_id} ---")
        prompt = zero_shot_prompt_template.format(
            symptoms=medical_case.symptoms,
            history=medical_case.relevant_history
        )
        print("Zero-Shot Prompt:\n" + prompt)

        # Simulate LLM response based on case details
        if "fever" in medical_case.symptoms.lower() and "rash" in medical_case.symptoms.lower():
            cot = (
                "The patient presents with fever and rash. \n"
                "Differential diagnoses include viral exanthems like measles, rubella, or varicella.\n"
                "Considering the specific rash characteristics and other symptoms is crucial.\n"
                "Further investigation would involve checking vaccination history, recent exposures, and specific rash morphology.\n"
                "Given the general description, a common viral infection is highly probable."
            )
        elif "chest pain" in medical_case.symptoms.lower() and "shortness of breath" in medical_case.symptoms.lower():
            cot = (
                "The patient reports chest pain and shortness of breath.\n"
                "This immediately raises concern for cardiac (e.g., MI, angina) or pulmonary (e.g., PE, pneumonia) etiologies.\n"
                "Emergency assessment of vital signs, ECG, and cardiac biomarkers is indicated.\n"
                "A detailed history differentiating cardiac vs. pulmonary pain is essential.\n"
                "Without more data, acute coronary syndrome or pulmonary embolism must be ruled out first."
            )
        elif "headache" in medical_case.symptoms.lower() and "stiff neck" in medical_case.symptoms.lower():
            cot = (
                "The patient presents with headache and stiff neck.\n"
                "These symptoms are classic for meningitis, a serious infection of the meninges.\n"
                "Other possibilities include subarachnoid hemorrhage or severe tension headache.\n"
                "Immediate medical evaluation including neurological exam and potentially lumbar puncture is critical.\n"
                "Meningitis, bacterial or viral, needs urgent differentiation and treatment."
            )
        else:
            cot = (
                "The patient presents with a set of symptoms that require careful evaluation.\n"
                "A systematic approach involves considering common etiologies first, then rarer ones.\n"
                "Diagnostic steps would typically include a detailed history, physical examination, and targeted investigations (e.g., labs, imaging) based on the most prominent symptoms.\n"
                "Without more specific information, a broad differential diagnosis is maintained, pending further data collection and expert consultation."
            )

        print("Generated CoT:\n" + cot)
        return cot


# --- 3. Few-Shot CoT Prompt Builder ---
class FewShotCoTBuilder:
    """Builds a Few-Shot CoT prompt using generated exemplars and a new case."""

    def build_few_shot_prompt(self, exemplars: List[Dict], new_case: MedicalCase, few_shot_prompt_template: str) -> str:
        """
        Constructs a few-shot prompt by embedding exemplars and the new case query.
        Each exemplar is a dictionary containing 'case' (MedicalCase) and 'cot' (str).
        """
        print("\n--- Building Few-Shot CoT Prompt ---")
        exemplar_strings = []
        for i, exemplar in enumerate(exemplars):
            exemplar_strings.append(
                f"Example {i+1}:\n"
                f"Symptoms: {exemplar['case'].symptoms}\n"
                f"History: {exemplar['case'].relevant_history}\n"
                f"Thought Process:\n{exemplar['cot']}\n"
                f"Diagnosis: {exemplar['case'].known_diagnosis}\n"
            )
        
        # The final prompt structure including the new case
        full_prompt = few_shot_prompt_template.format(
            exemplars="\n".join(exemplar_strings),
            new_symptoms=new_case.symptoms,
            new_history=new_case.relevant_history
        )
        print("Generated Few-Shot Prompt:\n" + full_prompt)
        return full_prompt


# --- 4. Medical Diagnosis Assistant (Orchestrator) ---
class MedicalDiagnosisAssistant:
    """Orchestrates the AutoCoT process for medical diagnosis assistance."""

    def __init__(self):
        self.cot_generator = ZeroShotCoTGenerator()
        self.few_shot_builder = FewShotCoTBuilder()

        self.zero_shot_template = (
            "You are a highly experienced medical diagnostician. Analyze the following patient case and provide a step-by-step diagnostic thought process.\n"
            "Patient Symptoms: {symptoms}\n"
            "Relevant History: {history}\n"
            "Diagnostic Thought Process:"
        )

        self.few_shot_template = (
            "You are an expert medical diagnostician. Based on the provided examples of diagnostic thought processes and their corresponding diagnoses, derive a likely diagnosis for the new patient case.\n"
            "{exemplars}"
            "New Patient Case:\n"
            "Symptoms: {new_symptoms}\n"
            "History: {new_history}\n"
            "Diagnostic Thought Process (following the examples):\n"
            "Diagnosis:"
        )

    def assist_diagnosis(self, new_medical_case: MedicalCase, example_cases: List[MedicalCase], num_exemplars: int = 2) -> str:
        """
        Main function to generate diagnosis using AutoCoT.
        1. Generates CoT for example cases using Zero-Shot prompting.
        2. Builds a Few-Shot prompt with these CoT exemplars.
        3. Simulates LLM call to get the final diagnosis.
        """
        print(f"\n\n--- Starting AutoCoT Diagnosis for Case: {new_medical_case.case_id} ---")
        generated_exemplars = []
        for i in range(min(num_exemplars, len(example_cases))):
            example_case = example_cases[i]
            cot = self.cot_generator.generate_cot(example_case, self.zero_shot_template)
            generated_exemplars.append({"case": example_case, "cot": cot})

        few_shot_prompt = self.few_shot_builder.build_few_shot_prompt(generated_exemplars, new_medical_case, self.few_shot_template)

        # Simulate the final LLM call for diagnosis based on the few-shot prompt
        print("\n--- Simulating Final LLM Diagnosis ---")
        # In a real scenario, this would be an actual LLM call using the few_shot_prompt
        # and the LLM would complete the 'Diagnostic Thought Process' and 'Diagnosis'.
        
        # For simulation, we'll infer based on the new case's symptoms, assuming the CoT guides it.
        if "fever" in new_medical_case.symptoms.lower() and "cough" in new_medical_case.symptoms.lower():
            final_diagnosis = "Viral Bronchitis"
            thought_process = "Considering the fever and cough, and assuming no red flags from history, a common viral respiratory infection like bronchitis is a reasonable initial diagnosis. Further tests may be needed to rule out bacterial pneumonia or influenza."
        elif "abdominal pain" in new_medical_case.symptoms.lower() and "vomiting" in new_medical_case.symptoms.lower():
            final_diagnosis = "Gastroenteritis"
            thought_process = "Acute onset abdominal pain with vomiting strongly suggests gastroenteritis, especially if there's no localized tenderness or signs of obstruction. Hydration and symptomatic relief are key."
        elif "fatigue" in new_medical_case.symptoms.lower() and "weight loss" in new_medical_case.symptoms.lower():
             final_diagnosis = "Thyroid Dysfunction or Chronic Illness Screening Recommended"
             thought_process = "Persistent fatigue and unexplained weight loss warrant investigation for systemic issues, including thyroid disorders, malignancy, or chronic infections. Comprehensive blood work is essential."
        else:
            final_diagnosis = "Requires further investigation/Specialist Consult"
            thought_process = "The symptoms are vague or complex. Following the general diagnostic principles outlined in the exemplars, a thorough physical examination, targeted lab tests, and possibly imaging are required. Referral to a relevant specialist is recommended for definitive diagnosis."

        print(f"Simulated Final Thought Process: {thought_process}")
        print(f"Simulated Final Diagnosis: {final_diagnosis}")
        return final_diagnosis


# --- Example Usage --- 
if __name__ == "__main__":
    # Define some example medical cases with known outcomes for AutoCoT to draw from
    example_cases = [
        MedicalCase("EC001", "Sudden onset fever, widespread maculopapular rash starting on face, cough, runny nose, conjunctivitis.", "No recent travel, unvaccinated child.", "Measles"),
        MedicalCase("EC002", "Crushing chest pain radiating to left arm, shortness of breath, sweating.", "History of hypertension and hyperlipidemia, 60-year-old male.", "Myocardial Infarction"),
        MedicalCase("EC003", "Severe headache, fever, stiff neck, photophobia.", "Recent upper respiratory infection.", "Bacterial Meningitis"),
        MedicalCase("EC004", "Intermittent abdominal pain, bloating, alternating constipation and diarrhea.", "Long-standing digestive issues, stress.", "Irritable Bowel Syndrome"),
        MedicalCase("EC005", "Joint pain and swelling in small joints of hands and feet, morning stiffness lasting over an hour.", "Family history of autoimmune disease.", "Rheumatoid Arthritis")
    ]

    # Define a new medical case for which we need a diagnosis
    new_case = MedicalCase(
        "NC001", 
        "Persistent low-grade fever, dry cough, and increasing fatigue over 2 weeks. \nOccasional night sweats.",
        "No significant past medical history, no recent travel or sick contacts. \nPatient reports feeling generally unwell."
    )

    assistant = MedicalDiagnosisAssistant()
    
    # Run the AutoCoT diagnosis process
    final_diagnosis = assistant.assist_diagnosis(new_case, example_cases, num_exemplars=2)
    print(f"\n\n--- AutoCoT Process Completed ---")
    print(f"For New Case {new_case.case_id}, the AutoCoT-assisted Diagnosis is: {final_diagnosis}")

    # Another example
    new_case_2 = MedicalCase(
        "NC002", 
        "Sudden severe headache, worst of my life, followed by neck stiffness and vomiting. \nNo fever.",
        "No history of migraines. 45-year-old female, known hypertension."
    )
    print("\n" + "="*80 + "\n")
    final_diagnosis_2 = assistant.assist_diagnosis(new_case_2, example_cases, num_exemplars=3)
    print(f"\n\n--- AutoCoT Process Completed ---")
    print(f"For New Case {new_case_2.case_id}, the AutoCoT-assisted Diagnosis is: {final_diagnosis_2}")