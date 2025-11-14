from abc import ABC, abstractmethod

# 1. Abstract Base Class for Medical Modules
class MedicalModule(ABC):
    """Abstract base class for all plug-and-play medical modules."""
    @abstractmethod
    def process(self, patient_data: dict) -> str:
        """Processes patient data and returns a specialized analysis string."""
        pass

# 2. Mock Large Language Model (LLM)
class MockLLM:
    """Simulates a general-purpose LLM for diagnostic suggestions."""
    def generate_response(self, prompt: str) -> str:
        """Generates a mock diagnostic response based on the prompt."""
        print(f"\n--- MockLLM received prompt ---\n{prompt}\n---\n")
        # Simple mock logic: if 'pneumonia' is in prompt, suggest related treatment
        if "pneumonia" in prompt.lower():
            return "MockLLM Diagnosis: Based on symptoms and imaging, highly suspect pneumonia. Recommend antibiotics and chest physiotherapy."
        elif "diabetes" in prompt.lower():
            return "MockLLM Diagnosis: Potential diabetes. Recommend blood glucose tests and lifestyle modifications."
        elif "drug interaction" in prompt.lower():
            return "MockLLM Diagnosis: Identified potential drug interaction. Advise reviewing medication regimen with a pharmacist/specialist before proceeding."
        else:
            return "MockLLM Diagnosis: Based on the provided context, further investigation may be needed. Consider standard diagnostic protocols."

# 3. Plug-and-Play Modules
class SymptomCheckerModule(MedicalModule):
    """Identifies potential conditions based on patient symptoms."""
    def __init__(self):
        self.symptom_to_condition = {
            "cough": "Possible Respiratory Infection or Bronchitis",
            "fever": "Possible Infection",
            "fatigue": "Possible Anemia or Chronic Fatigue Syndrome",
            "chest pain": "Possible Cardiac Issue or Pneumonia",
            "shortness of breath": "Possible Asthma, COPD, or Pneumonia",
            "high blood sugar": "Possible Diabetes",
            "nausea": "Possible Gastrointestinal Issue",
        }

    def process(self, patient_data: dict) -> str:
        symptoms = patient_data.get("symptoms", [])
        if not symptoms:
            return "Symptom Checker Module: No symptoms provided."

        potential_conditions = set()
        for symptom in symptoms:
            if symptom.lower() in self.symptom_to_condition:
                potential_conditions.add(self.symptom_to_condition[symptom.lower()])

        if potential_conditions:
            return f"Symptom Checker Module: Potential conditions based on symptoms: {', '.join(potential_conditions)}."
        else:
            return "Symptom Checker Module: No specific conditions identified from provided symptoms."

class DrugInteractionModule(MedicalModule):
    """Checks for adverse drug interactions."""
    def __init__(self):
        self.known_interactions = {
            ("warfarin", "aspirin"): "Increased bleeding risk",
            ("metformin", "iodinated contrast"): "Risk of lactic acidosis (kidney issues)",
            ("sildenafil", "nitrates"): "Severe hypotension",
        }

    def process(self, patient_data: dict) -> str:
        current_meds = patient_data.get("current_medications", [])
        new_meds = patient_data.get("new_prescriptions", [])

        if not current_meds and not new_meds:
            return "Drug Interaction Module: No medications to check for interactions."

        all_meds = [m.lower() for m in current_meds + new_meds]
        interactions_found = []

        for i, med1 in enumerate(all_meds):
            for med2 in all_meds[i+1:]:
                if (med1, med2) in self.known_interactions:
                    interactions_found.append(f"{med1} and {med2}: {self.known_interactions[(med1, med2)]}")
                elif (med2, med1) in self.known_interactions: # Check reverse order
                    interactions_found.append(f"{med2} and {med1}: {self.known_interactions[(med2, med1)]}")

        if interactions_found:
            return "Drug Interaction Module: Identified potential drug interactions:\n" + "\n".join(interactions_found)
        else:
            return "Drug Interaction Module: No significant drug interactions found."

class MedicalImagingAnalysisModule(MedicalModule):
    """Interprets simulated medical imaging reports."""
    def process(self, patient_data: dict) -> str:
        imaging_report = patient_data.get("imaging_report", "")
        if not imaging_report:
            return "Medical Imaging Analysis Module: No imaging report provided."

        findings = []
        if "consolidation" in imaging_report.lower() or "infiltrate" in imaging_report.lower():
            findings.append("Suggestive of pulmonary infection/pneumonia.")
        if "fracture" in imaging_report.lower():
            findings.append("Evidence of bone fracture.")
        if "mass" in imaging_report.lower() or "nodule" in imaging_report.lower():
            findings.append("Presence of mass or nodule, warrants further investigation.")

        if findings:
            return f"Medical Imaging Analysis Module: Key findings from imaging report: {'; '.join(findings)}"
        else:
            return "Medical Imaging Analysis Module: No specific critical findings identified from the imaging report."

class LatestResearchModule(MedicalModule):
    """Provides up-to-date information from recent medical research and clinical guidelines."""
    def __init__(self):
        self.research_snippets = {
            "pneumonia": [
                "Recent guidelines emphasize early broad-spectrum antibiotics for community-acquired pneumonia.",
                "Studies show telithromycin is effective for severe pneumonia but with cardiac risk."
            ],
            "diabetes": [
                "New GLP-1 receptor agonists show promise in both glycemic control and cardiovascular benefits for Type 2 Diabetes.",
                "Personalized dietary interventions are becoming more prevalent in diabetes management."
            ],
            "hypertension": [
                "Updated JNC guidelines suggest lower blood pressure targets for certain patient populations.",
                "Combination therapy often superior to monotherapy for achieving blood pressure goals."
            ]
        }

    def process(self, patient_data: dict) -> str:
        condition = patient_data.get("patient_condition", "").lower()
        if not condition or condition not in self.research_snippets:
            return "Latest Research Module: No specific research found for the given condition."

        snippets = self.research_snippets[condition]
        return f"Latest Research Module: Relevant research and guidelines for {condition.capitalize()}:\n- " + "\n- ".join(snippets)

# 4. Medical Diagnosis System (Orchestrator)
class MedicalDiagnosisSystem:
    """Orchestrates the interaction between patient data, plug-and-play modules, and the LLM."""
    def __init__(self, llm: MockLLM):
        self.llm = llm
        self.modules: list[MedicalModule] = []

    def register_module(self, module: MedicalModule):
        """Registers a new plug-and-play medical module."""
        self.modules.append(module)
        print(f"Registered module: {module.__class__.__name__}")

    def diagnose_patient(self, patient_data: dict) -> str:
        """Processes patient data, gathers module outputs, and queries the LLM."""
        print("\n--- Starting Diagnosis Process ---")
        print(f"Patient Data: {patient_data}")

        module_outputs = []
        for module in self.modules:
            output = module.process(patient_data)
            module_outputs.append(output)
            print(f"  [Module Output: {module.__class__.__name__}] {output.split(':', 1)[-1].strip()}")

        # Construct the prompt for the LLM
        prompt_parts = [
            "Patient Information:",
            f"  Symptoms: {', '.join(patient_data.get('symptoms', ['None']))}",
            f"  Patient History: {patient_data.get('history', 'N/A')}",
            f"  Current Medications: {', '.join(patient_data.get('current_medications', ['None']))}",
            f"  New Prescriptions: {', '.join(patient_data.get('new_prescriptions', ['None']))}",
            f"  Imaging Report: {patient_data.get('imaging_report', 'N/A')}",
            f"  Patient Condition (as perceived/known): {patient_data.get('patient_condition', 'N/A')}",
            "\nAugmentation Module Outputs:",
        ]
        prompt_parts.extend(module_outputs)
        prompt_parts.append("\nBased on all the above information, provide a concise medical diagnosis and recommended treatment plan.")

        full_prompt = "\n".join(prompt_parts)
        
        llm_response = self.llm.generate_response(full_prompt)
        print("\n--- Diagnosis Complete ---")
        return llm_response

# Example Usage
if __name__ == "__main__":
    # Initialize Mock LLM
    mock_llm = MockLLM()

    # Initialize and register modules
    diagnosis_system = MedicalDiagnosisSystem(mock_llm)
    diagnosis_system.register_module(SymptomCheckerModule())
    diagnosis_system.register_module(DrugInteractionModule())
    diagnosis_system.register_module(MedicalImagingAnalysisModule())
    diagnosis_system.register_module(LatestResearchModule())

    # Define patient data scenarios
    patient_data_1 = {
        "symptoms": ["cough", "fever", "shortness of breath", "chest pain"],
        "history": "Smoker, recent flu-like symptoms",
        "current_medications": [],
        "new_prescriptions": [],
        "imaging_report": "Chest X-ray shows right lower lobe consolidation.",
        "patient_condition": "pneumonia" # For research module context
    }

    patient_data_2 = {
        "symptoms": ["fatigue", "increased thirst", "high blood sugar (reported)"],
        "history": "Family history of diabetes",
        "current_medications": ["lisinopril"], # For testing drug interactions with a new med
        "new_prescriptions": ["metformin"],
        "imaging_report": "",
        "patient_condition": "diabetes" # For research module context
    }

    patient_data_3 = {
        "symptoms": ["mild headache"],
        "history": "No significant history",
        "current_medications": ["warfarin"],
        "new_prescriptions": ["aspirin"], # Known interaction
        "imaging_report": "MRI Brain: Normal findings.",
        "patient_condition": "general checkup" # No specific research
    }

    # Run diagnosis for patient 1
    print("\n=======================================================")
    print("Diagnosing Patient 1 (Suspected Pneumonia)")
    print("=======================================================")
    diagnosis_result_1 = diagnosis_system.diagnose_patient(patient_data_1)
    print(f"Final Diagnosis Result for Patient 1: {diagnosis_result_1}")

    # Run diagnosis for patient 2
    print("\n=======================================================")
    print("Diagnosing Patient 2 (Suspected Diabetes)")
    print("=======================================================")
    diagnosis_result_2 = diagnosis_system.diagnose_patient(patient_data_2)
    print(f"Final Diagnosis Result for Patient 2: {diagnosis_result_2}")

    # Run diagnosis for patient 3
    print("\n=======================================================")
    print("Diagnosing Patient 3 (Drug Interaction Scenario)")
    print("=======================================================")
    diagnosis_result_3 = diagnosis_system.diagnose_patient(patient_data_3)
    print(f"Final Diagnosis Result for Patient 3: {diagnosis_result_3}")