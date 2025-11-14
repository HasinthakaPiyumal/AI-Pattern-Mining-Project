from typing import List, Dict, Any
from pydantic import BaseModel, Field

# --- Pydantic Models for Structured Output ---
class ReasoningStep(BaseModel):
    step_number: int
    description: str
    output: str

class DiagnosticHypothesis(BaseModel):
    condition: str
    likelihood: float  # A score from 0.0 to 1.0
    reasoning_summary: str

class MedicalDiagnosis(BaseModel):
    final_diagnosis: str
    confidence: float
    detailed_reasoning_path: List[ReasoningStep]
    differential_diagnoses: List[DiagnosticHypothesis]
    verification_status: str
    ensembling_notes: str

# --- Simulated LLM for demonstration ---
class SimulatedLLM:
    def generate(self, prompt: str) -> str:
        # In a real application, this would call an actual LLM API
        # For demonstration, we'll return canned responses based on keywords
        if "initial reasoning" in prompt.lower():
            return "Step 1: Analyzing patient's chief complaint of fever and cough. Step 2: Considering infectious causes like flu, common cold, or pneumonia. Step 3: Noting duration and severity. Initial hypothesis: Viral infection." 
        elif "decompose problem" in prompt.lower():
            return "Key Symptoms: Fever (101F), Cough (persistent, dry), Fatigue. Medical History: None significant. Current Medications: None. Potential Categories: Respiratory infections, allergic reactions." 
        elif "self-correction and verification" in prompt.lower():
            return "Initial hypothesis (viral infection) seems plausible. Verifying against common symptoms of bacterial pneumonia (productive cough, higher fever) - not present. No signs of allergic reaction. Hypothesis holds." 
        elif "alternative reasoning" in prompt.lower():
            return "Expert 1 Opinion: Likely viral bronchitis. Expert 2 Opinion: Could be early stage atypical pneumonia, recommend chest X-ray. Expert 3 Opinion: Common cold, monitor symptoms." 
        elif "final aggregation" in prompt.lower():
            return "Aggregated result: Most consistent with viral bronchitis, but atypical pneumonia cannot be fully ruled out without further tests. Recommending symptomatic treatment and follow-up if symptoms worsen." 
        else:
            return "Simulated LLM response for: " + prompt


class MedicalDiagnosticAssistant:
    def __init__(self, llm: Any):
        self.llm = llm

    def diagnose(self, patient_data: Dict[str, Any]) -> MedicalDiagnosis:
        print(f"\n--- Starting Diagnosis for Patient: {patient_data.get('patient_id', 'N/A')} ---")
        detailed_reasoning_path = []

        # Step 1: Problem Decomposition & Initial Reasoning (Chain-of-Thought)
        print("\n[Step 1] Initial Reasoning and Problem Decomposition...")
        cot_prompt = f"Given the patient data: {patient_data}, perform initial reasoning and decompose the problem into key symptoms and potential categories." 
        cot_response = self.llm.generate(cot_prompt)
        detailed_reasoning_path.append(ReasoningStep(step_number=1, description="Initial Chain-of-Thought Reasoning and Problem Decomposition", output=cot_response))
        print(f"  CoT Response: {cot_response}")

        # Extracting initial hypothesis and differential for demonstration
        initial_hypothesis_text = "Viral infection (e.g., bronchitis)"
        differential_list = [
            DiagnosticHypothesis(condition="Viral Bronchitis", likelihood=0.7, reasoning_summary="Based on dry cough, fever, fatigue."),
            DiagnosticHypothesis(condition="Common Cold", likelihood=0.6, reasoning_summary="Similar symptoms, generally milder."),
            DiagnosticHypothesis(condition="Early Atypical Pneumonia", likelihood=0.3, reasoning_summary="Cannot be entirely ruled out without imaging."),
        ]

        # Step 2: Self-Correction & Verification
        print("\n[Step 2] Self-Correction and Verification...")
        verification_prompt = f"Evaluate the initial hypothesis '{initial_hypothesis_text}' for faithfulness and consistency against patient data: {patient_data}. Also consider potential counter-evidence." 
        verification_response = self.llm.generate(verification_prompt)
        detailed_reasoning_path.append(ReasoningStep(step_number=2, description="Self-Correction and Verification of Initial Hypothesis", output=verification_response))
        print(f"  Verification Response: {verification_response}")

        verification_status = "Verified as plausible, with minor caveats."
        if "not present" in verification_response.lower() or "hypothesis holds" in verification_response.lower():
            verification_status = "Hypothesis largely confirmed by verification."
        elif "inconsistent" in verification_response.lower():
            verification_status = "Hypothesis requires reconsideration."

        # Step 3: Robust Aggregation / Ensembling
        print("\n[Step 3] Robust Aggregation / Ensembling...")
        ensemble_prompt = f"Given the patient data {patient_data} and initial hypothesis '{initial_hypothesis_text}', solicit alternative reasoning paths or expert opinions to reduce variance and improve accuracy." 
        ensemble_response = self.llm.generate(ensemble_prompt)
        detailed_reasoning_path.append(ReasoningStep(step_number=3, description="Ensembling Multiple Reasoning Paths/Expert Opinions", output=ensemble_response))
        print(f"  Ensembling Response: {ensemble_response}")
        
        # Simulate final diagnosis based on ensembling
        final_diagnosis_text = "Viral Bronchitis"
        final_confidence = 0.8
        if "atypical pneumonia" in ensemble_response.lower():
            final_diagnosis_text += " (consider atypical pneumonia if no improvement)"
            final_confidence = 0.75 # Lower confidence if more uncertainty

        # Step 4: Final Output Generation
        print("\n[Step 4] Generating Final Medical Diagnosis...")
        final_notes = self.llm.generate("final aggregation of all reasoning steps")
        detailed_reasoning_path.append(ReasoningStep(step_number=4, description="Final Aggregation and Diagnosis Formulation", output=final_notes))
        print(f"  Final Notes: {final_notes}")

        return MedicalDiagnosis(
            final_diagnosis=final_diagnosis_text,
            confidence=final_confidence,
            detailed_reasoning_path=detailed_reasoning_path,
            differential_diagnoses=differential_list,
            verification_status=verification_status,
            ensembling_notes=ensemble_response # Store the raw ensembling output for transparency
        )


# --- Example Usage ---
if __name__ == "__main__":
    # Initialize the simulated LLM
    simulated_llm = SimulatedLLM()

    # Initialize the Medical Diagnostic Assistant
    assistant = MedicalDiagnosticAssistant(llm=simulated_llm)

    # Define patient data
    patient_case_1 = {
        "patient_id": "P001",
        "chief_complaint": "Fever, persistent dry cough, fatigue",
        "symptoms": ["fever (101F)", "dry cough", "fatigue", "sore throat (mild)"],
        "medical_history": "None significant",
        "medications": "None",
        "physical_exam": "Clear lung sounds, no rash, mild pharyngeal redness",
        "lab_results": "Pending"
    }

    patient_case_2 = {
        "patient_id": "P002",
        "chief_complaint": "Severe headache, stiff neck, sensitivity to light",
        "symptoms": ["severe headache", "stiff neck", "photophobia", "fever (102F)", "nausea"],
        "medical_history": "Hypertension",
        "medications": "Lisinopril",
        "physical_exam": "Positive Brudzinski's sign, normal reflexes",
        "lab_results": "Lumbar puncture pending"
    }

    # Get diagnosis for patient case 1
    diagnosis_p001 = assistant.diagnose(patient_case_1)
    print("\n--- Final Diagnosis for P001 ---")
    print(diagnosis_p001.model_dump_json(indent=2))

    # Get diagnosis for patient case 2 (this will use simulated responses, not actual reasoning for meningitis)
    diagnosis_p002 = assistant.diagnose(patient_case_2)
    print("\n--- Final Diagnosis for P002 (Simulated) ---")
    print(diagnosis_p002.model_dump_json(indent=2))
