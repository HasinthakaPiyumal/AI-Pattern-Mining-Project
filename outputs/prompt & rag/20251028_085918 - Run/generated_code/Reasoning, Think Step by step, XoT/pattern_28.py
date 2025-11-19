
from typing import List, Dict, Any
from pydantic import BaseModel

# --- Pydantic Models for Data Structuring ---

class PatientInfo(BaseModel):
    symptoms: List[str]
    medical_history: Dict[str, Any]

class ReasoningStep(BaseModel):
    step_number: int
    description: str
    intermediate_result: Any = None

class DiagnosticHypothesis(BaseModel):
    diagnosis: str
    confidence: float
    reasoning_path: List[ReasoningStep]

class VerifiedDiagnosis(BaseModel):
    final_diagnosis: str
    verified_reasoning: List[ReasoningStep]
    verification_status: str # e.g., "Verified", "Partially Verified", "Unverified"
    confidence_score: float
    notes: str = None

# --- LLM Abstraction (Placeholder) ---

class LLMWrapper:
    """Simulates an LLM call for generating text based on prompts."""
    def __init__(self, model_name: str = "gpt-4-like"):
        self.model_name = model_name

    def generate_text(self, prompt: str, max_tokens: int = 500) -> str:
        # In a real application, this would involve calling an actual LLM API
        # For demonstration, we'll return a mock response.
        print(f"[LLM simulating response for prompt (first 100 chars): {prompt[:100]}...]\n")
        if "chain of thought" in prompt.lower():
            return (
                "Step 1: Patient presents with headache and fever. These are common symptoms for various conditions.\n" +
                "Step 2: Considering medical history of recent travel, infections like influenza or viral meningitis are possible.\n" +
                "Step 3: Without further specific symptoms (e.g., stiff neck for meningitis), influenza is a more common initial consideration.\n" +
                "Therefore, an initial hypothesis is Influenza." 
            )
        elif "tree of thoughts" in prompt.lower():
            return (
                "Path A (Influenza): Headache, fever, recent travel. Hypothesis: Viral infection. Confidence: 0.7.\n" +
                "Path B (Meningitis): Headache, fever, *lack* of stiff neck but *possible* other neurological signs. Hypothesis: Bacterial or viral meningitis. Confidence: 0.3.\n" +
                "Path C (Migraine): Headache, no fever, no recent travel. Hypothesis: Primary headache disorder. Confidence: 0.1 (less likely given fever)." 
            )
        elif "verify" in prompt.lower() and "influenza" in prompt.lower():
            return "Verification: Influenza symptoms (headache, fever, body aches) align with presented symptoms and recent travel context. Medical guidelines confirm this is a plausible initial diagnosis. Confidence in reasoning: High."
        else:
            return "Mock LLM response: Based on the input, a general reasoning process would be applied."


# --- Structured Reasoning Modules ---

class ChainOfThoughtModule:
    """Generates a step-by-step reasoning path towards a diagnosis."""
    def __init__(self, llm: LLMWrapper):
        self.llm = llm

    def reason(self, patient_info: PatientInfo) -> List[ReasoningStep]:
        prompt = (
            f"Given the patient's symptoms: {', '.join(patient_info.symptoms)} "
            f"and medical history: {patient_info.medical_history}. "
            "Please provide a detailed chain of thought for a potential diagnosis, "
            "breaking down your reasoning step-by-step." 
        )
        llm_response = self.llm.generate_text(prompt)
        
        # Parse the LLM response into ReasoningStep objects
        steps = []
        for i, line in enumerate(llm_response.split('\n')):
            if line.strip():
                steps.append(ReasoningStep(step_number=i+1, description=line.strip()))
        return steps


class TreeOfThoughtsModule:
    """Explores multiple diagnostic hypotheses and their reasoning paths."""
    def __init__(self, llm: LLMWrapper):
        self.llm = llm

    def explore_hypotheses(self, patient_info: PatientInfo, initial_reasoning: List[ReasoningStep]) -> List[DiagnosticHypothesis]:
        initial_cot = "\n".join([step.description for step in initial_reasoning])
        prompt = (
            f"Given the patient's symptoms: {', '.join(patient_info.symptoms)} "
            f"and medical history: {patient_info.medical_history}. "
            f"Initial reasoning pathway: {initial_cot}\n\n" 
            "Based on this, generate several distinct diagnostic hypotheses. "
            "For each hypothesis, briefly outline its reasoning path and an estimated confidence score. "
            "Structure this as multiple 'Path X (Diagnosis): Reasoning. Confidence: Y.' entries." 
        )
        llm_response = self.llm.generate_text(prompt)
        
        hypotheses = []
        for line in llm_response.split('\n'):
            if "Path " in line and "Hypothesis: " in line and "Confidence: " in line:
                try:
                    diagnosis_part = line.split("Hypothesis:")[1].split(". Confidence:")[0].strip()
                    confidence_str = line.split("Confidence:")[1].split('.')[0].strip()
                    confidence = float(confidence_str) / 100 if float(confidence_str) > 1 else float(confidence_str)
                    reasoning_desc = line.split(":")[1].split("Hypothesis:")[0].strip()
                    hypotheses.append(DiagnosticHypothesis(
                        diagnosis=diagnosis_part,
                        confidence=confidence,
                        reasoning_path=[ReasoningStep(step_number=1, description=reasoning_desc)] # Simplified for mock
                    ))
                except Exception as e:
                    print(f"Error parsing hypothesis line '{line}': {e}")
                    continue
        return hypotheses


class VerifierModule:
    """Verifies the accuracy and consistency of reasoning and diagnoses against a knowledge base."""
    def __init__(self, llm: LLMWrapper, medical_knowledge_base: Dict[str, Any] = None):
        self.llm = llm
        # In a real system, this would be a sophisticated medical knowledge base lookup
        self.medical_knowledge_base = medical_knowledge_base or {
            "Influenza": {
                "symptoms": ["fever", "headache", "body aches", "cough", "sore throat"],
                "risk_factors": ["recent travel", "seasonal exposure"],
                "treatment": "rest, fluids, antivirals"
            },
            "Meningitis": {
                "symptoms": ["fever", "headache", "stiff neck", "nausea", "vomiting"],
                "risk_factors": ["immunocompromised", "close contact"],
                "treatment": "antibiotics, antivirals"
            }
        }

    def verify(self, hypothesis: DiagnosticHypothesis, patient_info: PatientInfo) -> VerifiedDiagnosis:
        prompt = (
            f"Verify the following diagnostic hypothesis: {hypothesis.diagnosis}. "
            f"Reasoning provided: {hypothesis.reasoning_path[0].description if hypothesis.reasoning_path else 'N/A'}. "
            f"Patient symptoms: {', '.join(patient_info.symptoms)}. "
            f"Medical history: {patient_info.medical_history}. "
            "Cross-reference this against general medical knowledge. Assess truthfulness, consistency, and potential hallucinations. "
            "Provide a verification status, a refined confidence score, and any notes." 
        )
        llm_response = self.llm.generate_text(prompt)
        
        # Mock parsing of verification response
        if "Verification: Influenza symptoms" in llm_response:
            return VerifiedDiagnosis(
                final_diagnosis=hypothesis.diagnosis,
                verified_reasoning=hypothesis.reasoning_path,
                verification_status="Verified",
                confidence_score=min(1.0, hypothesis.confidence + 0.1), # Boost confidence slightly on verification
                notes="Symptoms and history strongly align with medical guidelines for influenza."
            )
        else:
            return VerifiedDiagnosis(
                final_diagnosis=hypothesis.diagnosis,
                verified_reasoning=hypothesis.reasoning_path,
                verification_status="Unverified" if "Error" in llm_response else "Partially Verified",
                confidence_score=max(0.0, hypothesis.confidence - 0.2), # Reduce confidence if not fully verified
                notes=llm_response
            )


# --- Main Medical Diagnostic Assistant ---

class MedicalDiagnosticAssistant:
    """Orchestrates structured and verified reasoning for medical diagnostics."""
    def __init__(self, llm: LLMWrapper):
        self.cot_module = ChainOfThoughtModule(llm)
        self.tot_module = TreeOfThoughtsModule(llm)
        self.verifier_module = VerifierModule(llm)

    def diagnose(self, patient_info: PatientInfo) -> List[VerifiedDiagnosis]:
        print("\n--- Step 1: Chain of Thought Reasoning ---")
        initial_reasoning = self.cot_module.reason(patient_info)
        print("Initial Reasoning Path:")
        for step in initial_reasoning:
            print(f"  {step.step_number}. {step.description}")

        print("\n--- Step 2: Tree of Thoughts - Exploring Hypotheses ---")
        hypotheses = self.tot_module.explore_hypotheses(patient_info, initial_reasoning)
        print("Explored Hypotheses:")
        for hyp in hypotheses:
            print(f"  - {hyp.diagnosis} (Confidence: {hyp.confidence:.2f}) - Reasoning: {hyp.reasoning_path[0].description if hyp.reasoning_path else 'N/A'}")

        print("\n--- Step 3: Verification of Hypotheses ---")
        verified_diagnoses = []
        for hyp in hypotheses:
            print(f"  Verifying: {hyp.diagnosis}...")
            verified_diag = self.verifier_module.verify(hyp, patient_info)
            verified_diagnoses.append(verified_diag)
            print(f"    -> Status: {verified_diag.verification_status}, Confidence: {verified_diag.confidence_score:.2f}, Notes: {verified_diag.notes}")
        
        # Sort by confidence and prioritize verified ones
        verified_diagnoses.sort(key=lambda x: (x.verification_status == "Verified", x.confidence_score), reverse=True)

        return verified_diagnoses


# --- Example Usage ---
if __name__ == "__main__":
    # Initialize LLM wrapper
    mock_llm = LLMWrapper()

    # Initialize the diagnostic assistant
    assistant = MedicalDiagnosticAssistant(mock_llm)

    # Define patient information
    patient1_info = PatientInfo(
        symptoms=["headache", "fever", "body aches"],
        medical_history={
            "age": 35,
            "gender": "female",
            "recent_travel": "Europe (2 weeks ago)",
            "allergies": [],
            "pre_existing_conditions": []
        }
    )

    print("\n===== Diagnosing Patient 1 ====")
    final_diagnoses = assistant.diagnose(patient1_info)

    print("\n--- Final Diagnoses (Sorted by Confidence and Verification Status) ---")
    for diag in final_diagnoses:
        print(f"Diagnosis: {diag.final_diagnosis}")
        print(f"  Status: {diag.verification_status}")
        print(f"  Confidence: {diag.confidence_score:.2f}")
        print(f"  Reasoning: {' '.join([s.description for s in diag.verified_reasoning])}")
        print(f"  Notes: {diag.notes}\n")

    print("\n===== Diagnosing Another Patient (Less clear case for demonstration) ====")
    patient2_info = PatientInfo(
        symptoms=["mild cough", "fatigue", "sore throat"],
        medical_history={
            "age": 28,
            "gender": "male",
            "recent_travel": "None",
            "allergies": ["pollen"],
            "pre_existing_conditions": []
        }
    )

    final_diagnoses_2 = assistant.diagnose(patient2_info)

    print("\n--- Final Diagnoses (Patient 2) ---")
    for diag in final_diagnoses_2:
        print(f"Diagnosis: {diag.final_diagnosis}")
        print(f"  Status: {diag.verification_status}")
        print(f"  Confidence: {diag.confidence_score:.2f}")
        print(f"  Reasoning: {' '.join([s.description for s in diag.verified_reasoning])}")
        print(f"  Notes: {diag.notes}\n")

