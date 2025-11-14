
import json
from typing import List, Dict, Tuple, Optional

# --- 1. Pydantic-like Models (Simulated without actual Pydantic import) ---
class PatientData:
    """Simulates a Pydantic model for patient input data."""
    def __init__(self, symptoms: List[str], medical_history: List[str], lab_results: Dict[str, str]):
        self.symptoms = symptoms
        self.medical_history = medical_history
        self.lab_results = lab_results

    def to_dict(self):
        return {
            "symptoms": self.symptoms,
            "medical_history": self.medical_history,
            "lab_results": self.lab_results
        }

    @staticmethod
    def from_dict(data: Dict):
        return PatientData(data["symptoms"], data["medical_history"], data["lab_results"])


class DiagnosticOutput:
    """Simulates a Pydantic model for diagnostic output."""
    def __init__(
        self, 
        primary_diagnosis: str, 
        confidence_level: str, 
        reasoning_steps: List[str], 
        verification_questions_answered: Dict[str, str], 
        refined_explanation: str
    ):
        self.primary_diagnosis = primary_diagnosis
        self.confidence_level = confidence_level
        self.reasoning_steps = reasoning_steps
        self.verification_questions_answered = verification_questions_answered
        self.refined_explanation = refined_explanation

    def to_dict(self):
        return {
            "primary_diagnosis": self.primary_diagnosis,
            "confidence_level": self.confidence_level,
            "reasoning_steps": self.reasoning_steps,
            "verification_questions_answered": self.verification_questions_answered,
            "refined_explanation": self.refined_explanation
        }


# --- 2. Medical Knowledge Base (Simulated) ---
class MedicalKnowledgeBase:
    """A simulated medical knowledge base for demonstration purposes.
    In a real application, this would be a vector store (FAISS, Chroma) with actual medical data.
    """
    def __init__(self):
        self.data = {
            "common cold symptoms": "Runny nose, sore throat, cough, congestion. Typically self-limiting.",
            "influenza symptoms": "Fever, body aches, chills, fatigue, cough, sore throat. More severe than common cold. Antiviral medications may be used.",
            "appendicitis symptoms": "Right lower quadrant abdominal pain, nausea, vomiting, fever. Requires urgent surgical evaluation.",
            "diabetes diagnosis": "Elevated blood glucose levels (fasting plasma glucose >= 126 mg/dL, HbA1c >= 6.5%).",
            "hypertension treatment": "Lifestyle changes (diet, exercise), medications (ACE inhibitors, diuretics, beta-blockers).",
            "covid-19 transmission": "Primarily through respiratory droplets when an infected person coughs, sneezes, or talks.",
            "fever management": "Rest, fluids, antipyretics like acetaminophen or ibuprofen."
        }

    def search(self, query: str) -> List[str]:
        """Simulates searching the knowledge base for relevant information."""
        results = []
        query_lower = query.lower()
        for key, value in self.data.items():
            if query_lower in key or any(q_word in key for q_word in query_lower.split()):
                results.append(f"Relevant medical fact for '{query}': {value}")
            # Simple keyword matching for demo
            if any(word in value.lower() for word in query_lower.split() if len(word) > 3): 
                 results.append(f"Relevant medical fact for '{query}': {value}")
        return list(set(results)) # Return unique results


# --- 3. Prompt Engineering Module ---
class PromptEngineeringModule:
    """Generates various prompts for the LLM based on the reasoning pattern."""

    _cot_prompt_template = """
    You are a highly intelligent medical diagnostic assistant. Your task is to analyze patient data, think step-by-step, and provide an initial differential diagnosis along with your reasoning.

    Patient Symptoms: {symptoms}
    Medical History: {medical_history}
    Lab Results: {lab_results}

    Please provide a Chain-of-Thought (CoT) reasoning process. First, identify key symptoms. Second, list potential differential diagnoses. Third, evaluate each differential diagnosis against the patient's data. Fourth, propose initial diagnostic tests if needed. Finally, state your most likely initial diagnosis.

    Reasoning:
    """

    _cove_question_template = """
    Based on the following initial diagnostic reasoning and proposed diagnosis, generate 2-3 specific verification questions that challenge this diagnosis or seek further evidence. Think about common misdiagnoses, contradictory symptoms, or crucial missing information.

    Initial Reasoning: {initial_reasoning}
    Proposed Diagnosis: {initial_diagnosis}

    Verification Questions (list each on a new line):
    1.
    """

    _cove_refinement_template = """
    Re-evaluate the patient's case considering the initial reasoning, proposed diagnosis, and new information from the verification process. Identify any inconsistencies, incorporate new facts, and provide a refined, robust diagnosis with a clear explanation and a confidence level (e.g., High, Medium, Low).

    Patient Data: {patient_data}
    Initial Reasoning & Diagnosis: {initial_reasoning}
    Verification Questions and Answers: {verification_answers}

    Refined Diagnosis and Explanation:
    """

    def construct_cot_prompt(self, patient_data: PatientData) -> str:
        return self._cot_prompt_template.format(
            symptoms=", ".join(patient_data.symptoms),
            medical_history=", ".join(patient_data.medical_history),
            lab_results=json.dumps(patient_data.lab_results)
        )

    def construct_cove_question_prompt(self, initial_reasoning: str, initial_diagnosis: str) -> str:
        return self._cove_question_template.format(
            initial_reasoning=initial_reasoning,
            initial_diagnosis=initial_diagnosis
        )

    def construct_cove_refinement_prompt(
        self, patient_data: PatientData, initial_reasoning: str, verification_answers: Dict[str, str]
    ) -> str:
        return self._cove_refinement_template.format(
            patient_data=json.dumps(patient_data.to_dict()),
            initial_reasoning=initial_reasoning,
            verification_answers=json.dumps(verification_answers)
        )


# --- 4. LLM Orchestrator (Simulated) ---
class LLMOrchestrator:
    """A simulated LLM Orchestrator. In a real application, this would use Langchain
    and integrate with actual LLM APIs (e.g., OpenAI, Gemini, Hugging Face models).
    """
    def __init__(self):
        # In a real app: self.llm = ChatOpenAI(temperature=0.7) or similar
        pass

    def invoke(self, prompt: str) -> str:
        """Simulates an LLM call based on the prompt content.
        This is a placeholder for actual LLM interaction.
        """
        print(f"\n--- LLM Input (Simulated) ---\n{prompt[:500]}...\n")

        if "Chain-of-Thought (CoT)" in prompt:
            if "right lower quadrant abdominal pain" in prompt.lower() and "nausea" in prompt.lower():
                return (
                    "Reasoning:\n1. Key symptoms: Right lower quadrant abdominal pain, nausea, vomiting, fever.\n"
                    "2. Differential Diagnoses: Appendicitis, gastroenteritis, kidney stones, ovarian cyst (if female).\n"
                    "3. Evaluation: Acute onset, localized pain strongly suggests appendicitis. Gastroenteritis usually diffuse pain. Fever supports infection.\n"
                    "4. Diagnostic Tests: CBC (white blood cell count), urinalysis, abdominal ultrasound/CT scan.\n"
                    "5. Initial Diagnosis: Acute Appendicitis.\n"
                )
            elif "runny nose" in prompt.lower() and "sore throat" in prompt.lower():
                 return (
                    "Reasoning:\n1. Key symptoms: Runny nose, sore throat, cough.\n"
                    "2. Differential Diagnoses: Common cold, influenza, allergic rhinitis.\n"
                    "3. Evaluation: Symptoms are mild and typical for viral URI. No severe systemic symptoms like high fever or body aches.\n"
                    "4. Diagnostic Tests: None immediately indicated, possibly a rapid flu test if symptoms worsen.\n"
                    "5. Initial Diagnosis: Common Cold.\n"
                )
            else:
                return (
                    "Reasoning:\n1. Key symptoms: Patient presents with various symptoms.\n"
                    "2. Differential Diagnoses: Many possibilities given the general input.\n"
                    "3. Evaluation: More specific details needed for a precise evaluation.\n"
                    "4. Diagnostic Tests: Recommend a full workup based on specific chief complaint.\n"
                    "5. Initial Diagnosis: Undetermined, needs further investigation.\n"
                )
        elif "Verification Questions" in prompt:
            if "Acute Appendicitis" in prompt:
                return (
                    "1. Is the pain migratory (starting periumbilical then moving to RLQ)?\n"
                    "2. Is there rebound tenderness or guarding on physical exam?\n"
                    "3. What is the patient's white blood cell count? Is it elevated?"
                )
            elif "Common Cold" in prompt:
                 return (
                    "1. Has the patient been exposed to anyone with similar symptoms?\n"
                    "2. Are there any signs of bacterial infection, such as green/yellow discharge or prolonged fever?\n"
                    "3. Is the patient experiencing significant body aches or extreme fatigue (suggesting flu)?"
                )
            else:
                return "1. What are the most common alternative diagnoses given these symptoms?\n2. Are there any contradictory findings?"
        elif "Refined Diagnosis and Explanation" in prompt:
            if "Acute Appendicitis" in prompt and "elevated white blood cell count" in prompt.lower():
                return (
                    "Primary Diagnosis: Acute Appendicitis (High Confidence)\n"
                    "Explanation: The patient's symptoms (RLQ pain, nausea, vomiting, fever), migratory pain pattern, and elevated WBC strongly indicate acute appendicitis. "
                    "Rebound tenderness would further confirm. Urgent surgical consultation is required. No contradictory findings were identified. "
                    "The knowledge base confirmed the typical presentation and urgency."
                )
            elif "Common Cold" in prompt and "no signs of bacterial infection" in prompt.lower():
                return (
                    "Primary Diagnosis: Common Cold (High Confidence)\n"
                    "Explanation: The patient's mild respiratory symptoms, lack of severe systemic signs (like high fever or extreme fatigue), and reported exposure to similar cases are consistent with a common viral cold. "
                    "There are no red flags for bacterial complications or influenza. Treatment should focus on symptomatic relief. "
                    "The knowledge base supported typical cold symptoms and self-limiting nature."
                )
            else:
                return (
                    "Primary Diagnosis: Unclear/Further Investigation Needed (Medium Confidence)\n"
                    "Explanation: While initial reasoning pointed to a possibility, the verification process highlighted ambiguities or insufficient data. "
                    "More diagnostic tests or a more detailed patient history are required to confirm a definitive diagnosis. "
                    "The confidence level is moderate due to remaining uncertainties. Additional information on specific symptoms or recent exposures could help clarify."
                )
        return "Simulated LLM response: Could not process this specific prompt pattern."


# --- 5. Verification Module ---
class VerificationModule:
    """Handles generating verification questions, querying the knowledge base, and refining diagnosis."""
    def __init__(self, knowledge_base: MedicalKnowledgeBase, llm_orchestrator: LLMOrchestrator, prompt_engineer: PromptEngineeringModule):
        self.knowledge_base = knowledge_base
        self.llm_orchestrator = llm_orchestrator
        self.prompt_engineer = prompt_engineer

    def verify_and_refine(
        self, patient_data: PatientData, initial_reasoning: str, initial_diagnosis: str
    ) -> Tuple[Dict[str, str], str, str, str]:
        """Generates questions, answers them via KB, and refines the diagnosis."""

        # Step 1: Generate verification questions
        cove_question_prompt = self.prompt_engineer.construct_cove_question_prompt(
            initial_reasoning, initial_diagnosis
        )
        questions_raw = self.llm_orchestrator.invoke(cove_question_prompt)
        verification_questions = [q.strip() for q in questions_raw.split('\n') if q.strip() and q.strip().startswith(('1.', '2.', '3.'))]
        
        # Step 2: Answer questions using Knowledge Base
        verification_answers = {}
        for q_num, question in enumerate(verification_questions):
            answer_snippets = self.knowledge_base.search(question)
            if answer_snippets:
                # In a real system, LLM would synthesize answer from snippets
                verification_answers[question] = " ".join(answer_snippets[:2]) # Take first 2 for brevity
            else:
                verification_answers[question] = "No direct information found in knowledge base."

        # Simulate adding a 'found' lab result for appendicitis path
        if initial_diagnosis == "Acute Appendicitis" and "white blood cell count" in str(verification_answers):
             verification_answers["Simulated Lab Result"] = "Patient's WBC is 18,000 cells/uL (elevated)."
        elif initial_diagnosis == "Common Cold" and "bacterial infection" in str(verification_answers):
             verification_answers["Simulated Clinical Finding"] = "Patient has clear nasal discharge and no prolonged fever."

        # Step 3: Refine diagnosis using CoVe refinement prompt
        cove_refinement_prompt = self.prompt_engineer.construct_cove_refinement_prompt(
            patient_data, initial_reasoning, verification_answers
        )
        refined_output_raw = self.llm_orchestrator.invoke(cove_refinement_prompt)
        
        # Parse refined output
        primary_diagnosis = "Unknown"
        confidence_level = "Uncertain"
        refined_explanation = refined_output_raw

        if "Primary Diagnosis: " in refined_output_raw:
            lines = refined_output_raw.split('\n')
            for line in lines:
                if line.startswith("Primary Diagnosis: "):
                    primary_diagnosis_line = line.replace("Primary Diagnosis: ", "").strip()
                    parts = primary_diagnosis_line.split('(')
                    primary_diagnosis = parts[0].strip()
                    if len(parts) > 1:
                        confidence_level = parts[1].replace(')', '').strip()
                elif line.startswith("Explanation: "):
                    refined_explanation = line.replace("Explanation: ", "").strip()

        return primary_diagnosis, confidence_level, refined_explanation, verification_answers


# --- 6. FastAPI Application (Simulated) ---
# This section simulates the main application logic that would typically be exposed via FastAPI.
# Actual FastAPI decorators and app instance are omitted due to import restrictions.

# Initialize components
knowledge_base = MedicalKnowledgeBase()
llm_orchestrator = LLMOrchestrator()
prompt_engineer = PromptEngineeringModule()
verification_module = VerificationModule(knowledge_base, llm_orchestrator, prompt_engineer)


def diagnose_patient(patient_data_dict: Dict) -> Dict:
    """Simulates the FastAPI endpoint for diagnosing a patient."""
    print("\n*** Starting Diagnostic Process ***")
    patient_data = PatientData.from_dict(patient_data_dict)
    
    # 1. Initial Chain-of-Thought (CoT) Reasoning
    cot_prompt = prompt_engineer.construct_cot_prompt(patient_data)
    initial_cot_output = llm_orchestrator.invoke(cot_prompt)
    
    # Parse initial CoT output
    initial_reasoning_steps = [step.strip() for step in initial_cot_output.split('\n') if step.strip() and not step.startswith('Initial Diagnosis:')]
    initial_diagnosis = "Undetermined"
    for line in initial_cot_output.split('\n'):
        if line.startswith('5. Initial Diagnosis: '):
            initial_diagnosis = line.replace('5. Initial Diagnosis: ', '').strip()
            break

    print(f"\nInitial Diagnosis (CoT): {initial_diagnosis}")
    print(f"Initial Reasoning Steps:\n{initial_cot_output}")

    # 2. Chain-of-Verification (CoVe) and Refinement
    primary_diagnosis, confidence_level, refined_explanation, verification_answers = \
        verification_module.verify_and_refine(patient_data, initial_cot_output, initial_diagnosis)

    print(f"\nRefined Diagnosis: {primary_diagnosis} (Confidence: {confidence_level})")
    print(f"Refined Explanation: {refined_explanation}")
    print(f"Verification Questions & Answers: {json.dumps(verification_answers, indent=2)}")

    # Prepare final output
    final_output = DiagnosticOutput(
        primary_diagnosis=primary_diagnosis,
        confidence_level=confidence_level,
        reasoning_steps=initial_reasoning_steps + ["(Refined based on verification process)"],
        verification_questions_answered=verification_answers,
        refined_explanation=refined_explanation
    )

    print("\n*** Diagnostic Process Complete ***")
    return final_output.to_dict()


# --- Example Usage (Simulates API call) ---
if __name__ == "__main__":
    # Example 1: Suspected Appendicitis
    patient_data_1 = {
        "symptoms": ["right lower quadrant abdominal pain", "nausea", "vomiting", "fever"],
        "medical_history": ["no significant past medical history"],
        "lab_results": {"temperature": "38.5 C", "WBC": "not available"}
    }
    print("\n\n----- Diagnosing Patient 1 (Suspected Appendicitis) -----")
    output_1 = diagnose_patient(patient_data_1)
    print(f"Final Output 1:\n{json.dumps(output_1, indent=2)}")

    print("\n\n" + "="*80 + "\n\n")

    # Example 2: Common Cold
    patient_data_2 = {
        "symptoms": ["runny nose", "sore throat", "mild cough"],
        "medical_history": ["seasonal allergies"],
        "lab_results": {"temperature": "37.2 C"}
    }
    print("\n\n----- Diagnosing Patient 2 (Common Cold) -----")
    output_2 = diagnose_patient(patient_data_2)
    print(f"Final Output 2:\n{json.dumps(output_2, indent=2)}")

    print("\n\n" + "="*80 + "\n\n")

    # Example 3: General/Unclear Case
    patient_data_3 = {
        "symptoms": ["headache", "fatigue"],
        "medical_history": ["stress at work"],
        "lab_results": {"blood pressure": "120/80"}
    }
    print("\n\n----- Diagnosing Patient 3 (General/Unclear) -----")
    output_3 = diagnose_patient(patient_data_3)
    print(f"Final Output 3:\n{json.dumps(output_3, indent=2)}")
