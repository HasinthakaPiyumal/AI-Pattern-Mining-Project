import os
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field
from langchain.llms import Ollama  # Or ChatOpenAI, etc.
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from loguru import logger

# --- 1. Pydantic Models for Data Structuring ---

class CoTStep(BaseModel):
    step_number: int = Field(..., description="The sequential number of the reasoning step.")
    thought: str = Field(..., description="The detailed thought process for this step.")
    action: str = Field(..., description="The action taken or information sought in this step.")
    observation: Optional[str] = Field(None, description="The outcome or result of the action.")

class Diagnosis(BaseModel):
    disease: str = Field(..., description="The identified disease or condition.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for the diagnosis (0.0 to 1.0).")
    reasoning_summary: str = Field(..., description="A concise summary of the key reasoning steps leading to this diagnosis.")
    intermediate_thoughts: List[CoTStep] = Field(default_factory=list, description="Detailed Chain of Thought steps.")

class TreatmentPlan(BaseModel):
    diagnosis_id: str = Field(..., description="ID of the diagnosis this treatment plan corresponds to.")
    medications: List[str] = Field(default_factory=list, description="List of recommended medications.")
    procedures: List[str] = Field(default_factory=list, description="List of recommended procedures.")
    lifestyle_changes: List[str] = Field(default_factory=list, description="List of recommended lifestyle modifications.")
    follow_up_recommendations: str = Field(..., description="Recommendations for follow-up care.")
    rationale: str = Field(..., description="Explanation for the proposed treatment plan.")

class PatientData(BaseModel):
    patient_id: str
    symptoms: List[str]
    medical_history: List[str]
    lab_results: Dict[str, Any] # e.g., {'blood_pressure': '140/90', 'glucose': 180}
    age: Optional[int] = None
    gender: Optional[str] = None

# --- Simulated Medical Knowledge Base ---

class MedicalKnowledgeBase:
    """A simulated medical knowledge base for verification and information retrieval."""
    def __init__(self):
        self.knowledge = {
            "diabetes": {
                "symptoms": ["frequent urination", "increased thirst", "fatigue", "blurred vision"],
                "lab_results": {"glucose_level_high": True, "hba1c_high": True},
                "treatment": ["metformin", "insulin", "dietary changes", "exercise"]
            },
            "hypertension": {
                "symptoms": ["headaches", "shortness of breath", "nosebleeds"], # Often asymptomatic
                "lab_results": {"blood_pressure_high": True},
                "treatment": ["ACE inhibitors", "diuretics", "low-sodium diet", "exercise"]
            },
            "common cold": {
                "symptoms": ["runny nose", "sore throat", "cough", "sneezing"],
                "lab_results": {},
                "treatment": ["rest", "fluids", "over-the-counter medication"]
            },
            "pneumonia": {
                "symptoms": ["cough with phlegm", "fever", "chills", "shortness of breath", "chest pain"],
                "lab_results": {"chest_xray_infiltrates": True},
                "treatment": ["antibiotics", "antivirals"], # Depending on cause
            }
        }

    def get_info(self, query: str) -> Optional[Dict[str, Any]]:
        logger.info(f"Querying knowledge base for: {query}")
        query_lower = query.lower()
        for condition, data in self.knowledge.items():
            if condition in query_lower:
                return data
            for symptom in data.get("symptoms", []):
                if symptom in query_lower:
                    return data
        return None

    def verify_symptom_diagnosis_consistency(self, symptoms: List[str], diagnosis_disease: str) -> bool:
        info = self.get_info(diagnosis_disease)
        if not info:
            logger.warning(f"No knowledge found for {diagnosis_disease} to verify consistency.")
            return True # Cannot verify, assume consistent for now

        known_symptoms = set(info.get("symptoms", []))
        patient_symptoms = set(symptoms)

        # Check if key patient symptoms align with known symptoms of the diagnosis
        # This is a simple check, could be more sophisticated
        matching_symptoms = known_symptoms.intersection(patient_symptoms)
        if len(matching_symptoms) > 0 and len(matching_symptoms) >= len(patient_symptoms) / 2:
            logger.info(f"Symptoms {patient_symptoms} show good consistency with {diagnosis_disease}.")
            return True
        elif len(patient_symptoms) == 0: # If no symptoms provided, hard to verify
            return True
        else:
            logger.warning(f"Symptoms {patient_symptoms} show low consistency with {diagnosis_disease}. Known symptoms: {known_symptoms}")
            return False

    def verify_lab_results_diagnosis_consistency(self, lab_results: Dict[str, Any], diagnosis_disease: str) -> bool:
        info = self.get_info(diagnosis_disease)
        if not info or not info.get("lab_results"):
            logger.warning(f"No relevant lab result knowledge for {diagnosis_disease} to verify consistency.")
            return True

        known_lab_indicators = info.get("lab_results", {})

        for indicator, expected_value in known_lab_indicators.items():
            # This is a simplified check. Real-world would need ranges, specific values, etc.
            if indicator in lab_results:
                # For boolean indicators
                if isinstance(expected_value, bool) and expected_value == lab_results[indicator]:
                    logger.info(f"Lab indicator {indicator} matches for {diagnosis_disease}.")
                    continue
                # For generic presence (e.g., if a specific lab test indicates something)
                elif expected_value is not None and lab_results[indicator] is not None:
                    logger.info(f"Lab indicator {indicator} present for {diagnosis_disease}.")
                    continue # Assume consistency if present and expected
            else:
                logger.warning(f"Expected lab indicator {indicator} for {diagnosis_disease} not found in patient data.")
                return False # Missing expected lab result
        logger.info(f"Lab results show consistency with {diagnosis_disease}.")
        return True

# --- LLM and Prompt Setup ---

# Placeholder for LLM - replace with your actual LLM setup
# Example for Ollama (make sure Ollama is running and model is pulled, e.g., 'llama2')
# llm = Ollama(model="llama2")

# Example for OpenAI (uncomment and replace with your API key)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY") # Set your API key in .env or directly
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model_name="gpt-4-turbo-preview", temperature=0.7)

# Prompt for Problem Decomposition
PROBLEM_DECOMPOSITION_PROMPT = PromptTemplate.from_template(
    """
    You are a medical diagnostic assistant. Your task is to break down a patient's medical case into smaller, manageable diagnostic sub-problems. 
    Consider the patient's symptoms, medical history, and lab results.

    Patient Data:
    Symptoms: {symptoms}
    Medical History: {medical_history}
    Lab Results: {lab_results}
    Age: {age}
    Gender: {gender}

    Decompose the overall diagnostic challenge into 3-5 distinct sub-problems. Each sub-problem should focus on a specific aspect of the patient's condition.
    Output each sub-problem as a concise question or statement.

    Example Output:
    1. Evaluate potential infectious agents.
    2. Assess cardiovascular risk factors.
    3. Investigate metabolic abnormalities.
    """
)

# Prompt for Chain of Thought Diagnosis
COT_DIAGNOSIS_PROMPT = PromptTemplate.from_template(
    """
    You are an expert medical diagnostician. Given the patient's data and a specific sub-problem, provide a detailed Chain-of-Thought reasoning process.
    Your reasoning should be step-by-step, explicit, and transparent. Identify potential diagnoses, rule out others, and explain your confidence.

    Patient Data:
    Symptoms: {symptoms}
    Medical History: {medical_history}
    Lab Results: {lab_results}
    Age: {age}
    Gender: {gender}

    Sub-Problem: {sub_problem}

    Please structure your output as follows, explicitly detailing your thought process:

    Thought Process Steps:
    1. **Observation/Fact:** [State a relevant piece of patient data or general medical fact.]
    2. **Hypothesis Generation:** [Propose a hypothesis based on the observation.]
    3. **Reasoning/Analysis:** [Explain how the observation supports or contradicts the hypothesis, considering other factors.]
    4. **Information Needed/Action:** [What further information or verification is required?]
    ...
    N. **Intermediate Conclusion:** [Summarize the conclusion for this sub-problem.]

    Final Diagnosis for Sub-Problem: [Concise diagnosis]
    Confidence (0.0-1.0): [Your confidence score]
    """
)

# Prompt for Self-Correction and Verification
SELF_CORRECTION_PROMPT = PromptTemplate.from_template(
    """
    You have made an initial diagnosis and reasoning. Your task now is to critically evaluate your own reasoning for consistency, accuracy, and completeness.
    Consider the patient's original data and any external medical knowledge provided. Identify any potential flaws, contradictions, or areas of uncertainty.

    Patient Data:
    Symptoms: {symptoms}
    Medical History: {medical_history}
    Lab Results: {lab_results}

    Initial Diagnosis: {initial_diagnosis_disease}
    Initial Reasoning Summary: {initial_reasoning_summary}
    Knowledge Base Verification Results (if any): {kb_verification_results}

    Instructions:
    1. **Critique:** Point out any inconsistencies between the diagnosis, symptoms, lab results, or knowledge base verification.
    2. **Reverse Reasoning (Optional):** If the diagnosis {initial_diagnosis_disease} is true, what other symptoms or lab results would typically be expected or *not* expected? Are these present in the patient data?
    3. **Refinement/Correction:** Suggest any necessary adjustments to the diagnosis or highlight areas requiring further investigation.
    4. **Confidence Adjustment:** If your confidence has changed based on this self-correction, state the new confidence.

    Provide your self-correction analysis in a detailed, step-by-step manner.
    """
)

# Prompt for Treatment Recommendation
TREATMENT_RECOMMENDATION_PROMPT = PromptTemplate.from_template(
    """
    Based on the confirmed diagnosis and patient data, formulate a comprehensive and personalized treatment plan.
    Consider medications, procedures, lifestyle changes, and follow-up recommendations.
    Provide a clear rationale for each part of the treatment plan.

    Patient Data:
    Symptoms: {symptoms}
    Medical History: {medical_history}
    Lab Results: {lab_results}
    Age: {age}
    Gender: {gender}

    Confirmed Diagnosis: {diagnosis_disease}
    Diagnosis Reasoning Summary: {diagnosis_reasoning_summary}
    Confidence: {diagnosis_confidence}

    Please structure your treatment plan as follows:

    Treatment Plan for {diagnosis_disease}:
    1. **Medications:** [List specific medications, dosages if applicable, and rationale.]
    2. **Procedures:** [List any necessary procedures, therapies, or interventions and rationale.]
    3. **Lifestyle Changes:** [Recommend diet, exercise, stress management, etc., with rationale.]
    4. **Follow-up:** [Specify when and how the patient should be monitored or reviewed.]
    5. **Overall Rationale:** [Summarize the holistic reasoning behind this treatment plan.]
    """
)


# --- 2. Reasoning Engine (`MedicalReasoner`) ---

class MedicalReasoner:
    """Orchestrates the LLM-based diagnostic and treatment recommendation process."""

    def __init__(self, llm: Any, knowledge_base: MedicalKnowledgeBase):
        self.llm = llm
        self.knowledge_base = knowledge_base

        self.decomposition_chain = LLMChain(prompt=PROBLEM_DECOMPOSITION_PROMPT, llm=self.llm, verbose=True)
        self.cot_diagnosis_chain = LLMChain(prompt=COT_DIAGNOSIS_PROMPT, llm=self.llm, verbose=True)
        self.self_correction_chain = LLMChain(prompt=SELF_CORRECTION_PROMPT, llm=self.llm, verbose=True)
        self.treatment_chain = LLMChain(prompt=TREATMENT_RECOMMENDATION_PROMPT, llm=self.llm, verbose=True)

    def _parse_cot_output(self, llm_output: str) -> Dict[str, Any]:
        """Parses the LLM's Chain of Thought output into structured data."""
        logger.debug(f"Parsing CoT output:\n{llm_output}")
        diagnosis_lines = [line for line in llm_output.split('\n') if line.startswith('Final Diagnosis for Sub-Problem:') or line.startswith('Confidence (0.0-1.0):')]
        
        disease = "Unknown"
        confidence = 0.0
        if diagnosis_lines:
            if len(diagnosis_lines) > 0 and 'Final Diagnosis for Sub-Problem:' in diagnosis_lines[0]:
                disease = diagnosis_lines[0].split(':', 1)[1].strip()
            if len(diagnosis_lines) > 1 and 'Confidence (0.0-1.0):' in diagnosis_lines[1]:
                try:
                    confidence = float(diagnosis_lines[1].split(':', 1)[1].strip())
                except ValueError:
                    logger.warning(f"Could not parse confidence from: {diagnosis_lines[1]}")

        # Extract intermediate steps (simplified parsing, can be improved with regex/more robust logic)
        intermediate_thoughts_raw = []
        in_thought_process = False
        for line in llm_output.split('\n'):
            if "Thought Process Steps:" in line:
                in_thought_process = True
                continue
            if in_thought_process and line.strip() and not line.startswith("Final Diagnosis for Sub-Problem:"):
                intermediate_thoughts_raw.append(line.strip())
            elif line.startswith("Final Diagnosis for Sub-Problem:"):
                in_thought_process = False

        cot_steps: List[CoTStep] = []
        current_step = 1
        current_thought_parts = []

        for raw_step in intermediate_thoughts_raw:
            if raw_step.startswith(f"{current_step}. ") or raw_step.startswith(f"{current_step}.**"):
                if current_thought_parts:
                    # Heuristic to determine action/observation - needs robust parsing for real app
                    thought_text = " ".join(current_thought_parts)
                    action_match = next((part for part in current_thought_parts if "Action:" in part or "Information Needed:" in part), None)
                    observation_match = next((part for part in current_thought_parts if "Observation:" in part or "Result:" in part), None)

                    cot_steps.append(CoTStep(
                        step_number=current_step - 1,
                        thought=thought_text.replace(f"Thought Process Steps:{current_step-1}. ", '').replace(f"{current_step-1}. ", ''),
                        action=action_match.replace('Action:', '').replace('Information Needed:', '').strip() if action_match else "",
                        observation=observation_match.replace('Observation:', '').replace('Result:', '').strip() if observation_match else ""
                    ))
                current_thought_parts = [raw_step]
                current_step += 1
            elif current_thought_parts: # Continue current step's thought
                current_thought_parts.append(raw_step)
        
        if current_thought_parts: # Add the last step
             thought_text = " ".join(current_thought_parts)
             action_match = next((part for part in current_thought_parts if "Action:" in part or "Information Needed:" in part), None)
             observation_match = next((part for part in current_thought_parts if "Observation:" in part or "Result:" in part), None)
             cot_steps.append(CoTStep(
                step_number=current_step - 1,
                thought=thought_text.replace(f"Thought Process Steps:{current_step-1}. ", '').replace(f"{current_step-1}. ", ''),
                action=action_match.replace('Action:', '').replace('Information Needed:', '').strip() if action_match else "",
                observation=observation_match.replace('Observation:', '').replace('Result:', '').strip() if observation_match else ""
            ))


        return {
            "disease": disease,
            "confidence": confidence,
            "reasoning_summary": " ".join(intermediate_thoughts_raw), # Simple summary for now
            "intermediate_thoughts": cot_steps
        }

    def diagnose_patient(self, patient_data: PatientData) -> List[Diagnosis]:
        logger.info(f"Starting diagnosis for patient: {patient_data.patient_id}")

        # Problem Decomposition
        decomposition_output = self.decomposition_chain.run(
            symptoms=", ".join(patient_data.symptoms),
            medical_history=", ".join(patient_data.medical_history),
            lab_results=str(patient_data.lab_results),
            age=patient_data.age,
            gender=patient_data.gender
        )
        sub_problems = [p.strip() for p in decomposition_output.split('\n') if p.strip() and not p.startswith('Example Output:')]
        logger.info(f"Decomposed into sub-problems: {sub_problems}")

        all_diagnoses: List[Diagnosis] = []

        for sub_problem in sub_problems:
            logger.info(f"Addressing sub-problem: {sub_problem}")

            # Chain-of-Thought Diagnosis
            cot_output = self.cot_diagnosis_chain.run(
                symptoms=", ".join(patient_data.symptoms),
                medical_history=", ".join(patient_data.medical_history),
                lab_results=str(patient_data.lab_results),
                age=patient_data.age,
                gender=patient_data.gender,
                sub_problem=sub_problem
            )
            initial_diagnosis_data = self._parse_cot_output(cot_output)
            initial_diagnosis = Diagnosis(
                disease=initial_diagnosis_data['disease'],
                confidence=initial_diagnosis_data['confidence'],
                reasoning_summary=initial_diagnosis_data['reasoning_summary'],
                intermediate_thoughts=initial_diagnosis_data['intermediate_thoughts']
            )
            logger.info(f"Initial diagnosis for '{sub_problem}': {initial_diagnosis.disease} (Confidence: {initial_diagnosis.confidence:.2f})")

            # Self-Correction & Verification
            kb_verification_results = {
                "symptom_consistency": self.knowledge_base.verify_symptom_diagnosis_consistency(patient_data.symptoms, initial_diagnosis.disease),
                "lab_result_consistency": self.knowledge_base.verify_lab_results_diagnosis_consistency(patient_data.lab_results, initial_diagnosis.disease)
            }
            logger.info(f"Knowledge Base Verification: {kb_verification_results}")

            self_correction_output = self.self_correction_chain.run(
                symptoms=", ".join(patient_data.symptoms),
                medical_history=", ".join(patient_data.medical_history),
                lab_results=str(patient_data.lab_results),
                initial_diagnosis_disease=initial_diagnosis.disease,
                initial_reasoning_summary=initial_diagnosis.reasoning_summary,
                kb_verification_results=str(kb_verification_results)
            )
            logger.info(f"Self-correction analysis for {initial_diagnosis.disease}:\n{self_correction_output}")

            # For simplicity, we'll assume the self-correction either confirms or suggests a revised diagnosis.
            # A more advanced system would parse the self-correction output to update the diagnosis.
            # Here, we just store the initial diagnosis with the self-correction context.
            all_diagnoses.append(initial_diagnosis)

        # Robust Aggregation (Simulated: For a real system, this would involve weighting, voting, or specialized models)
        # For demonstration, we'll just return all individual diagnoses for now.
        # A real system would consolidate these into a primary diagnosis with supporting alternatives.
        logger.info("Aggregating diagnoses (simulated - returning all for now).")
        return all_diagnoses

    def recommend_treatment(self, patient_data: PatientData, diagnosis: Diagnosis) -> TreatmentPlan:
        logger.info(f"Generating treatment plan for {diagnosis.disease} for patient {patient_data.patient_id}")
        treatment_output = self.treatment_chain.run(
            symptoms=", ".join(patient_data.symptoms),
            medical_history=", ".join(patient_data.medical_history),
            lab_results=str(patient_data.lab_results),
            age=patient_data.age,
            gender=patient_data.gender,
            diagnosis_disease=diagnosis.disease,
            diagnosis_reasoning_summary=diagnosis.reasoning_summary,
            diagnosis_confidence=diagnosis.confidence
        )
        logger.info(f"Generated treatment plan output:\n{treatment_output}")

        # Simple parsing of treatment plan output (needs more robust parsing for production)
        medications = []
        procedures = []
        lifestyle_changes = []
        follow_up = ""
        rationale = ""

        # This parsing is highly dependent on the LLM's output format.
        # Pydantic `OutputParser` from langchain could be used for more robust parsing if the LLM output is JSON.
        lines = treatment_output.split('\n')
        current_section = None
        for line in lines:
            if line.startswith('1. **Medications**:'):
                current_section = medications
            elif line.startswith('2. **Procedures**:'):
                current_section = procedures
            elif line.startswith('3. **Lifestyle Changes**:'):
                current_section = lifestyle_changes
            elif line.startswith('4. **Follow-up**:'):
                current_section = 'follow_up'
            elif line.startswith('5. **Overall Rationale**:'):
                current_section = 'rationale'
            elif current_section is not None and line.strip():
                if isinstance(current_section, list):
                    clean_line = line.strip().lstrip('- ').lstrip('* ').replace('[', '').replace(']', '').replace('Rationale:', '').strip()
                    if clean_line: # Avoid adding empty strings
                        current_section.append(clean_line)
                elif current_section == 'follow_up':
                    follow_up += line.strip() + " "
                elif current_section == 'rationale':
                    rationale += line.strip() + " "
        
        return TreatmentPlan(
            diagnosis_id=diagnosis.disease.replace(" ", "-").lower(), # Simple ID generation
            medications=medications,
            procedures=procedures,
            lifestyle_changes=lifestyle_changes,
            follow_up_recommendations=follow_up.strip(),
            rationale=rationale.strip()
        )


# --- Main Application Logic (Example Usage) ---

if __name__ == "__main__":
    logger.add("medical_system.log", rotation="10 MB")
    logger.info("Medical Diagnosis and Treatment Recommendation System Started.")

    knowledge_base = MedicalKnowledgeBase()
    reasoner = MedicalReasoner(llm=llm, knowledge_base=knowledge_base)

    # Example Patient Data 1: Diabetes-like symptoms
    patient1_data = PatientData(
        patient_id="P001",
        symptoms=["frequent urination", "increased thirst", "fatigue", "blurred vision", "unexplained weight loss"],
        medical_history=["family history of diabetes"],
        lab_results={
            "glucose_level": 250, # mg/dL
            "hba1c": 8.5,
            "blood_pressure": "130/85"
        },
        age=55,
        gender="Male"
    )

    # Example Patient Data 2: Hypertension-like symptoms
    patient2_data = PatientData(
        patient_id="P002",
        symptoms=["occasional headaches", "dizziness", "shortness of breath on exertion"],
        medical_history=["smoker", "sedentary lifestyle"],
        lab_results={
            "blood_pressure": "150/95",
            "cholesterol_ldl": 160 # mg/dL
        },
        age=62,
        gender="Female"
    )

    # Process Patient 1
    logger.info("\n--- Processing Patient P001 ---")
    diagnoses1 = reasoner.diagnose_patient(patient1_data)
    for diag in diagnoses1:
        logger.info(f"\nFinal Diagnosis for P001: {diag.disease} (Confidence: {diag.confidence:.2f})")
        logger.info(f"Reasoning Summary: {diag.reasoning_summary}")
        # logger.info("Intermediate Thoughts:")
        # for step in diag.intermediate_thoughts:
        #     logger.info(f"  Step {step.step_number}: {step.thought}")
        
        treatment_plan1 = reasoner.recommend_treatment(patient1_data, diag)
        logger.info(f"Treatment Plan for P001 (Diagnosis: {diag.disease}):")
        logger.info(f"  Medications: {treatment_plan1.medications}")
        logger.info(f"  Procedures: {treatment_plan1.procedures}")
        logger.info(f"  Lifestyle Changes: {treatment_plan1.lifestyle_changes}")
        logger.info(f"  Follow-up: {treatment_plan1.follow_up_recommendations}")
        logger.info(f"  Rationale: {treatment_plan1.rationale}")

    # Process Patient 2
    logger.info("\n--- Processing Patient P002 ---")
    diagnoses2 = reasoner.diagnose_patient(patient2_data)
    for diag in diagnoses2:
        logger.info(f"\nFinal Diagnosis for P002: {diag.disease} (Confidence: {diag.confidence:.2f})")
        logger.info(f"Reasoning Summary: {diag.reasoning_summary}")
        
        treatment_plan2 = reasoner.recommend_treatment(patient2_data, diag)
        logger.info(f"Treatment Plan for P002 (Diagnosis: {diag.disease}):")
        logger.info(f"  Medications: {treatment_plan2.medications}")
        logger.info(f"  Procedures: {treatment_plan2.procedures}")
        logger.info(f"  Lifestyle Changes: {treatment_plan2.lifestyle_changes}")
        logger.info(f"  Follow-up: {treatment_plan2.follow_up_recommendations}")
        logger.info(f"  Rationale: {treatment_plan2.rationale}")

    logger.info("Medical Diagnosis and Treatment Recommendation System Finished.")
