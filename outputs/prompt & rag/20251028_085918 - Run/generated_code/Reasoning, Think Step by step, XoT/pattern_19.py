import pandas as pd
import spacy
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import streamlit as st

# --- 1. Pydantic Models for Data Structures ---

class PatientData(BaseModel):
    """Represents the patient's comprehensive medical information."""
    medical_history: str
    symptoms: List[str]
    lab_results: Dict[str, Any]
    genetic_data: Optional[str] = None # Simplified; could be a more complex model

class ReasoningStep(BaseModel):
    """Represents a single step in the LLM's reasoning process."""
    step_number: int
    thought: str
    intermediate_result: Optional[str] = None
    confidence: Optional[float] = None

class DiagnosisProposal(BaseModel):
    """Represents a proposed diagnosis with reasoning."""
    disease_name: str
    probability: float
    reasoning_path: List[ReasoningStep]
    verified_externally: bool = False
    self_corrected: bool = False

class TreatmentPlan(BaseModel):
    """Represents a proposed treatment plan."""
    diagnosis: str
    treatment_recommendations: List[str]
    dosage_instructions: List[str]
    monitoring_guidelines: List[str]
    reasoning_path: List[ReasoningStep]

# --- 2. Mock LLM Class (Simulates LLM behavior) ---

class MockLLM:
    """A mock LLM to simulate responses for demonstration purposes."""
    def __init__(self, model_name: str = "MockGPT-4"):
        self.model_name = model_name

    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        """Simulates LLM text generation based on the prompt content."""
        if "decompose the diagnostic task" in prompt.lower():
            return (
                "Step 1: Analyze patient symptoms and history. "
                "Step 2: Interpret lab results. "
                "Step 3: Identify potential genetic markers. "
                "Step 4: Cross-reference with rare disease profiles to form differential diagnoses."
            )
        elif "explain why certain possibilities are considered" in prompt.lower():
            return (
                f"Thought Process for Step {prompt.split('Step ')[1].split(':')[0].strip()}:\n"
                "Initial assessment based on fever and fatigue suggests inflammatory or infectious process. "
                "Considering rare diseases, we evaluate autoimmune conditions due to chronic nature."
            )
        elif "evaluate its own reasoning for logical consistency" in prompt.lower():
            if "rare genetic disorder X" in prompt: # Simulate a correct verification
                return "Self-verification passed: The reasoning for 'rare genetic disorder X' is logically consistent with provided symptoms and lab results."
            else: # Simulate a potential inconsistency
                return "Self-verification initiated: While 'common cold' was considered, it doesn't align with the chronic fatigue and specific lab marker. Re-evaluating."
        elif "query an external medical knowledge graph" in prompt.lower():
            if "rare genetic disorder X" in prompt:
                return "External knowledge base confirms 'rare genetic disorder X' symptoms, genetic markers (e.g., gene ABC1 mutation), and common lab findings match patient profile. Confidence increased."
            return "External knowledge base query for 'hypothetical disease Y' returned no definitive matches. Further investigation needed or consider alternative diagnoses."
        elif "aggregate findings" in prompt.lower():
            return "Aggregated diagnosis: Primary differential diagnosis is 'Rare Genetic Disorder X' (probability 85%), followed by 'Autoimmune Condition Y' (probability 10%)."
        elif "propose personalized treatment plans" in prompt.lower():
            return (
                "Treatment Plan for Rare Genetic Disorder X:\n"
                "- Medication A (5mg daily)\n"
                "- Supportive therapy (Physical Therapy 3x/week)\n"
                "- Monitor blood markers quarterly. "
                "Reasoning: Medication A targets the underlying pathway, supportive therapy manages symptoms, and monitoring ensures safety and efficacy."
            )
        return "LLM Mock Response: This is a generic response for a non-specific prompt."

# --- 3. Data Ingestion and Preprocessing Layer ---

# Load a simple spaCy model for NER (requires 'python -m spacy download en_core_web_sm')
# Mocking for environment where spaCy model might not be available or downloadable directly
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("SpaCy 'en_core_web_sm' model not found. Mocking NLP processing.")
    class MockNlp:
        def __call__(self, text):
            class MockDoc:
                ents = []
                def __init__(self, text):
                    # Simple mock for symptoms/entities
                    if "fever" in text.lower():
                        self.ents.append(MockEnt("fever", "SYMPTOM"))
                    if "fatigue" in text.lower():
                        self.ents.append(MockEnt("fatigue", "SYMPTOM"))
                    if "chest pain" in text.lower():
                        self.ents.append(MockEnt("chest pain", "SYMPTOM"))
            return MockDoc(text)

    class MockEnt:
        def __init__(self, text, label):
            self.text = text
            self.label_ = label
    nlp = MockNlp()


def parse_patient_data(
    medical_history_text: str,
    symptoms_text: str,
    lab_results_json: str,
    genetic_data_text: Optional[str] = None
) -> PatientData:
    """
    Parses raw patient data into a structured PatientData object.
    Uses NLP for symptom extraction from text.
    """
    # Use spaCy to extract symptoms/entities
    doc = nlp(medical_history_text + " " + symptoms_text)
    extracted_symptoms = [ent.text for ent in doc.ents if ent.label_ == "SYMPTOM"]
    if not extracted_symptoms and symptoms_text: # Fallback if spaCy doesn't find, just take as list
        extracted_symptoms = [s.strip() for s in symptoms_text.split(',') if s.strip()]


    try:
        lab_results = json.loads(lab_results_json)
    except json.JSONDecodeError:
        lab_results = {"error": "Invalid JSON for lab results"}

    return PatientData(
        medical_history=medical_history_text,
        symptoms=list(set(extracted_symptoms)), # Remove duplicates
        lab_results=lab_results,
        genetic_data=genetic_data_text
    )

# --- 4. LLM Orchestration Layer (Simplified LangChain/LlamaIndex concept) ---

class DiagnosisAssistant:
    def __init__(self, llm_model: MockLLM):
        self.llm = llm_model
        self.reasoning_history: List[ReasoningStep] = []

    def _generate_prompt(self, task_description: str, context: Dict[str, Any]) -> str:
        """Dynamically generates prompts for the LLM."""
        prompt = f"Given the following context, {task_description}:\n\n"
        for key, value in context.items():
            prompt += f"{key}: {value}\n"
        return prompt

    def _decompose_task(self, patient_data: PatientData) -> List[str]:
        """Decomposes the diagnostic task into sub-steps using LLM."""
        prompt = self._generate_prompt(
            "decompose the diagnostic task for a patient with the following data, outlining steps like symptom analysis, lab interpretation, genetic marker identification, and cross-referencing with rare diseases",
            {"Patient Summary": patient_data.model_dump_json(indent=2)}
        )
        response = self.llm.generate(prompt)
        steps = [s.strip() for s in response.split("Step ") if s.strip()]
        return steps

    def _generate_cot_reasoning(self, step: str, patient_data: PatientData, current_findings: List[str]) -> ReasoningStep:
        """Generates Chain-of-Thought reasoning for a given step."""
        context = {
            "Patient Data": patient_data.model_dump_json(indent=2),
            "Current Findings": current_findings,
            "Current Step": step
        }
        prompt = self._generate_prompt(
            f"Explain why certain possibilities are considered or discarded for '{step}', and how different pieces of information contribute to the evolving diagnosis. Generate a detailed thought process.",
            context
        )
        response = self.llm.generate(prompt)
        reasoning_step = ReasoningStep(
            step_number=len(self.reasoning_history) + 1,
            thought=response,
            intermediate_result=f"Processing '{step}'."
        )
        self.reasoning_history.append(reasoning_step)
        return reasoning_step

    def _self_verify(self, proposed_diagnosis: str, patient_data: PatientData, reasoning_path: List[ReasoningStep]) -> bool:
        """LLM performs self-correction/verification."""
        context = {
            "Proposed Diagnosis": proposed_diagnosis,
            "Patient Data": patient_data.model_dump_json(indent=2),
            "Reasoning Path": [r.model_dump() for r in reasoning_path]
        }
        prompt = self._generate_prompt(
            f"Critically evaluate its own reasoning for logical consistency and faithfulness to input data for the proposed diagnosis '{proposed_diagnosis}'. Perform a 'reverse check' to see if all patient data aligns.",
            context
        )
        response = self.llm.generate(prompt)
        self.reasoning_history.append(ReasoningStep(
            step_number=len(self.reasoning_history) + 1,
            thought=f"Self-verification initiated for '{proposed_diagnosis}': {response}",
            intermediate_result=f"Verification of {proposed_diagnosis}"
        ))
        return "self-verification passed" in response.lower() or "confidence increased" in response.lower()

    def _external_validate(self, proposed_diagnosis: str) -> bool:
        """Queries an external 'medical knowledge graph' (mocked)."""
        context = {"Proposed Diagnosis": proposed_diagnosis}
        prompt = self._generate_prompt(
            f"Query an external medical knowledge graph or specialized expert system to validate less common diagnostic paths for '{proposed_diagnosis}'.",
            context
        )
        response = self.llm.generate(prompt)
        self.reasoning_history.append(ReasoningStep(
            step_number=len(self.reasoning_history) + 1,
            thought=f"External validation for '{proposed_diagnosis}': {response}",
            intermediate_result=f"External validation of {proposed_diagnosis}"
        ))
        return "confirms" in response.lower() # Simple check for success

    def _ensemble_reasoning(self, candidate_diagnoses: List[DiagnosisProposal]) -> DiagnosisProposal:
        """Aggregates findings from multiple reasoning paths (mocked as selecting the highest probability)."""
        if not candidate_diagnoses:
            raise ValueError("No candidate diagnoses to ensemble.")

        # In a real scenario, this would involve more complex aggregation
        # For this mock, we'll just pick the one with highest probability or a predefined one.
        best_diagnosis = max(candidate_diagnoses, key=lambda x: x.probability)

        self.reasoning_history.append(ReasoningStep(
            step_number=len(self.reasoning_history) + 1,
            thought=f"Ensembling results from {len(candidate_diagnoses)} candidate diagnoses. "
                    f"Selected '{best_diagnosis.disease_name}' as the primary differential.",
            intermediate_result=f"Ensembled Diagnosis: {best_diagnosis.disease_name}"
        ))
        return best_diagnosis

    def _generate_treatment_plan(self, diagnosis: DiagnosisProposal, patient_data: PatientData) -> TreatmentPlan:
        """Generates a personalized treatment plan."""
        context = {
            "Diagnosis": diagnosis.disease_name,
            "Patient Data": patient_data.model_dump_json(indent=2),
            "Diagnostic Reasoning": [r.model_dump() for r in diagnosis.reasoning_path]
        }
        prompt = self._generate_prompt(
            f"Propose personalized treatment plans for '{diagnosis.disease_name}', considering patient comorbidities, drug interactions, and latest therapeutic guidelines.",
            context
        )
        response = self.llm.generate(prompt)

        # Parse the mock response into structured treatment plan
        recommendations = []
        dosage = []
        monitoring = []
        for line in response.split('\n'):
            if "- Medication" in line:
                recommendations.append(line.strip())
            elif "- Supportive therapy" in line:
                recommendations.append(line.strip())
            elif "- Monitor" in line:
                monitoring.append(line.strip())

        treatment_reasoning_step = ReasoningStep(
            step_number=len(self.reasoning_history) + 1,
            thought=response,
            intermediate_result=f"Treatment plan for {diagnosis.disease_name}"
        )
        self.reasoning_history.append(treatment_reasoning_step)

        return TreatmentPlan(
            diagnosis=diagnosis.disease_name,
            treatment_recommendations=recommendations,
            dosage_instructions=dosage, # Mock LLM doesn't parse specific dosages well, so keep generic
            monitoring_guidelines=monitoring,
            reasoning_path=[treatment_reasoning_step]
        )

    def diagnose_and_plan(self, patient_data: PatientData) -> Dict[str, Any]:
        """
        Orchestrates the entire diagnosis and treatment planning process.
        """
        self.reasoning_history = []
        st.subheader("Starting Diagnosis Process...")

        # 1. Decompose Task
        st.write("Step 1: Decomposing the diagnostic task...")
        task_steps = self._decompose_task(patient_data)
        st.write(f"Task Decomposition: {', '.join(task_steps)}")
        current_findings = []

        candidate_diagnoses: List[DiagnosisProposal] = []

        # Simulate a primary diagnostic path
        st.write("Step 2: Generating Chain-of-Thought reasoning for primary path...")
        for i, step in enumerate(task_steps):
            st.markdown(f"**Current Step:** `{step}`")
            reasoning = self._generate_cot_reasoning(step, patient_data, current_findings)
            st.write(f"  - Thought: {reasoning.thought[:150]}...") # Truncate for display
            current_findings.append(reasoning.intermediate_result)
            self.reasoning_history.append(reasoning)

        # Simulate a proposed diagnosis after initial reasoning
        # For demonstration, let's assume one path leads to "Rare Genetic Disorder X"
        proposed_diagnosis_name = "Rare Genetic Disorder X" # Hardcode for mock
        initial_diagnosis_path = self.reasoning_history[:] # Capture reasoning up to this point
        initial_proposal = DiagnosisProposal(
            disease_name=proposed_diagnosis_name,
            probability=0.7, # Initial probability
            reasoning_path=initial_diagnosis_path
        )
        candidate_diagnoses.append(initial_proposal)

        # 3. Self-Verification
        st.write(f"Step 3: Initiating self-verification for '{proposed_diagnosis_name}'...")
        is_self_verified = self._self_verify(proposed_diagnosis_name, patient_data, initial_diagnosis_path)
        st.write(f"  - Self-verification result: {'Passed' if is_self_verified else 'Failed'}")
        if is_self_verified:
            candidate_diagnoses[-1].self_corrected = True
            candidate_diagnoses[-1].probability += 0.1 # Increase confidence

        # 4. External Validation
        st.write(f"Step 4: Querying external knowledge for '{proposed_diagnosis_name}'...")
        is_externally_validated = self._external_validate(proposed_diagnosis_name)
        st.write(f"  - External validation result: {'Confirmed' if is_externally_validated else 'Not confirmed'}")
        if is_externally_validated:
            candidate_diagnoses[-1].verified_externally = True
            candidate_diagnoses[-1].probability += 0.1 # Increase confidence further

        # 5. Simulate another reasoning path leading to a different diagnosis
        st.write("Step 5: Simulating an alternative reasoning path (e.g., for 'Autoimmune Condition Y')...")
        alt_reasoning_steps = [
            ReasoningStep(step_number=len(self.reasoning_history) + 1, thought="Considering chronic fatigue and specific blood markers, autoimmune conditions are possible.", intermediate_result="Autoimmune pathway considered."),
            ReasoningStep(step_number=len(self.reasoning_history) + 2, thought="Lack of strong genetic markers for 'Rare Genetic Disorder X' in some variants, pivoting to autoimmune.", intermediate_result="Autoimmune strength increases.")
        ]
        self.reasoning_history.extend(alt_reasoning_steps)
        alt_diagnosis_proposal = DiagnosisProposal(
            disease_name="Autoimmune Condition Y",
            probability=0.4,
            reasoning_path=alt_reasoning_steps
        )
        candidate_diagnoses.append(alt_diagnosis_proposal)

        # 6. Ensemble Reasoning
        st.write("Step 6: Ensembling all candidate diagnoses...")
        final_diagnosis = self._ensemble_reasoning(candidate_diagnoses)
        st.write(f"Final Differential Diagnosis: **{final_diagnosis.disease_name}** (Probability: {final_diagnosis.probability:.2f})")

        # 7. Treatment Planning
        st.write("Step 7: Generating personalized treatment plan...")
        treatment_plan = self._generate_treatment_plan(final_diagnosis, patient_data)
        st.write("Treatment Plan Generated.")

        return {
            "final_diagnosis": final_diagnosis.model_dump(),
            "treatment_plan": treatment_plan.model_dump(),
            "full_reasoning_history": [r.model_dump() for r in self.reasoning_history]
        }

# --- Streamlit UI ---

def main():
    st.set_page_config(page_title="Rare Disease AI Assistant", layout="wide")
    st.title("AI-powered Differential Diagnosis and Treatment Planning Assistant for Rare Diseases")
    st.markdown("This application assists medical professionals by leveraging LLMs for complex, multi-step reasoning, self-correction, and external validation in diagnosing rare diseases and formulating treatment plans.")

    st.sidebar.header("Patient Information Input")

    medical_history_input = st.sidebar.text_area(
        "Medical History:",
        "Patient presents with chronic fatigue, intermittent fever for 6 months, and unexplained weight loss. Family history includes autoimmune disorders. No significant travel history. Prior treatments for fatigue were ineffective.",
        height=150
    )
    symptoms_input = st.sidebar.text_input(
        "Symptoms (comma-separated):",
        "fever, fatigue, joint pain, muscle weakness, skin rash"
    )
    lab_results_input = st.sidebar.text_area(
        "Lab Results (JSON format):",
        """
{
  "CRP": {"value": 15, "unit": "mg/L", "ref_range": "0-5"},
  "ESR": {"value": 80, "unit": "mm/hr", "ref_range": "0-20"},
  "ANA_Titer": {"value": "1:640", "pattern": "speckled"},
  "Genetic_Panel": {"gene_ABC1_mutation": "present", "gene_XYZ2_mutation": "absent"}
}
        """,
        height=200
    )
    genetic_data_input = st.sidebar.text_area(
        "Genetic Data (e.g., specific gene mutations or findings):",
        "Genetic sequencing shows a suspected pathogenic variant in the ABC1 gene, associated with 'Rare Genetic Disorder X'.",
        height=100
    )

    if st.sidebar.button("Run Diagnosis and Treatment Plan"):
        if not medical_history_input or not symptoms_input or not lab_results_input:
            st.error("Please fill in all required patient information: Medical History, Symptoms, and Lab Results.")
            return

        try:
            patient_data = parse_patient_data(
                medical_history_text=medical_history_input,
                symptoms_text=symptoms_input,
                lab_results_json=lab_results_input,
                genetic_data_text=genetic_data_input
            )

            st.write("### Patient Data Processed:")
            st.json(patient_data.model_dump_json(indent=2))
            st.divider()

            mock_llm = MockLLM()
            assistant = DiagnosisAssistant(mock_llm)
            results = assistant.diagnose_and_plan(patient_data)

            st.divider()
            st.header("Diagnosis Results")

            final_diag = results["final_diagnosis"]
            st.subheader(f"Primary Differential Diagnosis: {final_diag['disease_name']}")
            st.metric("Probability", f"{final_diag['probability']:.2f}")
            st.write(f"Self-verified: {final_diag['self_corrected']}")
            st.write(f"Externally validated: {final_diag['verified_externally']}")

            st.subheader("Proposed Treatment Plan")
            treatment_plan = results["treatment_plan"]
            st.write("#### Recommendations:")
            for rec in treatment_plan['treatment_recommendations']:
                st.write(f"- {rec}")
            if treatment_plan['dosage_instructions']:
                st.write("#### Dosage Instructions:")
                for dos in treatment_plan['dosage_instructions']:
                    st.write(f"- {dos}")
            if treatment_plan['monitoring_guidelines']:
                st.write("#### Monitoring Guidelines:")
                for mon in treatment_plan['monitoring_guidelines']:
                    st.write(f"- {mon}")

            st.header("Full Reasoning Trace (Chain of Thought)")
            with st.expander("View Detailed Reasoning"):
                for step in results["full_reasoning_history"]:
                    st.markdown(f"**Step {step['step_number']}**: {step['intermediate_result']}")
                    st.write(f"  *Thought*: {step['thought']}")
                    st.write("---")

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.exception(e)

if __name__ == "__main__":
    main()