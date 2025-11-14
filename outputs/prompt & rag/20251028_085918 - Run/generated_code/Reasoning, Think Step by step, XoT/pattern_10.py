
import streamlit as st
from pydantic import BaseModel
from typing import List, Optional
import os

# Mocking external libraries for demonstration purposes if not fully installed/configured
# In a real application, you would import and use these directly.

# Mock for OpenAI (or any LLM client)
class MockLLM:
    def __init__(self, model_name: str = "gpt-4-mock"):
        self.model_name = model_name

    def invoke(self, prompt: str) -> str:
        """Simulates an LLM call with Chain-of-Thought and self-correction logic."""
        st.sidebar.subheader("Mock LLM Interaction")
        st.sidebar.text(f"Prompting LLM ({self.model_name}):")
        st.sidebar.code(prompt[:200] + "..." if len(prompt) > 200 else prompt)

        # --- Simplified Chain-of-Thought Simulation ---
        if "Chain-of-Thought:" in prompt or "Reasoning Steps:" in prompt:
            if "diagnose" in prompt.lower() and "patient" in prompt.lower():
                response = (
                    "Chain-of-Thought: The patient presents with symptoms A, B, and C. "
                    "Considering these, I will first rule out common conditions, then consider rarer ones. "
                    "Step 1: Analyze symptoms. Symptom A suggests X. Symptom B suggests Y. Symptom C suggests Z. "
                    "Step 2: Cross-reference with medical history. "
                    "Step 3: Propose differential diagnoses based on the strongest evidence. "
                    "Initial Diagnosis: Possible Condition P, consider Condition Q, rule out Condition R. "
                    "Suggested Investigations: Blood test for P markers, imaging for Q, specific symptom diary."
                )
            elif "review and refine" in prompt.lower():
                response = (
                    "Self-Correction: Upon review, the initial diagnosis for Condition P was based on symptom A, "
                    "but symptom C could also indicate Condition S. I will refine the reasoning to consider S more. "
                    "Revised Diagnosis: Likely Condition P, but Condition S is a strong differential. "
                    "Further Investigation Priority: Differentiate between P and S."
                )
            elif "verify against facts" in prompt.lower():
                response = (
                    "Verification: The proposed diagnosis of Condition P is consistent with medical fact M1 and M2. "
                    "The symptom C-to-S link is also medically sound. "
                    "Final Confirmed Diagnosis: Condition P, with Condition S as a close differential based on evidence."
                )
            else:
                response = "Mock LLM: I am processing your complex medical query using Chain-of-Thought. " \
                           "Initial thought: " + prompt.split("\n")[-1].strip() + "..." \
                           "Generated response based on simulated reasoning."
        else:
            response = f"Mock LLM ({self.model_name}): {prompt}"

        st.sidebar.text("LLM Response:")
        st.sidebar.code(response[:200] + "..." if len(response) > 200 else response)
        return response


# Mock for Langchain (simplified components)
class MockPromptTemplate:
    def __init__(self, template: str, input_variables: List[str]):
        self.template = template
        self.input_variables = input_variables

    def format(self, **kwargs) -> str:
        formatted_prompt = self.template
        for var in self.input_variables:
            formatted_prompt = formatted_prompt.replace(f"{{{var}}}", str(kwargs.get(var, f"[{var} not provided]")))
        return formatted_prompt

class MockRetriever:
    def __init__(self, knowledge_base_data: dict):
        self.knowledge_base_data = knowledge_base_data

    def get_relevant_documents(self, query: str) -> List[str]:
        """Simulates retrieval from a knowledge base."""
        relevant_docs = []
        query_lower = query.lower()
        for fact_key, fact_value in self.knowledge_base_data.items():
            if query_lower in fact_key.lower() or query_lower in fact_value.lower():
                relevant_docs.append(f"Medical Fact: {fact_value}")
        return relevant_docs if relevant_docs else ["No direct medical facts found for this query."]


# Mock for Chroma (in-memory simple key-value store for demonstration)
class MockChromaDB:
    def __init__(self):
        self.db = {}

    def add_documents(self, documents: List[str]):
        for i, doc in enumerate(documents):
            self.db[f"doc_{i}"] = doc

    def as_retriever(self) -> MockRetriever:
        return MockRetriever(self.db)


# Pydantic Data Models
class PatientInput(BaseModel):
    symptoms: str
    medical_history: str
    test_results: Optional[str] = None
    specific_questions: Optional[str] = None

class DiagnosisOutput(BaseModel):
    proposed_diagnoses: List[str]
    reasoning_steps: str
    suggested_investigations: List[str]
    confidence_score: Optional[float] = None
    verified: bool = False


# --- Medical Knowledge Base (RAG System) Setup ---
# In a real application, this would be populated from actual medical texts
medical_knowledge_data = [
    "Symptoms of influenza often include fever, body aches, cough, and fatigue.",
    "Type 2 diabetes is characterized by insulin resistance and high blood sugar levels.",
    "Migraines are severe headaches often accompanied by nausea, vomiting, and sensitivity to light/sound.",
    "Appendicitis typically presents with right lower abdominal pain, nausea, and fever.",
    "The normal range for blood pressure is generally considered to be below 120/80 mmHg.",
    "High cholesterol is a major risk factor for heart disease and stroke.",
    "Pneumonia is an infection that inflames air sacs in one or both lungs, which may fill with fluid."
]

# Initialize Mock Chroma DB and add documents
mock_vector_db = MockChromaDB()
mock_vector_db.add_documents(medical_knowledge_data)
medical_retriever = mock_vector_db.as_retriever()


# --- Core Reasoning Engine (LLM & Orchestration) ---
mock_llm = MockLLM()

# Langchain-style prompt templates
initial_diagnosis_template = """
As a medical diagnostic assistant, analyze the following patient data and propose differential diagnoses. 
Provide your reasoning steps using a Chain-of-Thought approach. 

Patient Symptoms: {symptoms}
Medical History: {medical_history}
Test Results: {test_results}
Specific Questions: {specific_questions}

Medical Context (for reference): {medical_context}

Chain-of-Thought:
1. Deconstruct the primary symptoms.
2. Cross-reference with medical history and test results.
3. Consider relevant medical facts from the context.
4. Propose a list of differential diagnoses with justifications.
5. Suggest further investigations to confirm or rule out diagnoses.

Reasoning Steps:
"""
initial_diagnosis_prompt = MockPromptTemplate(
    template=initial_diagnosis_template,
    input_variables=["symptoms", "medical_history", "test_results", "specific_questions", "medical_context"]
)

self_correction_template = """
Review the following initial diagnosis and reasoning. Identify any potential inconsistencies, gaps, or areas for refinement based on general medical knowledge. If necessary, propose a revised diagnosis and updated reasoning. Be critical and thorough.

Initial Diagnosis: {initial_diagnosis}
Initial Reasoning: {initial_reasoning}

Self-Correction and Revised Reasoning:
"""
self_correction_prompt = MockPromptTemplate(
    template=self_correction_template,
    input_variables=["initial_diagnosis", "initial_reasoning"]
)

verification_template = """
Verify the following diagnosis and reasoning against the provided medical facts. Highlight any contradictions or strong supportive evidence. If there are contradictions, suggest a final refinement.

Diagnosis: {diagnosis}
Reasoning: {reasoning}
Medical Facts (retrieved from knowledge base): {medical_facts}

Verification and Final Refinement:
"""
verification_prompt = MockPromptTemplate(
    template=verification_template,
    input_variables=["diagnosis", "reasoning", "medical_facts"]
)


def run_diagnostic_pipeline(patient_data: PatientInput) -> DiagnosisOutput:
    st.subheader("Diagnostic Pipeline Execution")
    st.info("Starting diagnostic process...")

    # Step 1: Initial Retrieval based on symptoms for initial context
    initial_retrieval_query = f"{patient_data.symptoms} {patient_data.medical_history}"
    initial_medical_context = medical_retriever.get_relevant_documents(initial_retrieval_query)
    initial_medical_context_str = "\n".join(initial_medical_context)
    st.write(f"**Initial Medical Context Retrieved:** {initial_medical_context_str[:150]}...")

    # Step 2: Initial Diagnosis with Chain-of-Thought
    initial_prompt_formatted = initial_diagnosis_prompt.format(
        symptoms=patient_data.symptoms,
        medical_history=patient_data.medical_history,
        test_results=patient_data.test_results or "N/A",
        specific_questions=patient_data.specific_questions or "None",
        medical_context=initial_medical_context_str
    )
    st.write("**Generating Initial Diagnosis...**")
    initial_llm_output = mock_llm.invoke(initial_prompt_formatted)

    # Parse initial output (simplified parsing for mock LLM)
    initial_diagnosis_text = "N/A"
    initial_reasoning_text = "N/A"
    suggested_investigations_text = "N/A"

    if "Initial Diagnosis:" in initial_llm_output:
        initial_diagnosis_text = initial_llm_output.split("Initial Diagnosis:")[1].split("Suggested Investigations:")[0].strip()
        suggested_investigations_text = initial_llm_output.split("Suggested Investigations:")[1].strip()
        initial_reasoning_text = initial_llm_output.split("Chain-of-Thought:")[1].split("Initial Diagnosis:")[0].strip()
    else: # Fallback for simpler mock responses
        initial_diagnosis_text = initial_llm_output # Assume the whole output is the diagnosis
        initial_reasoning_text = initial_llm_output # Assume the whole output is reasoning

    st.write(f"**Initial Proposed Diagnosis:** {initial_diagnosis_text}")
    st.write(f"**Initial Reasoning:** {initial_reasoning_text}")

    # Step 3: Self-Correction Loop
    st.write("**Initiating Self-Correction...**")
    self_correction_prompt_formatted = self_correction_prompt.format(
        initial_diagnosis=initial_diagnosis_text,
        initial_reasoning=initial_reasoning_text
    )
    self_correction_output = mock_llm.invoke(self_correction_prompt_formatted)
    st.write(f"**Self-Correction Output:** {self_correction_output}")

    # Update diagnosis and reasoning based on self-correction (simplified parsing)
    refined_diagnosis_text = initial_diagnosis_text
    refined_reasoning_text = initial_reasoning_text
    if "Revised Diagnosis:" in self_correction_output:
        refined_diagnosis_text = self_correction_output.split("Revised Diagnosis:")[1].split("Further Investigation Priority:")[0].strip()
        refined_reasoning_text = self_correction_output.split("Self-Correction:")[1].split("Revised Diagnosis:")[0].strip()
    elif "Self-Correction:" in self_correction_output:
         refined_reasoning_text = self_correction_output.split("Self-Correction:")[1].strip()

    st.write(f"**Refined Diagnosis (after self-correction):** {refined_diagnosis_text}")
    st.write(f"**Refined Reasoning:** {refined_reasoning_text}")

    # Step 4: Knowledge-Based Verification
    st.write("**Performing Knowledge-Based Verification...**")
    # Use refined diagnosis/reasoning to query KB more precisely
    verification_query = f"{refined_diagnosis_text} {refined_reasoning_text}"
    verification_facts = medical_retriever.get_relevant_documents(verification_query)
    verification_facts_str = "\n".join(verification_facts)
    st.write(f"**Verification Facts Retrieved:** {verification_facts_str[:150]}...")

    verification_prompt_formatted = verification_prompt.format(
        diagnosis=refined_diagnosis_text,
        reasoning=refined_reasoning_text,
        medical_facts=verification_facts_str
    )
    verification_output = mock_llm.invoke(verification_prompt_formatted)
    st.write(f"**Verification Output:** {verification_output}")

    final_diagnosis = refined_diagnosis_text
    final_reasoning = refined_reasoning_text
    is_verified = False
    if "Final Confirmed Diagnosis:" in verification_output:
        final_diagnosis = verification_output.split("Final Confirmed Diagnosis:")[1].split("based on evidence")[0].strip()
        final_reasoning = verification_output.split("Verification:")[1].strip()
        is_verified = True
    elif "Verification:" in verification_output:
        final_reasoning = verification_output.split("Verification:")[1].strip()
        is_verified = "consistent" in verification_output.lower() # Simple heuristic for mock

    st.success("Diagnostic process completed.")

    return DiagnosisOutput(
        proposed_diagnoses=[final_diagnosis],
        reasoning_steps=final_reasoning,
        suggested_investigations=suggested_investigations_text.split(", ") if suggested_investigations_text != "N/A" else [],
        confidence_score=0.95 if is_verified else 0.7, # Mock confidence
        verified=is_verified
    )


# --- Streamlit UI --- 
st.set_page_config(layout="wide", page_title="Medical Diagnostic Assistant")
st.title("🩺 Medical Diagnostic Assistant with Enhanced Reasoning")
st.markdown("This AI assistant helps medical professionals by providing diagnostic support with Chain-of-Thought reasoning, self-correction, and knowledge-based verification.")

with st.sidebar:
    st.header("Patient Information Input")
    symptoms = st.text_area("Patient Symptoms (e.g., 'severe headache, nausea, sensitivity to light')", key="symptoms_input")
    medical_history = st.text_area("Medical History (e.g., 'no major illnesses, occasional migraines in the past')", key="history_input")
    test_results = st.text_area("Relevant Test Results (Optional) (e.g., 'Blood pressure 130/85 mmHg, no fever')", key="tests_input")
    specific_questions = st.text_area("Specific Diagnostic Questions (Optional) (e.g., 'Is this likely a migraine or something more serious?')", key="questions_input")

    st.markdown("--- This sidebar shows mock LLM interaction details ---")

if st.sidebar.button("Get Diagnosis", use_container_width=True):
    if not symptoms or not medical_history:
        st.error("Please provide at least patient symptoms and medical history.")
    else:
        patient_data = PatientInput(
            symptoms=symptoms,
            medical_history=medical_history,
            test_results=test_results if test_results else None,
            specific_questions=specific_questions if specific_questions else None,
        )

        with st.spinner("Analyzing patient data and generating diagnosis..."): # Using a spinner for better UX
            diagnosis_output = run_diagnostic_pipeline(patient_data)

        st.subheader("Final Diagnostic Report")
        st.write(f"**Proposed Diagnoses:** {', '.join(diagnosis_output.proposed_diagnoses)}")
        st.write(f"**Reasoning Steps:** {diagnosis_output.reasoning_steps}")
        st.write(f"**Suggested Investigations:** {', '.join(diagnosis_output.suggested_investigations) if diagnosis_output.suggested_investigations else 'None'}")
        st.write(f"**Confidence Score:** {diagnosis_output.confidence_score:.2f} (0-1 scale)")
        st.write(f"**Verification Status:** {'✅ Verified' if diagnosis_output.verified else '⚠️ Not fully verified (mock)'}")

        st.markdown("--- Disclaimer: This is a simulated diagnostic assistant and should not be used for actual medical advice. Always consult with a qualified medical professional. ---")
