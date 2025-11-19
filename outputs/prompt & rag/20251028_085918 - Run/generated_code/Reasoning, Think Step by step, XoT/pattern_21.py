import streamlit as st
import pandas as pd
from pydantic import BaseModel, Field
from typing import List, Optional
import time

# Pydantic Models for Data Validation
class PatientInput(BaseModel):
    patient_id: str
    symptoms: str
    medical_history: str
    lab_results: Optional[str] = None

class DiagnosisOutput(BaseModel):
    diagnosis: str
    confidence_score: float
    reasoning_steps: List[str]
    references: List[str]
    verified: bool

# --- Mock LLM and RAG Components --- #

# Mock LLM for Reasoning Engine
class MockLLM:
    def __init__(self, model_name="mock-llm"):
        self.model_name = model_name

    def generate_reasoning(self, prompt: str) -> dict:
        st.info("LLM is generating initial reasoning...")
        time.sleep(2)  # Simulate LLM processing time
        
        # Simple rule-based mock for demonstration
        if "fever" in prompt.lower() and "cough" in prompt.lower() and "fatigue" in prompt.lower():
            diagnosis = "Influenza"
            reasoning = [
                "Patient presents with common influenza symptoms: fever, cough, and fatigue.",
                "Influenza is a common respiratory viral infection.",
                "Further lab tests (e.g., flu swab) would confirm."
            ]
        elif "chest pain" in prompt.lower() and "shortness of breath" in prompt.lower():
            diagnosis = "Potential Cardiac Event"
            reasoning = [
                "Symptoms like chest pain and shortness of breath are red flags for cardiac issues.",
                "Immediate medical attention and ECG are recommended.",
                "Rule out conditions like angina, myocardial infarction, or pulmonary embolism."
            ]
        else:
            diagnosis = "General Illness / Unknown Cause"
            reasoning = [
                "Based on provided symptoms, a definitive diagnosis is not immediately apparent.",
                "Further investigation and specialist consultation may be required."
            ]
            
        return {"diagnosis": diagnosis, "reasoning_steps": reasoning}

# Mock Medical Knowledge Base (RAG System)
def mock_rag_system(query: str) -> List[str]:
    st.info(f"Querying medical knowledge base for: '{query}'...")
    time.sleep(1) # Simulate RAG lookup time
    
    # Simple mock responses based on keywords
    if "influenza" in query.lower():
        return [
            "Influenza (flu) is a contagious respiratory illness caused by influenza viruses.",
            "Symptoms include fever, cough, sore throat, muscle aches, and fatigue.",
            "Vaccination is the most effective way to prevent influenza."
        ]
    elif "cardiac event" in query.lower() or "chest pain" in query.lower():
        return [
            "Chest pain can be a symptom of various conditions, including heart attack, angina, or heartburn.",
            "Seek immediate medical help for sudden, severe chest pain.",
            "Electrocardiogram (ECG) and blood tests are crucial for diagnosing cardiac events."
        ]
    else:
        return ["No specific medical references found for this query in the mock database."]

# Mock Verifier (Self-Consistency / Secondary LLM)
def mock_verifier(primary_reasoning: List[str], rag_references: List[str]) -> bool:
    st.info("Verifying reasoning for consistency and accuracy...")
    time.sleep(1.5) # Simulate verification time
    
    # Simple verification logic: checks for presence of relevant references
    # In a real system, this would involve NLP for semantic comparison or another LLM
    
    is_consistent = True
    if "No specific medical references found" in rag_references[0]:
        is_consistent = False # If RAG found nothing, likely less verifiable
    
    # More sophisticated logic would analyze if reasoning steps are supported by references
    # For this mock, we'll assume a basic check passes if RAG returns *something* useful.
    
    if not is_consistent:
        st.warning("Verification failed: Reasoning may lack strong evidential support or consistency.")
    else:
        st.success("Verification successful: Reasoning appears consistent with available medical knowledge.")
        
    return is_consistent

# --- Streamlit UI --- #
st.set_page_config(layout="wide", page_title="MedVerify AI Diagnostic Assistant")
st.title("🩺 MedVerify AI: Diagnostic Assistant")

st.markdown(
    "This AI-powered assistant uses Structured and Verified Reasoning (SVR) to provide diagnostic support. "
    "Input patient data, and the system will generate a diagnosis with transparent, verified reasoning."
)

# Input Form
with st.form("patient_input_form"):
    st.header("Patient Information")
    patient_id = st.text_input("Patient ID", "P12345")
    symptoms = st.text_area("Symptoms (e.g., 'fever, cough, severe fatigue')", "")
    medical_history = st.text_area("Medical History (e.g., 'Hypertension, No known allergies')", "")
    lab_results = st.text_area("Lab Results (Optional, e.g., 'WBC: 12.5, CRP: 50mg/L')", "")

    submitted = st.form_submit_button("Generate Diagnosis")

    if submitted:
        try:
            patient_data = PatientInput(
                patient_id=patient_id,
                symptoms=symptoms,
                medical_history=medical_history,
                lab_results=lab_results if lab_results else None,
            )

            st.subheader("Processing Request...")
            st.json(patient_data.model_dump())

            # --- Workflow Orchestration --- #
            
            # 1. Reasoning Engine (Primary LLM)
            llm = MockLLM()
            prompt_for_llm = (
                f"Analyze the following patient data to provide a potential diagnosis and step-by-step reasoning. "
                f"Patient ID: {patient_data.patient_id}\n"
                f"Symptoms: {patient_data.symptoms}\n"
                f"Medical History: {patient_data.medical_history}\n"
                f"Lab Results: {patient_data.lab_results if patient_data.lab_results else 'N/A'}\n"
                f"Provide a diagnosis and a detailed Chain-of-Thought reasoning process."
            )
            
            llm_output = llm.generate_reasoning(prompt_for_llm)
            initial_diagnosis = llm_output["diagnosis"]
            initial_reasoning_steps = llm_output["reasoning_steps"]

            st.subheader("Initial LLM Diagnosis & Reasoning:")
            st.write(f"**Diagnosis:** {initial_diagnosis}")
            for i, step in enumerate(initial_reasoning_steps):
                st.write(f"- {step}")
            
            # 2. Verification Layer - Medical Knowledge Base (RAG)
            rag_query = f"Medical information on {initial_diagnosis} and symptoms like {patient_data.symptoms}"
            rag_references = mock_rag_system(rag_query)
            
            st.subheader("Medical Knowledge Base References (RAG):")
            for ref in rag_references:
                st.write(f"- {ref}")

            # 3. Verification Layer - Self-Consistency / Secondary LLM Verifier
            is_verified = mock_verifier(initial_reasoning_steps, rag_references)

            # 4. Output Layer
            final_diagnosis_output = DiagnosisOutput(
                diagnosis=initial_diagnosis,
                confidence_score=0.85 if is_verified else 0.60, # Mock confidence based on verification
                reasoning_steps=initial_reasoning_steps,
                references=rag_references,
                verified=is_verified,
            )

            st.subheader("\n--- Final Verified Diagnosis ---:")
            st.success(f"**Diagnosis: {final_diagnosis_output.diagnosis}**")
            st.info(f"**Confidence Score:** {final_diagnosis_output.confidence_score:.2f}")
            st.write(f"**Verification Status:** {'✅ Verified' if final_diagnosis_output.verified else '❌ Not Fully Verified'}")
            st.markdown("**Detailed Reasoning:**")
            for i, step in enumerate(final_diagnosis_output.reasoning_steps):
                st.write(f"_{i+1}. {step}_")
            st.markdown("**Supporting References:**")
            for ref in final_diagnosis_output.references:
                st.write(f"- _{ref}_")

        except Exception as e:
            st.error(f"An error occurred: {e}")