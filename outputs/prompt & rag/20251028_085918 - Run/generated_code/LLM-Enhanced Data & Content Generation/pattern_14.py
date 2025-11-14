import streamlit as st
import pandas as pd
import numpy as np
# from langchain.chains import RetrievalQA
# from langchain.vectorstores import Chroma
# from langchain.embeddings import SentenceTransformerEmbeddings
# from transformers import pipeline # For local LLM or fine-tuned model
# import openai # For OpenAI API
import time

# --- 1. Mock Data and Setup ---

# Simulate EHR data
mock_ehr_data = {
    "patient_101": {
        "name": "Alice Smith",
        "age": 45,
        "gender": "Female",
        "history": "Hypertension, Type 2 Diabetes (controlled), no known allergies.",
        "lab_results": "Blood pressure: 130/85, HbA1c: 6.5%, Cholesterol: 180 mg/dL.",
        "medications": ["Lisinopril", "Metformin"],
    },
    "patient_102": {
        "name": "Bob Johnson",
        "age": 60,
        "gender": "Male",
        "history": "Coronary Artery Disease, previous MI (5 years ago), smoker (quit 2 years ago).",
        "lab_results": "ECG: Old inferior infarct, Troponin: normal, HDL: 40 mg/dL.",
        "medications": ["Aspirin", "Atorvastatin", "Metoprolol"],
    },
}

# Simulate medical literature (simplified)
mock_medical_literature = [
    "Hypertension treatment guidelines often recommend ACE inhibitors or ARBs.",
    "Type 2 Diabetes management involves lifestyle changes and oral hypoglycemic agents like Metformin.",
    "Symptoms of myocardial infarction include chest pain, shortness of breath, and radiating pain to the arm.",
    "Diagnostic criteria for pneumonia include fever, cough, and infiltrates on chest X-ray.",
    "Common treatments for acute bronchitis include rest, fluids, and bronchodilators if wheezing is present.",
    "Differential diagnosis for headache includes tension headache, migraine, and cluster headache.",
]

class MockVectorStore:
    def __init__(self, texts):
        self.texts = texts

    def similarity_search(self, query: str, k: int = 2):
        # A very simple mock similarity search based on keyword presence
        query_words = set(query.lower().split())
        scores = []
        for text in self.texts:
            text_words = set(text.lower().split())
            common_words = len(query_words.intersection(text_words))
            scores.append((common_words, text))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [{"page_content": item[1]} for item in scores[:k]]

mock_vectorstore = MockVectorStore(mock_medical_literature)


# --- 2. LLM Integration Layer (Placeholder) ---
def call_llm(prompt: str, max_tokens: int = 200) -> str:
    """
    Placeholder function to simulate an LLM call.
    In a real app, this would use OpenAI API, Hugging Face transformers, etc.
    """
    st.info(f"LLM Prompt (simulated): {prompt[:200]}...") # Show part of the prompt
    time.sleep(1) # Simulate a delayed response

    if "diagnose" in prompt.lower():
        return "Simulated LLM Diagnosis: Based on the symptoms and patient history, consider 'Influenza' or 'Acute Bronchitis'. Further tests like a viral panel could confirm."
    elif "synthetic patient data" in prompt.lower():
        return "Simulated LLM Output: Synthetic Patient Case: A 32-year-old female presents with sudden onset severe headache, photophobia, and neck stiffness. No fever. Past medical history: migraines. Lab findings: unremarkable. Impression: Suspect 'Meningitis' or 'Severe Migraine'."
    elif "treatment plan" in prompt.lower():
        return "Simulated LLM Output: Personalized Treatment Plan: For Influenza, recommend rest, hydration, symptomatic relief (paracetamol/ibuprofen), and potentially antiviral medication if criteria met. Monitor for secondary bacterial infections."
    else:
        return "Simulated LLM Response: This is a generic LLM response to your query. In a real scenario, the LLM would provide a more specific answer based on the prompt."

# --- 3. Data Retrieval Module ---

def retrieve_ehr_data(patient_id: str) -> dict:
    """Retrieves mock EHR data for a given patient ID."""
    return mock_ehr_data.get(patient_id, {})

def search_medical_literature(query: str, k: int = 2) -> list[str]:
    """
    Performs a simulated RAG-like search on medical literature.
    In a real app, this would use vectorstore.similarity_search.
    """
    results = mock_vectorstore.similarity_search(query, k=k)
    return [doc["page_content"] for doc in results]

# --- 4. Diagnostic Reasoning Engine ---

def get_diagnostic_suggestions(patient_info: dict, symptoms: str) -> str:
    """
    Uses the LLM to provide diagnostic suggestions based on patient info and symptoms.
    """
    ehr_text = "\n".join([f"{k}: {v}" for k, v in patient_info.items()]) if patient_info else "No EHR data available."
    literature_context = "\n".join(search_medical_literature(symptoms + " diagnosis"))

    prompt = (
        f"Patient Information:\n{ehr_text}\n\n"
        f"Symptoms: {symptoms}\n\n"
        f"Relevant Medical Literature:\n{literature_context}\n\n"
        f"Based on the above information, provide potential diagnoses and reasoning. "
        f"Also, suggest any further tests if needed."
    )
    return call_llm(prompt)

# --- 5. Synthetic Data Generation Module ---

def generate_synthetic_patient_data(condition: str, age_range: str = "adult", severity: str = "moderate") -> str:
    """
    Generates synthetic patient case data using the LLM.
    """
    prompt = (
        f"Generate a detailed synthetic patient case for a {age_range} individual with {condition} "
        f"of {severity} severity. Include patient demographics, chief complaint, history of present illness, "
        f"relevant past medical history, family history, social history, a brief physical examination finding, "
        f"and potential initial lab/imaging results. Ensure the data is realistic and suitable for medical training."
    )
    return call_llm(prompt)

# --- 6. Personalized Treatment Recommendation Module ---

def get_personalized_treatment_plan(diagnosis: str, patient_info: dict, feedback: str = "") -> str:
    """
    Generates personalized treatment recommendations using the LLM.
    """
    ehr_text = "\n".join([f"{k}: {v}" for k, v in patient_info.items()]) if patient_info else "No EHR data available."
    literature_context = "\n".join(search_medical_literature(diagnosis + " treatment guidelines"))

    prompt = (
        f"Diagnosis: {diagnosis}\n\n"
        f"Patient Information:\n{ehr_text}\n\n"
        f"Relevant Medical Literature on Treatment:\n{literature_context}\n\n"
        f"User Feedback on previous treatment (if any): {feedback}\n\n"
        f"Based on the above, provide a personalized treatment plan. "
        f"Consider both medical guidelines and patient context. Include dosage suggestions where appropriate."
    )
    return call_llm(prompt)

# --- 7. Feedback and Learning Loop (Conceptual) ---
# In a real application, this would involve storing feedback and potentially
# using it for fine-tuning or prompt refinement. For this prototype,
# feedback is just an input to the treatment recommendation.

# --- Streamlit UI ---

st.set_page_config(layout="wide", page_title="Intelligent Diagnostic Assistant")
st.title("👨‍⚕️ Intelligent Diagnostic Assistant")
st.markdown("---")

# Sidebar for patient selection
st.sidebar.header("Patient Information")
patient_id_input = st.sidebar.text_input("Enter Patient ID (e.g., patient_101)", "patient_101")
current_patient_info = {}
if patient_id_input:
    current_patient_info = retrieve_ehr_data(patient_id_input)
    if current_patient_info:
        st.sidebar.success(f"Loaded data for: {current_patient_info.get('name', 'N/A')}")
        st.sidebar.json(current_patient_info)
    else:
        st.sidebar.warning("Patient ID not found. Using empty patient profile.")

# Main content tabs
tab1, tab2, tab3 = st.tabs(["Diagnostic Assistant", "Synthetic Data Generator", "Treatment Planner"])

with tab1:
    st.header("Diagnostic Assistant")
    st.write("Input patient symptoms to get diagnostic suggestions and relevant medical literature.")

    symptoms_input = st.text_area("Describe patient symptoms:",
                                  "Patient presents with sudden onset chest pain radiating to the left arm, shortness of breath, and sweating.")

    if st.button("Get Diagnosis"):
        if symptoms_input:
            with st.spinner("Generating diagnostic suggestions..."):
                st.subheader("Retrieved Medical Literature")
                retrieved_docs = search_medical_literature(symptoms_input)
                for i, doc in enumerate(retrieved_docs):
                    st.write(f"**Doc {i+1}:** {doc}")

                st.subheader("Diagnostic Suggestions")
                diagnosis_output = get_diagnostic_suggestions(current_patient_info, symptoms_input)
                st.success(diagnosis_output)
        else:
            st.warning("Please enter symptoms to get a diagnosis.")

with tab2:
    st.header("Synthetic Patient Data Generator")
    st.write("Generate realistic synthetic patient cases for training and education.")

    synthetic_condition = st.text_input("Condition for synthetic data:", "Pneumonia")
    synthetic_age_range = st.selectbox("Age Range:", ["adult", "pediatric", "elderly"], index=0)
    synthetic_severity = st.selectbox("Severity:", ["mild", "moderate", "severe"], index=1)

    if st.button("Generate Synthetic Case"):
        if synthetic_condition:
            with st.spinner("Generating synthetic patient data..."):
                synthetic_data = generate_synthetic_patient_data(synthetic_condition, synthetic_age_range, synthetic_severity)
                st.success(synthetic_data)
        else:
            st.warning("Please enter a condition to generate synthetic data.")

with tab3:
    st.header("Personalized Treatment Planner")
    st.write("Get personalized treatment recommendations based on diagnosis and patient context.")

    treatment_diagnosis = st.text_input("Confirmed Diagnosis:", "Influenza")
    treatment_feedback = st.text_area("Patient feedback or previous treatment efficacy:",
                                      "Patient prefers oral medication. Allergic to penicillin.")

    if st.button("Get Treatment Plan"):
        if treatment_diagnosis:
            with st.spinner("Generating personalized treatment plan..."):
                treatment_plan = get_personalized_treatment_plan(treatment_diagnosis, current_patient_info, treatment_feedback)
                st.success(treatment_plan)
        else:
            st.warning("Please enter a diagnosis to get a treatment plan.")

st.markdown("---")
st.caption("Disclaimer: This is a prototype intelligent diagnostic assistant and should not be used for actual medical decisions. Always consult with qualified healthcare professionals.")
