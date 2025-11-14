import streamlit as st
import time

# --- Simulated Backend/LLM Interaction --- #
def simulate_llm_diagnosis(symptoms, history, lab_results):
    st.session_state.processing = True
    with st.spinner("Analyzing patient data and consulting medical knowledge bases..."):
        time.sleep(3)  # Simulate LLM processing time

        # Simulate differential diagnosis
        if "fever" in symptoms.lower() and "cough" in symptoms.lower() and "fatigue" in symptoms.lower():
            diagnosis = "Probable Viral Infection (e.g., Influenza, Common Cold)"
            reasoning = "Based on the triad of fever, cough, and fatigue, a viral respiratory infection is highly suspected. Further testing (e.g., flu swab) may be warranted for definitive diagnosis."
            confidence = "High (85%)"
            treatment = "Rest, hydration, symptomatic relief (e.g., antipyretics for fever, cough suppressants). Advise monitoring for worsening symptoms or secondary bacterial infection."
        elif "chest pain" in symptoms.lower() and "shortness of breath" in symptoms.lower():
            diagnosis = "Possible Cardiac Event or Pulmonary Embolism"
            reasoning = "Acute chest pain with shortness of breath is a critical presentation requiring immediate investigation for cardiac ischemia, pulmonary embolism, or other life-threatening conditions. ECG, cardiac enzymes, and imaging are crucial."
            confidence = "Moderate to High (70-80% based on symptoms alone, awaiting diagnostics)"
            treatment = "Immediate emergency medical evaluation, oxygen, and specific interventions based on confirmed diagnosis (e.g., antiplatelets, anticoagulants, thrombolytics)."
        else:
            diagnosis = "Further investigation needed / Non-specific symptoms"
            reasoning = "The provided symptoms are too general or require more specific details (e.g., duration, severity, associated factors) to narrow down a differential diagnosis effectively. Integration with comprehensive EHR data and specific lab results would improve accuracy."
            confidence = "Low (30-50% without more data)"
            treatment = "Recommend detailed history taking, physical examination, and targeted diagnostic tests based on initial clinical suspicion."

        st.session_state.diagnosis_result = {
            "diagnosis": diagnosis,
            "reasoning": reasoning,
            "confidence": confidence,
            "treatment": treatment
        }
    st.session_state.processing = False
    st.rerun()

# --- Streamlit UI --- #
st.set_page_config(page_title="Agentic CDSS", layout="wide")
st.title("🧠 Agentic Clinical Decision Support System")
st.markdown("--- Generative AI for interpretable, reliable, and collaborative clinical insights ---")

# Input Forms
st.header("Patient Information")

with st.form("patient_form", clear_on_submit=False):
    col1, col2 = st.columns(2)
    with col1:
        patient_name = st.text_input("Patient Name", key="patient_name")
        age = st.number_input("Age", min_value=0, max_value=120, key="age")
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="gender")
        st.text_input("Medical Record Number (MRN)", key="mrn")

    st.subheader("Presenting Complaints & History")
    symptoms = st.text_area("Describe symptoms (e.g., 'fever for 3 days, cough, fatigue')", height=150, key="symptoms")
    medical_history = st.text_area("Relevant medical history (e.g., 'Hypertension, Diabetes')", height=100, key="history")
    lab_results = st.text_area("Relevant lab results (e.g., 'WBC 12.0, CRP 50 mg/L')", height=100, key="lab_results")

    submitted = st.form_submit_button("Get Diagnosis & Treatment Recommendation")

    if submitted:
        if not symptoms:
            st.warning("Please describe the patient's symptoms.")
        else:
            # Initialize session state for results if not already present
            if "diagnosis_result" not in st.session_state:
                st.session_state.diagnosis_result = None
            if "processing" not in st.session_state:
                st.session_state.processing = False

            # Simulate LLM call
            simulate_llm_diagnosis(symptoms, medical_history, lab_results)

# Display Results
if st.session_state.get("diagnosis_result") and not st.session_state.get("processing"):
    st.header("AI-Generated Insights")
    result = st.session_state.diagnosis_result

    st.subheader(f"Differential Diagnosis: :blue[{result['diagnosis']}]")

    # Progressive Information Disclosure
    with st.expander("View Treatment Plan"): # Collapsed by default
        st.success(f"**Recommended Treatment Plan:** {result['treatment']}")

    with st.expander("View AI Reasoning & Confidence", expanded=True): # Expanded by default for transparency
        st.info(f"**Reasoning Path:** {result['reasoning']}")
        st.metric(label="AI Confidence", value=result['confidence'])
        st.caption("Confidence scores are estimates; always cross-verify with clinical judgment.")

    st.markdown("--- ")
    st.subheader("Feedback & Data Collection")
    st.write("Help us improve the system. Was this recommendation helpful?")
    col_fb1, col_fb2 = st.columns([0.1, 0.9])
    with col_fb1:
        if st.button("👍 Yes"):
            st.success("Thank you for your feedback!")
        if st.button("👎 No"):
            st.error("We appreciate your honest feedback. Please provide more details below.")
    with col_fb2:
        st.text_area("Optional: Provide more details or correct information here.", height=70, key="feedback_details")
        if st.button("Submit Feedback Details"):
            if st.session_state.feedback_details:
                st.success("Details submitted. Thank you for contributing to system improvement!")
            else:
                st.warning("Please enter some details before submitting.")


# Initial state for session variables
if "diagnosis_result" not in st.session_state:
    st.session_state.diagnosis_result = None
if "processing" not in st.session_state:
    st.session_state.processing = False
